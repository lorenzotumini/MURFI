"""Search branching and validation-only output checks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot import select_beam


def node(bits, recall, loss, retain=0.95):
    return {'trace': [{'parameter': 'p', 'index': i, 'bit': b} for i, b in bits],
            'validation': {'forget_accuracy': recall, 'retain_accuracy': retain},
            'objective': loss}


class BeamSearchTests(unittest.TestCase):
    def test_equivalent_flip_orders_share_one_beam_slot(self):
        first = node([(0, 1), (2, 3)], 0.2, 2.0)
        duplicate = node([(2, 3), (0, 1)], 0.2, 2.0)
        other = node([(4, 5), (6, 7)], 0.1, 1.0)
        self.assertEqual(select_beam([first, duplicate, other], 2), [first, other])

    def test_recall_then_loss_rank_alternative_paths(self):
        good_loss = node([(0, 0)], 0.1, 1.0)
        recovered = node([(1, 0)], 0.2, 3.0)
        poor_loss = node([(2, 0)], 0.1, 2.0)
        self.assertEqual(select_beam([poor_loss, good_loss, recovered], 2),
                         [recovered, good_loss])

    def test_beam_smoke_audits_mutations_without_test_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'pilot.py'),
                            '--stage', 'smoke', '--device', 'cpu', '--root', directory,
                            '--algorithm', 'beam', '--validation-only'],
                           check=True, capture_output=True, text=True, timeout=60)
            root = Path(directory) / 'search'
            results = json.loads((root / 'results.json').read_text())
            self.assertEqual(results['evaluation_split'], 'selection')
            for result in results['targeted'].values():
                self.assertNotIn('test', result)
                self.assertTrue(result['validation']['valid'])
            diagnostics = json.loads((root / 'retrained-diagnostics.json').read_text())
            self.assertGreater(diagnostics['depth_reached'], 0)
            self.assertEqual(diagnostics['algorithm'], 'beam')
            for trial in results['random']:
                self.assertNotIn('test', trial)
                self.assertTrue(trial['validation']['valid'])


class FasterSearchTests(unittest.TestCase):
    def test_bit_scores_match_enumerated_storage_flips(self):
        import math
        import torch
        from pilot import bit_change_scores, flipped_float
        weights = torch.tensor([0.0, 1.0, -1.0, 0.125, -0.02], dtype=torch.float32)
        gradients = torch.tensor([1.0, -2.0, 3.0, 0.0, -0.4])
        before = weights.clone()
        expected = []
        for w, g in zip(weights.tolist(), gradients.tolist()):
            benefits = [0.0]
            for bit in range(32):
                changed = flipped_float(w, bit)
                if math.isfinite(changed) and abs(changed) <= 2.0:
                    benefits.append(-g * (changed - w))
            expected.append(max(benefits))
        torch.testing.assert_close(bit_change_scores(weights, gradients, 2.0), torch.tensor(expected))
        self.assertTrue(torch.equal(before, weights))

    def test_cached_batches_preserve_samples_order_and_values(self):
        import torch
        from pilot import cached_batches
        data = [(torch.arange(6).reshape(2, 3), torch.tensor([1, 2])),
                (torch.ones(1, 3), torch.tensor([3]))]
        cached = cached_batches(data, 'cpu')
        for actual, expected in zip(cached, data):
            for a, e in zip(actual, expected):
                self.assertTrue(torch.equal(a, e))
        self.assertEqual(len(cached), len(data))

    def test_bounded_screened_search_never_exceeds_candidate_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'pilot.py'),
                            '--stage', 'smoke', '--device', 'cpu', '--root', directory,
                            '--algorithm', 'beam', '--validation-only', '--cache-eval',
                            '--ranking', 'bit-change', '--screen-per-class', '1',
                            '--max-candidate-evals', '1', '--max-full-validations', '1',
                            '--validation-candidates', '1'],
                           check=True, capture_output=True, text=True, timeout=60)
            root = Path(directory) / 'search'
            for model in ['unlearned', 'retrained']:
                d = json.loads((root / (model + '-diagnostics.json')).read_text())
                self.assertLessEqual(d['candidate_evals'], 1)
                self.assertLessEqual(d['full_validations'], 1)
                self.assertTrue(d['search_incomplete'])
                self.assertTrue(d['stop_reason'].endswith('_limit'))
            result = json.loads((root / 'results.json').read_text())
            self.assertEqual(result['evaluation_split'], 'selection')
            for target in result['targeted'].values():
                self.assertNotIn('test', target)


class FrozenEvaluationTests(unittest.TestCase):
    def test_evaluation_reuses_exact_selected_states_and_refuses_second_look(self):
        import hashlib
        script = str(Path(__file__).resolve().parents[1] / 'pilot.py')
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, script, '--stage', 'smoke', '--device', 'cpu',
                            '--root', directory, '--validation-only'],
                           check=True, capture_output=True, text=True, timeout=60)
            search = Path(directory) / 'search'
            paths = [search / (name + '-selected.pt') for name in ['unlearned', 'retrained']]
            paths += [search / 'config.json', search / 'results.json']
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            command = [sys.executable, script, '--stage', 'evaluate', '--device', 'cpu',
                       '--root', directory, '--search-name', 'search', '--cache-eval']
            evaluated = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
            self.assertNotIn('depth=', evaluated.stdout)
            result = json.loads((search / 'frozen-test-results.json').read_text())
            self.assertEqual(result['evaluation_split'], 'test')
            for model in result['targeted'].values():
                self.assertTrue(model['test']['valid'])
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
            manifest = json.loads((search / 'frozen-test-manifest.json').read_text())
            self.assertEqual(manifest['seed'], 0)
            self.assertTrue(manifest['sha256'])
            repeated = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn('already started', repeated.stderr)
