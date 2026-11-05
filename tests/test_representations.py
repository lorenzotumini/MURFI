"""Scientific invariants, data integrity, and a CPU end-to-end artifact check."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import representation_analysis as rep
import representation_metrics as metrics
import representation_resnet as resnet


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12)
        self.x = self.rng.normal(size=(120, 9))
        self.q, _ = np.linalg.qr(self.rng.normal(size=(9, 9)))

    def test_cka_orthogonal_scale_translation_invariance(self):
        transformed = 3 * self.x @ self.q + 7
        for debiased in (False, True):
            self.assertAlmostEqual(metrics.cka(self.x, transformed, debiased), 1, places=12)
        self.assertLess(metrics.cka(self.x, self.x[self.rng.permutation(len(self.x))]), 0.2)

    def test_unbiased_hsic_corrects_high_dimension_small_sample_bias(self):
        x, y = self.rng.normal(size=(30, 2000)), self.rng.normal(size=(30, 2000))
        self.assertGreater(metrics.cka(x, y), 0.95)
        self.assertLess(abs(metrics.cka(x, y, debiased=True)), 0.15)

    def test_collapsed_and_tiny_groups_are_undefined(self):
        self.assertIsNone(metrics.cka(self.x, np.ones_like(self.x)))
        self.assertIsNone(metrics.cka(self.x[:3], self.x[:3], debiased=True))
        self.assertIsNone(metrics.cka(self.x[:1], self.x[:1]))

    def test_neighbor_overlap_matches_rotation_and_excludes_self(self):
        a, _ = metrics.cosine_neighbors(self.x, 10, block_size=7)
        b, _ = metrics.cosine_neighbors(self.x @ self.q, 10, block_size=31)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.any(a == np.arange(len(a))[:, None]))
        np.testing.assert_array_equal(metrics.neighbor_overlap(a, b), np.ones(len(a)))

    def test_neighbors_match_brute_force_and_zero_rows_are_excluded(self):
        z = self.x / np.linalg.norm(self.x, axis=1, keepdims=True)
        sim = z @ z.T
        np.fill_diagonal(sim, -np.inf)
        expected = np.argsort(-sim, axis=1, kind='stable')[:, :5]
        actual, _ = metrics.cosine_neighbors(self.x, 5)
        np.testing.assert_array_equal(actual, expected)
        zeros, diag = metrics.cosine_neighbors(np.zeros((10, 4)), 5)
        self.assertEqual(diag['zero_vectors'], 10)
        self.assertTrue(np.isnan(metrics.neighbor_overlap(zeros, zeros)).all())

    def test_alignment_uses_only_calibration(self):
        cal = np.arange(len(self.x)) < 40
        transformed = self.x @ self.q + 13
        aligned, _ = metrics.align_to_reference(
            {'original': self.x, 'unlearned': transformed}, cal
        )
        np.testing.assert_allclose(aligned['original'], aligned['unlearned'], atol=1e-12)
        transformed[~cal] += 50
        altered, _ = metrics.align_to_reference(
            {'original': self.x, 'unlearned': transformed}, cal
        )
        np.testing.assert_allclose(
            aligned['unlearned'][cal], altered['unlearned'][cal], atol=1e-12
        )
        self.assertGreater(
            np.linalg.norm(altered['unlearned'][~cal] - aligned['original'][~cal]), 10
        )


class BundleTests(unittest.TestCase):
    def test_full_synthetic_pipeline_and_nonoverwriting_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / 'features.npz'
            args = argparse.Namespace(seed=0, output=bundle)
            rep.extract_synthetic(args)
            meta, arrays = rep.load_bundle(bundle)
            self.assertEqual(meta['source'], 'synthetic')
            with self.assertRaises(FileExistsError):
                rep.extract_synthetic(args)
            result = rep.analyze(bundle, root / 'analysis', ks=(5, 10), null_repeats=2)
            for row in result['comparisons']:
                if row['pair'] == 'original--unlearned':
                    self.assertAlmostEqual(row['cka'], 1, places=10)
                    self.assertAlmostEqual(row['cka_debiased'], 1, places=10)
                    self.assertEqual(row['knn_10'], 1)
            for name in (
                'metrics.json',
                'comparisons.csv',
                'behavior.csv',
                'REPORT.md',
                'pca.png',
                'similarity_by_class.png',
                'similarity_change.png',
                'relations.csv',
            ):
                self.assertTrue((root / 'analysis' / name).is_file(), name)
            json.loads((root / 'analysis' / 'metrics.json').read_text())
            self.assertFalse((root / 'analysis' / 'representation_metrics.png').exists())
            self.assertEqual(result['relations']['modes'], ['raw', 'centered'])
            # Future ResNet runs have multiple layers and use the combined figure.
            meta['layers'].append('second')
            for model in meta['models']:
                arrays[rep.feature_key(model, 'second')] = arrays[
                    rep.feature_key(model, 'hidden')
                ][:, :8]
            rep.save_bundle(root / 'multilayer.npz', meta, arrays)
            multi = rep.analyze(
                root / 'multilayer.npz',
                root / 'multilayer',
                ks=(5,),
                null_repeats=1,
                layer='hidden',
            )
            self.assertEqual(multi['relations']['layer'], 'hidden')
            self.assertTrue((root / 'multilayer' / 'representation_metrics.png').exists())
            with self.assertRaises(FileExistsError):
                rep.analyze(bundle, root / 'analysis')
            arrays['sample_ids'][1] = arrays['sample_ids'][0]
            with self.assertRaisesRegex(ValueError, 'unique'):
                rep.validate_bundle(meta, arrays)

    def test_bad_rows_nonfinite_and_forget_calibration_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'features.npz'
            rep.extract_synthetic(argparse.Namespace(seed=0, output=bundle))
            meta, arrays = rep.load_bundle(bundle)
            key = rep.feature_key('original', 'hidden')
            arrays[key][0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                rep.validate_bundle(meta, arrays)
            arrays[key][0, 0] = 0
            arrays['partitions'][0] = 'calibration'
            with self.assertRaisesRegex(ValueError, 'retained'):
                rep.validate_bundle(meta, arrays)

    def test_archived_digits_extract_exact_activations_and_disjoint_sets(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                seed=0,
                forget_class=0,
                probe_seed=2026,
                per_class=10,
                evaluation='discovery',
                output=Path(directory) / 'digits.npz',
                reference=root / 'reports/targeted-flips/cpu/digits-main.json.gz',
                fixtures=root / 'reports/random-flips',
            )
            rep.extract_digits(args)
            meta, arrays = rep.load_bundle(args.output)
            from sklearn.datasets import load_digits

            x, _ = load_digits(return_X_y=True)
            x = (x / 16).astype('float32')
            ids = [int(i.split(':')[1]) for i in arrays['sample_ids']]
            self.assertEqual(len(ids), len(set(ids)))
            with np.load(args.fixtures / 'digit0-seed0.npz', allow_pickle=False) as fixture:
                expected = np.maximum(
                    x[ids] @ fixture['original_w1'] + fixture['original_b1'], 0
                )
            np.testing.assert_array_equal(
                arrays[rep.feature_key('original', 'hidden')], expected
            )
            self.assertEqual(meta['evaluation_split'], 'discovery')

    def test_resnet_split_overlap_rejected_without_torch(self):
        resnet.validate_splits({'train': [0, 1], 'search': [2], 'select': [3]}, total=4)
        with self.assertRaises(ValueError):
            resnet.validate_splits({'train': [0, 1], 'search': [1], 'select': [3]}, total=4)


class ClassRelationTests(unittest.TestCase):
    def test_centering_uses_calibration_and_removes_common_offset(self):
        rng = np.random.default_rng(8)
        h = rng.normal(size=(60, 5))
        arrays = {
            'labels': np.tile([1, 2, 3], 20),
            'partitions': np.array(['calibration'] * 30 + ['heldout'] * 30),
            rep.feature_key('original', 'hidden'): h,
            rep.feature_key('unlearned', 'hidden'): h + 10,
        }
        meta = {'models': ['original', 'unlearned'], 'layers': ['hidden'], 'forget_class': 0}
        result = metrics.compute_relations(meta, arrays)
        changes = {r['mode']: r for r in result['changes']}
        self.assertGreater(changes['raw']['retain_cross_mean_change'], 0.5)
        self.assertLess(changes['centered']['retain_cross_mean_absolute_change'], 1e-12)
        arrays[rep.feature_key('unlearned', 'hidden')][30:] += 10
        changed = metrics.compute_relations(meta, arrays)
        centered = [r for r in changed['changes'] if r['mode'] == 'centered'][0]
        self.assertGreater(centered['retain_cross_mean_change'], 0.5)

    def test_diagonal_excludes_self_and_cross_blocks_use_all_pairs(self):
        x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        mean, counts, zeros = metrics.class_cosines(x, np.array([0, 0, 1, 1]), [0, 1])
        np.testing.assert_allclose(mean, [[0, 1 / np.sqrt(2)], [1 / np.sqrt(2), 1]])
        np.testing.assert_array_equal(counts, [[1, 4], [4, 1]])
        self.assertEqual(zeros, 0)

    def test_rotation_and_per_example_positive_scale_preserve_summary(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(30, 8))
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        yy = np.arange(30) % 3
        original = metrics.class_cosines(x, yy, [0, 1, 2])[0]
        changed = metrics.class_cosines((x @ q) * rng.uniform(0.1, 3, (30, 1)), yy, [0, 1, 2])[
            0
        ]
        np.testing.assert_allclose(original, changed, atol=1e-12)

    def test_zero_vectors_and_singletons_are_not_self_similarity_evidence(self):
        mean, counts, zeros = metrics.class_cosines(
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([0, 0, 1]), [0, 1]
        )
        self.assertTrue(np.isnan(np.diag(mean)).all())
        self.assertEqual(mean[0, 1], 0)
        np.testing.assert_array_equal(counts, [[0, 1], [1, 0]])
        self.assertEqual(zeros, 1)


if __name__ == '__main__':
    unittest.main()
