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
import representation_resnet as resnet


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12)
        self.x = self.rng.normal(size=(120, 9))
        self.q, _ = np.linalg.qr(self.rng.normal(size=(9, 9)))

    def test_cka_orthogonal_scale_translation_invariance(self):
        transformed = 3 * self.x @ self.q + 7
        for debiased in (False, True):
            self.assertAlmostEqual(rep.cka(self.x, transformed, debiased), 1, places=12)
        self.assertLess(rep.cka(self.x, self.x[self.rng.permutation(len(self.x))]), .2)

    def test_unbiased_hsic_corrects_high_dimension_small_sample_bias(self):
        x, y = self.rng.normal(size=(30, 2000)), self.rng.normal(size=(30, 2000))
        self.assertGreater(rep.cka(x, y), .95)
        self.assertLess(abs(rep.cka(x, y, debiased=True)), .15)

    def test_collapsed_and_tiny_groups_are_undefined(self):
        self.assertIsNone(rep.cka(self.x, np.ones_like(self.x)))
        self.assertIsNone(rep.cka(self.x[:3], self.x[:3], debiased=True))
        self.assertIsNone(rep.cka(self.x[:1], self.x[:1]))

    def test_neighbor_overlap_matches_rotation_and_excludes_self(self):
        a, _ = rep.cosine_neighbors(self.x, 10, block_size=7)
        b, _ = rep.cosine_neighbors(self.x @ self.q, 10, block_size=31)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.any(a == np.arange(len(a))[:, None]))
        np.testing.assert_array_equal(rep.neighbor_overlap(a, b), np.ones(len(a)))

    def test_neighbors_match_brute_force_and_zero_rows_are_excluded(self):
        z = self.x / np.linalg.norm(self.x, axis=1, keepdims=True)
        sim = z @ z.T
        np.fill_diagonal(sim, -np.inf)
        expected = np.argsort(-sim, axis=1, kind='stable')[:, :5]
        actual, _ = rep.cosine_neighbors(self.x, 5)
        np.testing.assert_array_equal(actual, expected)
        zeros, diag = rep.cosine_neighbors(np.zeros((10, 4)), 5)
        self.assertEqual(diag['zero_vectors'], 10)
        self.assertTrue(np.isnan(rep.neighbor_overlap(zeros, zeros)).all())

    def test_alignment_uses_only_calibration(self):
        cal = np.arange(len(self.x)) < 40
        transformed = self.x @ self.q + 13
        aligned, _ = rep.align_to_reference({'original': self.x, 'unlearned': transformed}, cal)
        np.testing.assert_allclose(aligned['original'], aligned['unlearned'], atol=1e-12)
        transformed[~cal] += 50
        altered, _ = rep.align_to_reference({'original': self.x, 'unlearned': transformed}, cal)
        np.testing.assert_allclose(aligned['unlearned'][cal], altered['unlearned'][cal], atol=1e-12)
        self.assertGreater(np.linalg.norm(altered['unlearned'][~cal]-aligned['original'][~cal]), 10)


class BundleTests(unittest.TestCase):
    def test_full_synthetic_pipeline_and_nonoverwriting_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root/'features.npz'
            args = argparse.Namespace(seed=0, output=bundle)
            rep.extract_synthetic(args)
            meta, arrays = rep.load_bundle(bundle)
            self.assertEqual(meta['source'], 'synthetic')
            with self.assertRaises(FileExistsError):
                rep.extract_synthetic(args)
            result = rep.analyze(bundle, root/'analysis', ks=(5, 10), null_repeats=2)
            for row in result['comparisons']:
                if row['pair'] == 'original--unlearned':
                    self.assertAlmostEqual(row['cka'], 1, places=10)
                    self.assertAlmostEqual(row['cka_debiased'], 1, places=10)
                    self.assertEqual(row['knn_10'], 1)
            for name in ('metrics.json', 'comparisons.csv', 'behavior.csv', 'REPORT.md',
                         'pca.png', 'layerwise.png', 'similarity.png', 'neighbors.png'):
                self.assertTrue((root/'analysis'/name).is_file(), name)
            json.loads((root/'analysis'/'metrics.json').read_text())
            with self.assertRaises(FileExistsError):
                rep.analyze(bundle, root/'analysis')
            arrays['sample_ids'][1] = arrays['sample_ids'][0]
            with self.assertRaisesRegex(ValueError, 'unique'):
                rep.validate_bundle(meta, arrays)

    def test_bad_rows_nonfinite_and_forget_calibration_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)/'features.npz'
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
            args = argparse.Namespace(seed=0, forget_class=0, probe_seed=2026, per_class=10,
                                      evaluation='discovery', output=Path(directory)/'digits.npz',
                                      reference=root/'reports/targeted-flips/cpu/digits-main.json.gz',
                                      fixtures=root/'reports/random-flips')
            rep.extract_digits(args)
            meta, arrays = rep.load_bundle(args.output)
            from sklearn.datasets import load_digits
            x, _ = load_digits(return_X_y=True)
            x = (x/16).astype('float32')
            ids = [int(i.split(':')[1]) for i in arrays['sample_ids']]
            self.assertEqual(len(ids), len(set(ids)))
            with np.load(args.fixtures/'digit0-seed0.npz', allow_pickle=False) as fixture:
                expected = np.maximum(x[ids] @ fixture['original_w1'] + fixture['original_b1'], 0)
            np.testing.assert_array_equal(arrays[rep.feature_key('original', 'hidden')], expected)
            self.assertEqual(meta['evaluation_split'], 'discovery')

    def test_resnet_split_overlap_rejected_without_torch(self):
        resnet.validate_splits({'train': [0, 1], 'search': [2], 'select': [3]}, total=4)
        with self.assertRaises(ValueError):
            resnet.validate_splits({'train': [0, 1], 'search': [1], 'select': [3]}, total=4)


if __name__ == '__main__':
    unittest.main()
