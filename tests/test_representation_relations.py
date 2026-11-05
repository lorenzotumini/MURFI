"""Check class-block interpretation independently of visualization."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from representation_relations import class_cosines


class ClassRelationTests(unittest.TestCase):
    def test_diagonal_excludes_self_and_cross_blocks_use_all_pairs(self):
        x = np.array([[1., 0.], [0., 1.], [1., 1.], [1., 1.]])
        mean, counts, zeros = class_cosines(x, np.array([0, 0, 1, 1]), [0, 1])
        np.testing.assert_allclose(mean, [[0, 1/np.sqrt(2)], [1/np.sqrt(2), 1]])
        np.testing.assert_array_equal(counts, [[1, 4], [4, 1]])
        self.assertEqual(zeros, 0)

    def test_rotation_and_per_example_positive_scale_preserve_summary(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(30, 8))
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        yy = np.arange(30) % 3
        original = class_cosines(x, yy, [0, 1, 2])[0]
        changed = class_cosines((x @ q)*rng.uniform(.1, 3, (30, 1)), yy, [0, 1, 2])[0]
        np.testing.assert_allclose(original, changed, atol=1e-12)

    def test_zero_vectors_and_singletons_are_not_self_similarity_evidence(self):
        mean, counts, zeros = class_cosines(np.array([[0., 0.], [1., 0.], [0., 1.]]),
                                          np.array([0, 0, 1]), [0, 1])
        self.assertTrue(np.isnan(np.diag(mean)).all())
        self.assertEqual(mean[0, 1], 0)
        np.testing.assert_array_equal(counts, [[0, 1], [1, 0]])
        self.assertEqual(zeros, 1)


if __name__ == '__main__':
    unittest.main()
