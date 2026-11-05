import math
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import random_faults as rf


class RandomFaultTests(unittest.TestCase):
    def test_layout_includes_every_bias_and_weight_bit(self):
        model = rf.pilot.initial(0)
        keys, ends = rf.layout(model)
        self.assertEqual(int(ends[-1]), 153920)
        start = 0
        for key in keys:
            self.assertEqual(rf.decode(start, keys, ends), (key, 0, 0))
            end = start + model[key].size*32
            self.assertEqual(rf.decode(end-1, keys, ends), (key, model[key].size-1, 31))
            start = end

    def test_nonfinite_fault_is_not_rejected_or_filtered(self):
        model = {'p': np.array([1.0], dtype=np.float32)}
        before = rf.pilot.clone(model)
        rf.pilot.toggle(model, 'p', 0, 30)
        self.assertTrue(math.isinf(float(model['p'][0])))
        rf.pilot.audit(before, model, [{'parameter': 'p', 'index': 0, 'bit': 30}])
        rf.pilot.toggle(model, 'p', 0, 30)
        rf.pilot.audit(before, model, [])

    def test_recovery_with_damage_is_counted_as_damage(self):
        base = {'valid': True, 'forget': 0.0, 'retain': 0.95, 'fpr': 0.0}
        pred = np.array([0])
        self.assertEqual(rf.classify(base | {'forget': 0.5, 'retain': 0.90}, base, pred, pred),
                         'utility_or_false_positive_damage')
        self.assertEqual(rf.classify(base | {'forget': 0.5, 'fpr': 0.1}, base, pred, pred),
                         'utility_or_false_positive_damage')
        self.assertEqual(rf.classify(base | {'forget': 0.5}, base, pred, pred), 'selective_recovery')

    def test_zero_event_probability_bound(self):
        self.assertAlmostEqual(rf.upper95(0,1), 0.95)
        self.assertEqual(rf.upper95(1,1), 1.0)
        self.assertAlmostEqual((1-rf.upper95(0,10000))**10000, 0.05, places=12)


if __name__ == '__main__':
    unittest.main()
