"""Check the binary fault representation without a GPU or PyTorch dependency."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot import flipped_float, word


class BitRepresentationTests(unittest.TestCase):
    def test_each_flip_changes_exactly_one_storage_bit(self):
        for value in [0.0, -0.0, 1.0, -3.25, 0.001, 100.0]:
            for bit in range(32):
                changed = flipped_float(value, bit)
                if math.isnan(changed):
                    continue  # NaN payloads can be canonicalized by Python.
                self.assertEqual(word(value) ^ word(changed), 1 << bit)

    def test_repeated_flip_restores_word(self):
        for bit in range(32):
            changed = flipped_float(1.0, bit)
            self.assertEqual(word(flipped_float(changed, bit)), word(1.0))

    def test_sign_and_exponent_are_real_binary_changes(self):
        self.assertEqual(flipped_float(1.0, 31), -1.0)
        self.assertEqual(flipped_float(1.0, 23), 0.5)
        self.assertTrue(math.isinf(flipped_float(1.0, 30)))


if __name__ == '__main__':
    unittest.main()
