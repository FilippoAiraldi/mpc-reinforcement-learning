import unittest

import numpy as np

from mpcrl.util import RangeNormalization


class TestRangeNormalization(unittest.TestCase):
    def test_str_and_repr(self):
        N = RangeNormalization({"x": [-1, 1]})
        for S in [N.__str__(), N.__repr__()]:
            self.assertIn("normalization", S.lower())

    def test_register__raises__when_registering_duplicate_ranges(self):
        N = RangeNormalization({"x": [-1, 1]})
        with self.assertRaises(KeyError):
            N.register({"x": [0, 2]})
        with self.assertRaises(KeyError):
            N.register(x=[0, 2])
        N.register(y=[0, 2])

    def test_can_normalize__only_valid_ranges(self):
        N = RangeNormalization({"x": [-1, 1]})
        self.assertTrue(N.can_normalize("x"))
        self.assertFalse(N.can_normalize("u"))

    def test_normalize__raises__if_shape_is_modified(self):
        N = RangeNormalization({"x": np.array([[-1, 1], [-2, 2]])})
        x = np.array([-5, 2])
        N.normalize("x", x)
        N.normalize("x", x.reshape(1, -1))
        with self.assertRaises(AssertionError):
            N.normalize("x", x.reshape(-1, 1))

    def test_normalize__computes_right_values(self):
        N = RangeNormalization({"x1": [-5, 2], "x2": np.array([[-1, 1], [-2, 2]])})
        x1 = 5
        y1 = N.normalize("x1", x1)
        z1 = N.denormalize("x1", y1)
        np.testing.assert_equal(y1, 10 / 7)
        np.testing.assert_equal(x1, z1)

        x2 = np.array([-5, 2])
        y2 = N.normalize("x2", x2)
        z2 = N.denormalize("x2", y2)
        np.testing.assert_equal(y2, [-2, 1])
        np.testing.assert_equal(x2, z2)


if __name__ == "__main__":
    unittest.main()
