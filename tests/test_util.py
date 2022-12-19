import unittest

import numpy as np

from mpcrl.util import math, named


class DummyAgent(named.Named):
    ...


class TestNamedAgent(unittest.TestCase):
    def test_init__with_given_name__saves_name_correctly(self):
        dummy = DummyAgent(name="ciao")
        self.assertEqual(dummy.name, "ciao")

    def test_init__without_name__creates_name_with_class(self):
        dummy0 = DummyAgent()
        dummy1 = DummyAgent()
        self.assertEqual(dummy0.name, "DummyAgent0")
        self.assertEqual(dummy1.name, "DummyAgent1")

    def test_str_and_repr(self):
        name = "ciao"
        dummy = DummyAgent(name=name)
        S = dummy.__str__()
        self.assertIn(name, S)
        S = dummy.__repr__()
        self.assertIn(name, S)
        self.assertIn(DummyAgent.__name__, S)


class TestMath(unittest.TestCase):
    def test_cholesky_added_multiple_identities__performs_correct_factorization(self):
        n = 5
        A = np.random.rand(n, n) + np.eye(n) * 5
        A = 0.5 * (A + A.T)
        L = math.cholesky_added_multiple_identities(A)
        np.testing.assert_allclose(A, L @ L.T)

    def test_cholesky_added_multiple_identities__raises__max_iterations_reached(self):
        n = 5
        A = np.random.rand(n, n) - np.eye(n) * 5
        A = 0.5 * (A + A.T)
        with self.assertRaisesRegex(ValueError, "Maximum iterations reached."):
            math.cholesky_added_multiple_identities(A, maxiter=1)

    def test_dlqr__returns_correctly(self):
        K_exp = np.array([[1.075936290787970, 1.824593914133278]])
        P_exp = np.array(
            [
                [6.783278637425703, 2.903698804441288],
                [2.903698804441288, 5.026668672843932],
            ]
        )
        A = np.array([[1, 0.25], [0, 1]])
        B = np.array([[0.03], [0.25]])
        Q = np.eye(2)
        R = 0.5
        K, P = math.dlqr(A, B, Q, R)
        np.testing.assert_allclose(K, K_exp)
        np.testing.assert_allclose(P, P_exp)


if __name__ == "__main__":
    unittest.main()
