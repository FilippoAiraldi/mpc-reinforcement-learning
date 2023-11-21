import unittest

import casadi as cs
import numpy as np
from parameterized import parameterized, parameterized_class

from mpcrl.util import control, iters, math, named, seeding


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

    @parameterized.expand(
        [
            ((1, 1), 1),
            ((5, 4), 5),
            (
                (np.arange(2, 11, 2), 4),
                np.asarray(
                    [
                        [2, 4, 6, 8],
                        [2, 4, 6, 10],
                        [2, 4, 8, 10],
                        [2, 6, 8, 10],
                        [4, 6, 8, 10],
                    ]
                ),
            ),
            ((np.asarray([10, 20, 30]), 2), np.asarray([[10, 20], [10, 30], [20, 30]])),
        ]
    )
    def test_nchoosek__computes_correct_combinations(
        self, inp: tuple[int, int], out: int
    ):
        out_ = math.nchoosek(*inp)
        np.testing.assert_allclose(out_, out)

    @parameterized.expand([(1, 4, 4), (10, 1, np.eye(10)), (4, 3, None)])
    def test_monomial_powers__computes_correct_powers(self, n, k, out):
        p = math.monomial_powers(n, k)
        self.assertEqual(p.shape[1], n)
        np.testing.assert_allclose(p.sum(axis=1), k)
        if out is not None:
            np.testing.assert_allclose(p, out)


class TestControl(unittest.TestCase):
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
        K, P = control.dlqr(A, B, Q, R)
        np.testing.assert_allclose(K, K_exp)
        np.testing.assert_allclose(P, P_exp)

    def test_dlqr__with_M__returns_correctly(self):
        K_exp = np.array([[1.132928272226698, 1.820247185260309]])
        P_exp = np.array(
            [
                [6.426698776552665, 2.359469798498852],
                [2.359469798498852, 3.806839220681791],
            ]
        )
        A = np.array([[1, 0.25], [0, 1]])
        B = np.array([[0.03], [0.25]])
        Q = np.eye(2)
        R = np.atleast_2d(0.5)
        M = np.array([[0.1], [0.2]])
        K, P = control.dlqr(A, B, Q, R, M)
        np.testing.assert_allclose(K, K_exp)
        np.testing.assert_allclose(P, P_exp)

    def test_rk4__returns_correcly(self):
        def f(x):
            return -3 * x

        dt = 0.01
        x = cs.SX.sym("x")
        fd = cs.Function("fd", [x], [control.rk4(f, x, dt)])

        Y = [1]
        for _ in range(100):
            Y.append(fd(Y[-1]))
        Y = cs.hcat(Y).full().squeeze()
        Y_exact = np.exp(-3 * (np.arange(Y.size) * dt))
        np.testing.assert_allclose(Y, Y_exact)


@parameterized_class("starts_with", [(False,), (True,)])
class TestIters(unittest.TestCase):
    @parameterized.expand([(5,), (1,), (22,)])
    def test_bool_cycle(self, frequency: int):
        T = frequency * 10
        cycle = iters.bool_cycle(frequency, self.starts_with)
        cycle = [next(cycle) for _ in range(T)]
        self.assertEqual(cycle[0], self.starts_with or frequency == 1)
        self.assertEqual(T // frequency, sum(cycle))


class TestSeeding(unittest.TestCase):
    def test_mk_seed(self):
        rng = np.random.default_rng()
        self.assertTrue(0 <= seeding.mk_seed(rng) < 2**32)


if __name__ == "__main__":
    unittest.main()
