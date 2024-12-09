import unittest
from itertools import product

import casadi as cs
import numpy as np
from parameterized import parameterized, parameterized_class
from scipy.special import comb

from mpcrl.util import control, iters, math, named, seeding


class DummyAgent(named.Named): ...


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
        L = math.cholesky_added_multiple_identity(A)
        np.testing.assert_allclose(A, L @ L.T)

    def test_cholesky_added_multiple_identities__raises__max_iterations_reached(self):
        n = 5
        A = np.random.rand(n, n) - np.eye(n) * 5
        A = 0.5 * (A + A.T)
        with self.assertRaisesRegex(ValueError, "Maximum iterations reached."):
            math.cholesky_added_multiple_identity(A, maxiter=1)

    @parameterized.expand(
        [
            (
                3,
                5,
                {
                    (0, 0, 5),
                    (0, 1, 4),
                    (0, 2, 3),
                    (0, 3, 2),
                    (0, 4, 1),
                    (0, 5, 0),
                    (1, 0, 4),
                    (1, 1, 3),
                    (1, 2, 2),
                    (1, 3, 1),
                    (1, 4, 0),
                    (2, 0, 3),
                    (2, 1, 2),
                    (2, 2, 1),
                    (2, 3, 0),
                    (3, 0, 2),
                    (3, 1, 1),
                    (3, 2, 0),
                    (4, 0, 1),
                    (4, 1, 0),
                    (5, 0, 0),
                },
            ),
            (1, 4, {(4,)}),
            (10, 1, {tuple(row) for row in np.eye(10)}),
            (4, 3, None),
        ]
    )
    def test_monomial_powers__computes_correct_powers(self, n, k, out):
        p = math.monomial_powers(n, k)
        np.testing.assert_array_equal(p.sum(axis=1), k)
        expected_n_combinations = comb(k + n - 1, n - 1, exact=True)
        self.assertEqual(p.shape, (expected_n_combinations, n))
        if out is None:
            return
        set_of_combinations = {tuple(row) for row in p}
        self.assertSetEqual(set_of_combinations, out)

    @parameterized.expand([(3, 1, 2), (5, 0, 5), (4, 5, 4)])
    def test_monomials_basis_function(self, n, mindeg, maxdeg):
        Phi = math.monomials_basis_function(n, mindeg, maxdeg)
        self.assertEqual(Phi.n_in(), 1)
        self.assertEqual(Phi.n_out(), 1)

        self.assertEqual(n, Phi.size1_in(0))
        expected_out = sum(
            comb(deg + n - 1, n - 1, exact=True) for deg in range(mindeg, maxdeg + 1)
        )
        self.assertEqual(expected_out, Phi.size1_out(0))

    @parameterized.expand(
        product(
            [(), (1,), (1, 1), (10,), (10, 1)],
            [1, 2, np.inf, *(np.random.rand(3) * 10)],
        )
    )
    def test_dual_norm(self, shape, ord):
        x = np.random.randn(*shape)
        y_actual = float(math.dual_norm(x, ord))
        x = np.reshape(x, -1)
        if ord == 1:
            y_expected = np.linalg.norm(x, ord=np.inf)
        elif np.isposinf(ord).item():
            y_expected = np.linalg.norm(x, ord=1)
        else:
            dual_norm = ord / (ord - 1)
            y_expected = np.linalg.norm(x, dual_norm)
        self.assertAlmostEqual(y_actual, y_expected, msg=f"error: {shape}, {ord}")

    def test_clip(self):
        x, lb = np.random.randn(2, 10)
        ub = lb + np.abs(np.random.randn(10))
        x_actual = math.clip(x, lb, ub).toarray().flatten()
        x_expected = np.clip(x, lb, ub)
        np.testing.assert_allclose(x_actual, x_expected)


class TestControl(unittest.TestCase):
    def test_lqr__returns_correctly(self):
        K_exp = np.array([[0.307774887341029, 0.441249017604930]])
        P_exp = np.array(
            [
                [0.531874210236110, 0.035928068775907],
                [0.035928068775907, 0.422885782452801],
            ]
        )
        A = np.array([[-0.9, 0.25], [0, -1.1]])
        B = np.array([[0.23], [0.45]])
        Q = np.eye(2)
        R = np.atleast_2d(0.45)
        K, P = control.lqr(A, B, Q, R)
        np.testing.assert_allclose(K, K_exp)
        np.testing.assert_allclose(P, P_exp)

    def test_lqr__with_M__returns_correctly(self):
        K_exp = np.array([[0.462006509187942, 0.766379710152564]])
        P_exp = np.array(
            [
                [0.502193051922548, -0.016892161794694],
                [-0.016892161794694, 0.330569037292075],
            ]
        )
        A = np.array([[-0.9, 0.25], [0, -1.1]])
        B = np.array([[0.23], [0.45]])
        Q = np.eye(2)
        R = np.atleast_2d(0.45)
        M = np.array([[0.1], [0.2]])
        K, P = control.lqr(A, B, Q, R, M)
        np.testing.assert_allclose(K, K_exp)
        np.testing.assert_allclose(P, P_exp)

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

    def test_cbf_degree_1(self):
        M = cs.SX.sym("M")
        v0 = cs.SX.sym("v0")

        p = cs.SX.sym("p")
        v = cs.SX.sym("v")
        z = cs.SX.sym("z")
        x = cs.vertcat(p, v, z)
        u = cs.SX.sym("u")
        friction_coeffs = cs.SX.sym("friction_coeffs", 3, 1)
        friction = cs.dot(friction_coeffs, cs.vertcat(1, v, v**2))
        x_dot = cs.vertcat(v, (u - friction) / M, v0 - v)
        dynamics = cs.Function("dynamics", [x, u], [x_dot], {"allow_free": True})

        Th = cs.SX.sym("Th")
        h = lambda x_: x_[2] - Th * x_[1]  # >= 0

        gamma = cs.SX.sym("gamma")
        alphas = [lambda w: gamma * w]
        actual_cbf = control.cbf(h, x, u, dynamics, alphas)
        expected_cbf = Th / M * (friction - u) + (v0 - v) + gamma * (z - Th * v)
        self.assertTrue(all(cbf.shape == (1, 1) for cbf in [actual_cbf, expected_cbf]))

        variables = cs.symvar(actual_cbf)
        shapes = [v.shape for v in variables]
        diff = cs.Function("d", variables, (actual_cbf, expected_cbf))
        for _ in range(20):
            values = [np.random.randn(*shape) for shape in shapes]
            np.testing.assert_allclose(*diff(*values), atol=1e-12, rtol=1e-12)

    def test_cbf_degree_2(self):
        v = cs.SX.sym("v")
        z = cs.SX.sym("z")
        u = cs.SX.sym("u")
        v0 = cs.SX.sym("v0")
        x = cs.vertcat(v, z)
        f = cs.vertcat(0, v0 - v)
        g = cs.vertcat(1, 0)
        x_dot = f + g * u
        dynamics = cs.Function("dynamics", [x, u], [x_dot], {"allow_free": True})
        delta = cs.SX.sym("delta")
        h = lambda x_: x_[1] - delta  # >= 0

        # degree 1
        alphas = [lambda y: y]
        actual_cbf1 = control.cbf(h, x, u, dynamics, alphas)
        expected_cbf1 = v0 - v + cs.SX.zeros(1, 1) * u + z - delta

        # degree 2
        alphas = [lambda y: y**2] * 2
        actual_cbf2 = control.cbf(h, x, u, dynamics, alphas)
        h_ = h(x)
        Lfh_ = math.lie_derivative(h_, x, f)  # v0 - v
        Lf2h_ = math.lie_derivative(h_, x, f, 2)  # 0
        LgLfh_ = math.lie_derivative(Lfh_, x, g)  # -1
        expected_cbf2 = Lf2h_ + LgLfh_ * u + 2 * h_ * Lfh_ + (Lfh_ + h_**2) ** 2
        self.assertTrue(
            all(
                cbf.shape == (1, 1)
                for cbf in [actual_cbf1, expected_cbf1, actual_cbf2, expected_cbf2]
            )
        )

        variables = cs.symvar(actual_cbf2)
        shapes = [v.shape for v in variables]
        diff = cs.Function(
            "d", variables, (actual_cbf1, expected_cbf1, actual_cbf2, expected_cbf2)
        )
        for _ in range(200):
            values = [np.random.randn(*shape) for shape in shapes]
            out1, out2, out3, out4 = diff(*values)
            np.testing.assert_allclose(out1, out2, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(out3, out4, atol=1e-12, rtol=1e-12)

    def test_dcbf_degree_1(self):
        A = cs.SX.sym("A", 2, 2)
        B = cs.SX.sym("B", 2, 1)
        x = cs.SX.sym("x", A.shape[0], 1)
        u = cs.SX.sym("u", B.shape[1], 1)
        dynamics = lambda x, u: A @ x + B @ u
        M = cs.SX.sym("M")
        c = cs.SX.sym("c")
        gamma = cs.SX.sym("gamma")
        alphas = [lambda z: gamma * z]
        h = lambda x: M - c * x[0]  # >= 0

        actual_cbf = control.dcbf(h, x, u, dynamics, alphas)
        expected_cbf = -c * (A[0, :] @ x + B[0] * u - x[0]) + gamma * (M - c * x[0])
        self.assertTrue(all(cbf.shape == (1, 1) for cbf in [actual_cbf, expected_cbf]))

        variables = cs.symvar(actual_cbf)
        shapes = [v.shape for v in variables]
        diff = cs.Function("d", variables, (actual_cbf, expected_cbf))
        for _ in range(20):
            values = [np.random.randn(*shape) for shape in shapes]
            np.testing.assert_allclose(*diff(*values), atol=1e-12, rtol=1e-12)

    def test_dcbf_degree_2(self):
        A = cs.SX.sym("A", 2, 2)
        B = cs.SX.zeros(2, 1)
        B[1] = cs.SX.sym("b")
        x = cs.SX.sym("x", A.shape[0], 1)
        u = cs.SX.sym("u", B.shape[1], 1)
        dynamics = lambda x, u: A @ x + B @ u
        M = cs.SX.sym("M")
        h = lambda x: M - x[0]  # >= 0
        gamma = cs.SX.sym("gamma", 2, 1)

        # degree 1
        alphas = [lambda z: gamma[0] * z]
        actual_cbf1 = control.dcbf(h, x, u, dynamics, alphas)
        expected_cbf1 = -A[0, :] @ x + x[0] + gamma[0] * (M - x[0])

        # degree 2
        alphas = [lambda z: gamma[0] * z, lambda z: gamma[1] * z]
        actual_cbf2 = control.dcbf(h, x, u, dynamics, alphas)
        expected_cbf2 = (
            (A[0, :] @ x - x[0]) * (1 - A[0, 0] - cs.sum1(gamma))
            + (M - x[0]) * gamma[0] * gamma[1]
            - A[0, 1] * (A[1, :] @ x - x[1])
            - B[1] * A[0, 1] * u
        )
        self.assertTrue(
            all(
                cbf.shape == (1, 1)
                for cbf in [actual_cbf1, expected_cbf1, actual_cbf2, expected_cbf2]
            )
        )

        variables = cs.symvar(actual_cbf2)
        shapes = [v.shape for v in variables]
        diff = cs.Function(
            "d", variables, (actual_cbf1, expected_cbf1, actual_cbf2, expected_cbf2)
        )
        for _ in range(200):
            values = [np.random.randn(*shape) for shape in shapes]
            out1, out2, out3, out4 = diff(*values)
            np.testing.assert_allclose(out1, out2, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(out3, out4, atol=1e-12, rtol=1e-12)


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
