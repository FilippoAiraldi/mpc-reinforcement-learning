import unittest
from copy import deepcopy
from itertools import product
from unittest.mock import Mock

import casadi as cs
import numpy as np
from parameterized import parameterized

from mpcrl import LearnableParameter, LearnableParametersDict, LearningRate
from mpcrl import optim as O
from mpcrl import schedulers as S
from mpcrl.optim.gradient_based_optimizer import GradientBasedOptimizer

np.random.seed(10)

N_PARAMS = 5
pars = np.random.randn(N_PARAMS)
LEARNABLE_PARS = LearnableParametersDict(
    [LearnableParameter(f"p{i}", 1, pars[i]) for i in range(N_PARAMS)]
)


class DummyOptimizer(GradientBasedOptimizer):
    def update(self, *args, **kwargs):
        pass


class TestGradientBasedOptimizer(unittest.TestCase):
    def test_init(self):
        opt = DummyOptimizer(learning_rate=0.1, max_percentage_update=0.5)
        self.assertIsInstance(opt.learning_rate, LearningRate)
        self.assertEqual(opt.learning_rate.value, 0.1)
        self.assertEqual(opt.max_percentage_update, 0.5)
        self.assertIsNone(opt.learnable_parameters)
        self.assertIsNone(opt._update_solver)

    def test_init_with_scheduler(self):
        learning_rate = S.ExponentialScheduler(0.56, 0.99)
        opt = DummyOptimizer(learning_rate)
        self.assertIsInstance(opt.learning_rate, LearningRate)
        self.assertIsInstance(opt.learning_rate.scheduler, S.Scheduler)
        self.assertEqual(opt.learning_rate.value, 0.56)

    def test_set_learnable_parameters(self):
        opt = DummyOptimizer(0.1)
        opt.set_learnable_parameters(LEARNABLE_PARS)
        self.assertIs(opt.learnable_parameters, LEARNABLE_PARS)
        self.assertIsNone(opt._update_solver)

    def test_set_learnable_parameters__with_solver(self):
        opt = DummyOptimizer(0.1, max_percentage_update=0.5)
        opt.set_learnable_parameters(LEARNABLE_PARS)
        self.assertIs(opt.learnable_parameters, LEARNABLE_PARS)
        self.assertIsInstance(opt._update_solver, cs.Function)

    def test_init_update_solver(self):
        learnable_pars = deepcopy(LEARNABLE_PARS)
        learnable_pars.lb = -np.arange(1, N_PARAMS + 1, dtype=float)
        learnable_pars.ub = np.arange(1, N_PARAMS + 1, dtype=float)
        opt = DummyOptimizer(0.1)
        opt.set_learnable_parameters(learnable_pars)
        self.assertIsNotNone(opt._update_solver)

    def test_get_update_bounds__raises__with_no_learnable_parameters(self):
        opt = DummyOptimizer(learning_rate=0.1)
        msg = "'NoneType' object has no attribute 'lb'"
        with self.assertRaises(AttributeError, msg=msg):
            opt._get_update_bounds(np.random.rand(N_PARAMS))

    def test_get_update_bounds(self):
        learnable_pars = deepcopy(LEARNABLE_PARS)
        lb = -np.arange(1, N_PARAMS + 1, dtype=float)
        ub = theta = np.arange(1, N_PARAMS + 1, dtype=float)
        learnable_pars.lb = lb
        learnable_pars.ub = ub
        opt = DummyOptimizer(learning_rate=0.1)
        opt.set_learnable_parameters(learnable_pars)
        lb_dtheta, ub_dtheta = opt._get_update_bounds(theta)
        np.testing.assert_array_equal(lb_dtheta, lb - theta)
        np.testing.assert_array_equal(ub_dtheta, 0)

    def test_get_update_bounds_with_maximum_update_percentage(self):
        learnable_pars = deepcopy(LEARNABLE_PARS)
        lb = theta = -np.arange(1, N_PARAMS + 1, dtype=float)
        ub = np.arange(1, N_PARAMS + 1, dtype=float)
        learnable_pars.lb = lb
        learnable_pars.ub = ub
        opt = DummyOptimizer(learning_rate=0.1, max_percentage_update=0.5)
        opt.set_learnable_parameters(learnable_pars)
        lb_dtheta, ub_dtheta = opt._get_update_bounds(theta)
        np.testing.assert_array_equal(lb_dtheta, 0)
        np.testing.assert_array_equal(ub_dtheta, 0.5 * np.abs(theta))


class TestGradientDescent(unittest.TestCase):
    @parameterized.expand(product(("1st", "2nd"), (object(), None)))
    def test_update__calls_correct_method(self, order: str, solver: object):
        opt = O.GradientDescent(0.1)
        opt._update_solver = solver
        category = "constrained" if solver is not None else "unconstrained"
        mock = Mock()
        setattr(opt, f"_do_{order}_order_{category}_update", mock)
        g = np.random.randn(N_PARAMS)
        H = np.random.randn(N_PARAMS, N_PARAMS) if order == "2nd" else None
        opt.update(g, H)
        args = (g,) if order == "1st" else (g, H)
        mock.assert_called_once_with(*args)

    @parameterized.expand([(0.0,), (1e-100,)])
    def test_do_1st_order_unconstrained_update(self, wd: float):
        opt = O.GradientDescent(0.1, weight_decay=wd)
        learnable_pars = deepcopy(LEARNABLE_PARS)
        learnable_pars.value = theta = np.random.randn(N_PARAMS)
        opt.set_learnable_parameters(learnable_pars)
        g = np.random.randn(N_PARAMS)
        theta_new, status = opt.update(g)
        self.assertIsNone(status)
        np.testing.assert_array_almost_equal(theta_new, theta - 0.1 * g)

    @parameterized.expand([(0.0,), (1e-100,)])
    def test_do_2nd_order_unconstrained_update(self, wd: float):
        opt = O.GradientDescent(0.1, weight_decay=wd)
        learnable_pars = deepcopy(LEARNABLE_PARS)
        learnable_pars.value = theta = np.random.randn(N_PARAMS)
        opt.set_learnable_parameters(learnable_pars)
        g = np.random.randn(N_PARAMS)
        H = np.random.randn(N_PARAMS, N_PARAMS)
        H = H @ H.T + np.eye(N_PARAMS)
        theta_new, status = opt.update(g, H)
        self.assertIsNone(status)
        np.testing.assert_array_almost_equal(
            theta_new, theta - 0.1 * np.linalg.solve(H, g)
        )

    @parameterized.expand([(0.0,), (1e-100,)])
    def test_do_2nd_order_unconstrained_update_with_identity_hessian(self, wd: float):
        opt = O.GradientDescent(0.1, weight_decay=wd)
        learnable_pars = deepcopy(LEARNABLE_PARS)
        learnable_pars.value = theta = np.random.randn(N_PARAMS)
        opt.set_learnable_parameters(learnable_pars)
        g = np.random.randn(N_PARAMS)
        H = np.eye(N_PARAMS)
        theta_new, status = opt.update(g, H)
        self.assertIsNone(status)
        np.testing.assert_array_almost_equal(theta_new, theta - 0.1 * g)

    @parameterized.expand([(0.0,), (1e-100,)])
    def test_do_1st_order_constrained_update_with_small_bounds(self, wd: float):
        opt = O.GradientDescent(0.1, weight_decay=wd)
        learnable_pars = deepcopy(LEARNABLE_PARS)
        learnable_pars.value = theta = np.random.randn(N_PARAMS)
        lb = learnable_pars.lb = theta - np.abs(theta) * 5e-2
        ub = learnable_pars.ub = theta + np.abs(theta) * 5e-2
        opt.set_learnable_parameters(learnable_pars)
        g = np.random.randn(N_PARAMS) * 100
        theta_new, status = opt.update(g)
        self.assertIsNone(status)
        self.assertTrue(
            np.where(theta_new >= lb, True, np.isclose(theta_new, lb)).all()
        )
        self.assertTrue(
            np.where(theta_new <= ub, True, np.isclose(theta_new, ub)).all()
        )

    @parameterized.expand([(0.0,), (1e-100,)])
    def test_do_1st_order_constrained_update_with_large_bounds(self, wd: float):
        opt = O.GradientDescent(0.1, weight_decay=wd)
        learnable_pars = deepcopy(LEARNABLE_PARS)
        learnable_pars.value = theta = np.random.randn(N_PARAMS)
        learnable_pars.lb = -np.abs(theta) * 100
        learnable_pars.ub = np.abs(theta) * 100
        opt.set_learnable_parameters(learnable_pars)
        g = np.random.randn(N_PARAMS) / 100
        theta_new, status = opt.update(g)
        self.assertIsNone(status)
        np.testing.assert_array_equal(theta_new, theta - 0.1 * g)

    @parameterized.expand([(0.0,), (1e-100,)])
    def test_do_2nd_order_constrained_update_with_large_bounds_with_identity_hessian(
        self, wd: float
    ):
        opt = O.GradientDescent(0.1, weight_decay=wd)
        learnable_pars = deepcopy(LEARNABLE_PARS)
        learnable_pars.value = theta = np.random.randn(N_PARAMS)
        learnable_pars.lb = -np.abs(theta) * 100
        learnable_pars.ub = np.abs(theta) * 100
        opt.set_learnable_parameters(learnable_pars)
        g = np.random.randn(N_PARAMS) / 100
        H = np.eye(N_PARAMS)
        theta_new, status = opt.update(g, H)
        self.assertIsNone(status)
        np.testing.assert_array_equal(theta_new, theta - 0.1 * g)

    @parameterized.expand(product((0.0, 1e-100), (True, False)))
    def test_do_2nd_order_constrained_update_with_small_bounds(
        self, wd: float, cho_before_update: bool
    ):
        opt = O.GradientDescent(
            0.1, weight_decay=wd, cho_before_update=cho_before_update
        )
        learnable_pars = deepcopy(LEARNABLE_PARS)
        learnable_pars.value = theta = np.random.randn(N_PARAMS)
        lb = learnable_pars.lb = theta - np.abs(theta) * 5e-2
        ub = learnable_pars.ub = theta + np.abs(theta) * 5e-2
        opt.set_learnable_parameters(learnable_pars)
        g = np.random.randn(N_PARAMS) * 100
        H = np.random.randn(N_PARAMS, N_PARAMS)
        H = H @ H.T
        theta_new, status = opt.update(g, H)
        self.assertIsNone(status)
        self.assertTrue(
            np.where(theta_new >= lb, True, np.isclose(theta_new, lb)).all()
        )
        self.assertTrue(
            np.where(theta_new <= ub, True, np.isclose(theta_new, ub)).all()
        )


if __name__ == "__main__":
    unittest.main()
