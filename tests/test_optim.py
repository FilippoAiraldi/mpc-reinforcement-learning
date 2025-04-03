import unittest
from itertools import product
from unittest.mock import Mock

import casadi as cs
import numpy as np
import torch
from parameterized import parameterized

from mpcrl import LearnableParameter, LearnableParametersDict
from mpcrl import optim as O
from mpcrl import schedulers as S
from mpcrl.optim.gradient_based_optimizer import GradientBasedOptimizer

np.random.seed(10)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

SMALL = 1e-270
N_PARAMS = 5
pars = np.random.randn(N_PARAMS)
LEARNABLE_PARS = LearnableParametersDict(
    [LearnableParameter(f"p{i}", 1, pars[i]) for i in range(N_PARAMS)]
)


class DummyOptimizer(GradientBasedOptimizer):
    _hessian_sparsity = "dense"
    _update_solver = None

    def update(self, *args, **kwargs):
        pass


class TestGradientBasedOptimizer(unittest.TestCase):
    def test_init_and_step(self):
        lr = object()
        perc = object()
        opt = DummyOptimizer(learning_rate=lr, max_percentage_update=perc)
        self.assertIsInstance(opt.lr_scheduler, S.Scheduler)
        self.assertIs(opt.lr_scheduler.value, lr)
        self.assertIsNone(opt.hook)
        self.assertIs(opt.max_percentage_update, perc)
        self.assertIsNone(opt.learnable_parameters)
        self.assertIsNone(opt._update_solver)

    def test_init__with_scheduler(self):
        hook = object()
        learning_rate = S.ExponentialScheduler(0.56, 0.99)
        learning_rate.step = Mock()

        opt = DummyOptimizer(learning_rate, hook=hook)
        opt.step()

        self.assertIsInstance(opt.lr_scheduler, S.Scheduler)
        self.assertEqual(opt.lr_scheduler.value, 0.56)
        self.assertIs(opt.hook, hook)
        learning_rate.step.assert_called_once_with()

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

    def test_init__creates_update_solver(self):
        learnable_pars = LEARNABLE_PARS.copy(deep=True)
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
        learnable_pars = LEARNABLE_PARS.copy(deep=True)
        lb = -np.arange(1, N_PARAMS + 1, dtype=float)
        ub = theta = np.arange(1, N_PARAMS + 1, dtype=float)
        learnable_pars.lb = lb
        learnable_pars.ub = ub
        opt = DummyOptimizer(learning_rate=0.1)
        opt.set_learnable_parameters(learnable_pars)
        lb_dtheta, ub_dtheta = opt._get_update_bounds(theta)
        np.testing.assert_array_equal(lb_dtheta, lb - theta)
        np.testing.assert_array_equal(ub_dtheta, 0)

    def test_get_update_bounds__with_maximum_update_percentage(self):
        learnable_pars = LEARNABLE_PARS.copy(deep=True)
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
    @parameterized.expand([(False,), (True,)])
    def test_update__unconstrained(self, nesterov: bool):
        opt = O.GradientDescent(
            learning_rate=0.1,
            weight_decay=SMALL,
            momentum=SMALL,
            dampening=SMALL,
            nesterov=nesterov,
        )
        learnable_pars = LEARNABLE_PARS.copy(deep=True)
        learnable_pars.value = theta = np.random.randn(N_PARAMS)
        opt.set_learnable_parameters(learnable_pars)
        g = np.random.randn(N_PARAMS)
        status = opt.update(g)
        self.assertIsNone(status)
        np.testing.assert_array_almost_equal(learnable_pars.value, theta - 0.1 * g)

    @parameterized.expand([(False,), (True,)])
    def test_update__constrained__with_small_bounds(self, nesterov: bool):
        opt = O.GradientDescent(
            learning_rate=0.1,
            weight_decay=SMALL,
            momentum=SMALL,
            dampening=SMALL,
            nesterov=nesterov,
        )
        learnable_pars = LEARNABLE_PARS.copy(deep=True)
        learnable_pars.value = theta = np.random.randn(N_PARAMS)
        lb = learnable_pars.lb = theta - np.abs(theta) * 5e-2
        ub = learnable_pars.ub = theta + np.abs(theta) * 5e-2
        opt.set_learnable_parameters(learnable_pars)
        g = np.random.randn(N_PARAMS) * 100
        status = opt.update(g)
        theta_new = learnable_pars.value
        self.assertIsNone(status)
        self.assertTrue(
            np.where(theta_new >= lb, True, np.isclose(theta_new, lb)).all()
        )
        self.assertTrue(
            np.where(theta_new <= ub, True, np.isclose(theta_new, ub)).all()
        )

    @parameterized.expand([(False,), (True,)])
    def test_update__constrained__with_large_bounds(self, nesterov: bool):
        opt = O.GradientDescent(
            learning_rate=0.1,
            weight_decay=SMALL,
            momentum=SMALL,
            dampening=SMALL,
            nesterov=nesterov,
        )
        learnable_pars = LEARNABLE_PARS.copy(deep=True)
        learnable_pars.value = theta = np.random.randn(N_PARAMS)
        learnable_pars.lb = -np.abs(theta) * 100
        learnable_pars.ub = np.abs(theta) * 100
        opt.set_learnable_parameters(learnable_pars)
        g = np.random.randn(N_PARAMS) / 100
        status = opt.update(g)
        self.assertIsNone(status)
        np.testing.assert_array_equal(learnable_pars.value, theta - 0.1 * g)


class TestNewtonMethod(unittest.TestCase):
    @parameterized.expand(product((0.0, SMALL), (False, True)))
    def test_update__unconstrained(self, w: float, cho_before_update: bool):
        opt = O.NewtonMethod(0.1, weight_decay=w, cho_before_update=cho_before_update)
        learnable_pars = LEARNABLE_PARS.copy(deep=True)
        learnable_pars.value = theta = np.random.randn(N_PARAMS)
        opt.set_learnable_parameters(learnable_pars)
        g = np.random.randn(N_PARAMS)
        H = np.random.randn(N_PARAMS, N_PARAMS)
        H = H @ H.T + np.eye(N_PARAMS)
        status = opt.update(g, H)
        self.assertIsNone(status)
        np.testing.assert_array_almost_equal(
            learnable_pars.value, theta - 0.1 * np.linalg.solve(H, g)
        )

    @parameterized.expand(product((0.0, SMALL), (False, True)))
    def test_update__unconstrained__with_identity_hessian(
        self, w: float, cho_before_update: bool
    ):
        opt = O.NewtonMethod(0.1, weight_decay=w, cho_before_update=cho_before_update)
        learnable_pars = LEARNABLE_PARS.copy(deep=True)
        learnable_pars.value = theta = np.random.randn(N_PARAMS)
        opt.set_learnable_parameters(learnable_pars)
        g = np.random.randn(N_PARAMS)
        H = np.eye(N_PARAMS)
        status = opt.update(g, H)
        self.assertIsNone(status)
        np.testing.assert_array_almost_equal(learnable_pars.value, theta - 0.1 * g)

    @parameterized.expand(product((0.0, SMALL), (False, True)))
    def test_update__constrained__with_large_bounds_with_identity_hessian(
        self, w: float, cho_before_update: bool
    ):
        opt = O.NewtonMethod(0.1, weight_decay=w, cho_before_update=cho_before_update)
        learnable_pars = LEARNABLE_PARS.copy(deep=True)
        learnable_pars.value = theta = np.random.randn(N_PARAMS)
        learnable_pars.lb = -np.abs(theta) * 100
        learnable_pars.ub = np.abs(theta) * 100
        opt.set_learnable_parameters(learnable_pars)
        g = np.random.randn(N_PARAMS) / 100
        H = np.eye(N_PARAMS)
        status = opt.update(g, H)
        self.assertIsNone(status)
        np.testing.assert_array_equal(learnable_pars.value, theta - 0.1 * g)

    @parameterized.expand(product((0.0, SMALL), (False, True)))
    def test_update__constrained__with_small_bounds(
        self, w: float, cho_before_update: bool
    ):
        opt = O.NewtonMethod(0.1, weight_decay=w, cho_before_update=cho_before_update)
        learnable_pars = LEARNABLE_PARS.copy(deep=True)
        learnable_pars.value = theta = np.random.randn(N_PARAMS)
        lb = learnable_pars.lb = theta - np.abs(theta) * 5e-2
        ub = learnable_pars.ub = theta + np.abs(theta) * 5e-2
        opt.set_learnable_parameters(learnable_pars)
        g = np.random.randn(N_PARAMS) * 100
        H = np.random.randn(N_PARAMS, N_PARAMS)
        H = H @ H.T
        status = opt.update(g, H)
        theta_new = learnable_pars.value
        self.assertIsNone(status)
        self.assertTrue(
            np.where(theta_new >= lb, True, np.isclose(theta_new, lb)).all()
        )
        self.assertTrue(
            np.where(theta_new <= ub, True, np.isclose(theta_new, ub)).all()
        )


class TestAdam(unittest.TestCase):
    @parameterized.expand(product((0, 0.01), (False, True), (False, True)))
    def test(self, weight_decay: float, decouple_weight_decay: bool, amsgrad: bool):
        # prepare data
        betas = tuple(np.random.uniform(0.9, 1.0, size=2))
        eps = np.random.uniform(1e-8, 1e-6)
        lr = np.random.uniform(1e-4, 1e-3)

        # prepare torch elements
        x = torch.linspace(-np.pi, np.pi, 2, dtype=torch.float64)
        xx_torch = x.unsqueeze(-1).pow(torch.tensor([1, 2, 3]))
        y_actual_torch = torch.sin(x)
        model_torch = torch.nn.Linear(3, 1, dtype=torch.float64)
        loss_fn_torch = torch.nn.MSELoss(reduction="sum")
        cls = torch.optim.AdamW if decouple_weight_decay else torch.optim.Adam
        optimizer_torch = cls(
            params=model_torch.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

        # prepare mpcrl elements
        xx_dm = cs.DM(xx_torch.detach().numpy())
        A_sym = cs.MX.sym("A", *model_torch.weight.shape)
        b_sym = cs.MX.sym("b", *model_torch.bias.shape)
        y_pred_sym = xx_dm @ A_sym.T + b_sym
        y_actual_dm = cs.DM(y_actual_torch.detach().clone().numpy())
        loss_sym = cs.sumsqr(y_pred_sym - y_actual_dm)
        p_sym = cs.veccat(A_sym, b_sym)
        dldp_sym = cs.gradient(loss_sym, p_sym)
        model_mpcrl = cs.Function(
            "F", [p_sym], [y_pred_sym, loss_sym, dldp_sym], ["p"], ["y", "l", "dldp"]
        )
        A_mpcrl = model_torch.weight.data.detach().clone().numpy().flatten()
        b_mpcrl = model_torch.bias.data.detach().clone().numpy()
        learnable_pars = LearnableParametersDict(
            [
                LearnableParameter("A", A_mpcrl.size, A_mpcrl),
                LearnableParameter("b", b_mpcrl.size, b_mpcrl),
            ]
        )
        optimizer_mpcrl = O.Adam(
            learning_rate=lr,
            max_percentage_update=1e4,  # test constrained
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decoupled_weight_decay=decouple_weight_decay,
            amsgrad=amsgrad,
        )
        optimizer_mpcrl.set_learnable_parameters(learnable_pars)

        # run test
        cmp = lambda x, y, msg: np.testing.assert_allclose(
            x, y, rtol=1e-6, atol=1e-6, err_msg=msg
        )
        for i in range(20):
            # torch
            y_pred_torch = model_torch(xx_torch).flatten()
            loss_torch = loss_fn_torch(y_pred_torch, y_actual_torch)
            optimizer_torch.zero_grad()
            loss_torch.backward()
            optimizer_torch.step()
            grad_torch = np.concatenate(
                [
                    model_torch.weight.grad.detach().clone().numpy(),
                    model_torch.bias.grad.detach().clone().numpy(),
                ],
                None,
            )

            # mpcrl
            y_pred_mpcrl, loss_mpcrl, grad_mpcrl = model_mpcrl(
                np.concatenate([A_mpcrl, b_mpcrl], None)
            )
            grad_mpcrl = grad_mpcrl.full().flatten()
            status = optimizer_mpcrl.update(grad_mpcrl)
            p_new = learnable_pars.value
            A_mpcrl, b_mpcrl = np.array_split(p_new, [A_mpcrl.size])

            # check
            self.assertIsNone(status)
            cmp(
                y_pred_mpcrl.full().flatten(),
                y_pred_torch.detach().clone().numpy(),
                f"prediction mismatch at iteration {i}",
            )
            cmp(
                float(loss_mpcrl),
                loss_torch.detach().clone().item(),
                f"loss mismatch at iteration {i}",
            )
            cmp(grad_mpcrl, grad_torch, f"gradient mismatch at iteration {i}")
            cmp(
                A_mpcrl,
                model_torch.weight.detach().clone().numpy().reshape(A_mpcrl.shape),
                f"`A` mismatch at iteration {i}",
            )
            cmp(
                b_mpcrl,
                model_torch.bias.detach().clone().numpy().reshape(b_mpcrl.shape),
                f"`b` mismatch at iteration {i}",
            )


class TestRMSprop(unittest.TestCase):
    @parameterized.expand(product((0, 0.1), (0, 0.9), (False, True)))
    def test(self, weight_decay: float, momentum: float, centered: bool):
        # prepare data
        alpha = np.random.uniform(0.9, 0.99)
        eps = np.random.uniform(1e-8, 1e-6)
        lr = np.random.uniform(1e-4, 1e-3)

        # prepare torch elements
        x = torch.linspace(-np.pi, np.pi, 2, dtype=torch.float64)
        xx_torch = x.unsqueeze(-1).pow(torch.tensor([1, 2, 3]))
        y_actual_torch = torch.sin(x)
        model_torch = torch.nn.Linear(3, 1, dtype=torch.float64)
        loss_fn_torch = torch.nn.MSELoss(reduction="sum")
        optimizer_torch = torch.optim.RMSprop(
            params=model_torch.parameters(),
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )

        # prepare mpcrl elements
        xx_dm = cs.DM(xx_torch.detach().numpy())
        A_sym = cs.MX.sym("A", *model_torch.weight.shape)
        b_sym = cs.MX.sym("b", *model_torch.bias.shape)
        y_pred_sym = xx_dm @ A_sym.T + b_sym
        y_actual_dm = cs.DM(y_actual_torch.detach().clone().numpy())
        loss_sym = cs.sumsqr(y_pred_sym - y_actual_dm)
        p_sym = cs.veccat(A_sym, b_sym)
        dldp_sym = cs.gradient(loss_sym, p_sym)
        model_mpcrl = cs.Function(
            "F", [p_sym], [y_pred_sym, loss_sym, dldp_sym], ["p"], ["y", "l", "dldp"]
        )
        A_mpcrl = model_torch.weight.data.detach().clone().numpy().flatten()
        b_mpcrl = model_torch.bias.data.detach().clone().numpy()
        learnable_pars = LearnableParametersDict(
            [
                LearnableParameter("A", A_mpcrl.size, A_mpcrl),
                LearnableParameter("b", b_mpcrl.size, b_mpcrl),
            ]
        )
        optimizer_mpcrl = O.RMSprop(
            learning_rate=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
            max_percentage_update=1e4,  # test constrained
        )
        optimizer_mpcrl.set_learnable_parameters(learnable_pars)

        # run test
        cmp = lambda x, y, msg: np.testing.assert_allclose(
            x, y, rtol=1e-6, atol=1e-6, err_msg=msg
        )
        for i in range(20):
            # torch
            y_pred_torch = model_torch(xx_torch).flatten()
            loss_torch = loss_fn_torch(y_pred_torch, y_actual_torch)
            optimizer_torch.zero_grad()
            loss_torch.backward()
            optimizer_torch.step()
            grad_torch = np.concatenate(
                [
                    model_torch.weight.grad.detach().clone().numpy(),
                    model_torch.bias.grad.detach().clone().numpy(),
                ],
                None,
            )

            # mpcrl
            y_pred_mpcrl, loss_mpcrl, grad_mpcrl = model_mpcrl(
                np.concatenate([A_mpcrl, b_mpcrl], None)
            )
            grad_mpcrl = grad_mpcrl.full().flatten()
            status = optimizer_mpcrl.update(grad_mpcrl)
            p_new = learnable_pars.value
            A_mpcrl, b_mpcrl = np.array_split(p_new, [A_mpcrl.size])

            # check
            self.assertIsNone(status)
            cmp(
                y_pred_mpcrl.full().flatten(),
                y_pred_torch.detach().clone().numpy(),
                f"prediction mismatch at iteration {i}",
            )
            cmp(
                float(loss_mpcrl),
                loss_torch.detach().clone().item(),
                f"loss mismatch at iteration {i}",
            )
            cmp(grad_mpcrl, grad_torch, f"gradient mismatch at iteration {i}")
            cmp(
                A_mpcrl,
                model_torch.weight.detach().clone().numpy().reshape(A_mpcrl.shape),
                f"`A` mismatch at iteration {i}",
            )
            cmp(
                b_mpcrl,
                model_torch.bias.detach().clone().numpy().reshape(b_mpcrl.shape),
                f"`b` mismatch at iteration {i}",
            )


if __name__ == "__main__":
    unittest.main()
