from abc import ABC
from typing import Any, Generic, Optional, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt
from scipy.linalg import cho_solve

from mpcrl.agents.agent import SymType
from mpcrl.agents.learning_agent import LearningAgent
from mpcrl.core.learning_rate import LearningRate, LrType
from mpcrl.core.schedulers import Scheduler
from mpcrl.util.math import cholesky_added_multiple_identities

ExpType = TypeVar("ExpType")


class RlLearningAgent(
    LearningAgent[SymType, ExpType], ABC, Generic[SymType, ExpType, LrType]
):
    """Base class for learning agents that employe gradient-based RL strategies to
    learn/improve the MPC policy."""

    def __init__(
        self,
        discount_factor: float,
        learning_rate: Union[LrType, Scheduler[LrType], LearningRate[LrType]],
        max_percentage_update: float = float("+inf"),
        weight_decay: float = 0.0,
        cho_before_update: bool = False,
        cho_maxiter: int = 1000,
        cho_solve_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Instantiates the RL learning agent.

        Parameters
        ----------
        discount_factor : float
            In RL, the factor that discounts future rewards in favor of immediate
            rewards. Usually denoted as `\\gamma`. Should be a number in (0, 1).
        learning_rate : float/array, scheduler or LearningRate
            The learning rate of the algorithm. A float/array can be passed in case the
            learning rate must stay constant; otherwise, a scheduler can be passed which
            will be stepped `on_update` by default. Otherwise, a LearningRate can be
            passed, allowing to specify both the scheduling and stepping strategies of
            the learning rate.
        max_percentage_update : float, optional
            A positive float that specifies the maximum percentage the parameters can be
            changed during each update. For example, `max_percentage_update=0.5` means
            that the parameters can be updated by up to 50% of their current value. By
            default, it is set to `+inf`.
        weight_decay : float, optional
            A positive float that specifies the decay of the learnable parameters in the
            form of an L2 regularization term. By default, it is set to `0.0`, so no
            decay/regularization takes place.
        cho_before_update : bool, optional
            Whether to perform a Cholesky's factorization of the hessian in preparation
            of each update. If `False`, the QP update's objective is
            ```math
                min 1/2 * dtheta' * H * dtheta + (lr * g)' * dtheta
            ```
            else, if `True`, the objective is
            ```math
                min 1/2 * ||dtheta||^2' + (lr * H^-1 * g)' * dtheta
            ```
            where the hessian linear system is performed via Cholesky's factorization.
            Only relevant if the RL algorithm uses hessian info. By default, `False`.
        cho_maxiter : int, optional
            Maximum number of iterations in the Cholesky's factorization with additive
            multiples of the identity to ensure positive definiteness of the hessian. By
            default, `1000`. Only used if the algorithm exploits the hessian.
        cho_solve_kwargs : kwargs for scipy.linalg.cho_solve, optional
            The optional kwargs to be passed to `scipy.linalg.cho_solve` to solve for
            the inversion of the hessian. If `None`, it is equivalent to
            `cho_solve_kwargs = {'check_finite': False }`. Only used if the algorithm
            exploits the hessian.
        kwargs
            Additional arguments to be passed to `LearningAgent`.
        """
        self.discount_factor = discount_factor
        if not isinstance(learning_rate, LearningRate):
            learning_rate = LearningRate(learning_rate, "on_update")
        self._learning_rate: LearningRate[LrType] = learning_rate
        self.max_percentage_update = max_percentage_update
        self.weight_decay = weight_decay
        self.cho_before_update = cho_before_update
        self.cho_maxiter = cho_maxiter
        if cho_solve_kwargs is None:
            cho_solve_kwargs = {"check_finite": False}
        self.cho_solve_kwargs = cho_solve_kwargs
        super().__init__(**kwargs)
        self._update_solver = self._init_update_solver()

    @property
    def learning_rate(self) -> LrType:
        """Gets the learning rate of the learning agent."""
        return self._learning_rate.value

    def establish_callback_hooks(self) -> None:
        super().establish_callback_hooks()
        lr_hook = self._learning_rate.hook
        if lr_hook is not None:
            self.hook_callback(
                repr(self._learning_rate),
                lr_hook,
                lambda *_, **__: self._learning_rate.step(),
            )

    def _init_update_solver(self) -> Optional[cs.Function]:
        """Internal utility to initialize the update solver, in particular, a QP solver.
        If the update is unconstrained, then no solver is initialized, i.e., `None` is
        returned."""
        if (
            self.max_percentage_update == float("+inf")
            and np.isneginf(self._learnable_pars.lb).all()
            and np.isposinf(self._learnable_pars.ub).all()
        ):
            return None

        sym_type = cs.MX
        n_p = self._learnable_pars.size
        dtheta = sym_type.sym("dtheta", n_p, 1)
        J = sym_type.sym("J", n_p, 1)
        H = sym_type.sym("H", n_p, n_p)
        f = 0.5 * cs.bilin(H, dtheta) + cs.dot(J, dtheta)
        qp = {"x": dtheta, "f": f, "p": cs.veccat(J, H)}
        opts = {
            "expand": True,
            "print_info": False,
            "print_iter": False,
            "print_header": False,
            "error_on_fail": False,
            "max_iter": 2000,
        }
        return cs.qpsol(f"qpsol_{self.name}", "qrqp", qp, opts)

    def _get_update_bounds(
        self,
        theta: npt.NDArray[np.floating],
        eps: float = 0.1,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Internal utility to retrieve the current bounds on the QP solver for an
        update. Only called if the update problem is not unconstrained, i.e., there are
        either some lb or ub, or a maximum percentage update"""
        lb = self._learnable_pars.lb
        ub = self._learnable_pars.ub
        perc = self.max_percentage_update
        if perc == float("+inf"):
            return lb, ub
        max_update_delta = np.maximum(np.abs(perc * theta), eps)
        lbx = np.maximum(lb - theta, -max_update_delta)
        ubx = np.minimum(ub - theta, +max_update_delta)
        return lbx, ubx

    def _do_gradient_update(
        self, g: npt.NDArray[np.floating], H: Optional[npt.NDArray[np.floating]]
    ) -> Optional[str]:
        """Internal utility to do the actual gradient update by either calling the QP
        solver or by updating the parameters maually."""
        solver = self._update_solver
        theta = self._learnable_pars.value  # current values of parameters
        lr = self.learning_rate
        w = self.weight_decay
        no_decay = w <= 0.0
        no_hessian = H is None

        # if solver is None, then we do a manual update
        if solver is None:
            if no_hessian:
                if no_decay:
                    dtheta = -lr * g
                else:
                    dtheta = -(lr * g + w * theta) / (1 + w)
            else:
                L = cholesky_added_multiple_identities(H, maxiter=self.cho_maxiter)
                if no_decay:
                    dtheta = -lr * cho_solve((L, True), g)
                else:
                    dtheta = -np.linalg.solve(
                        L @ L.T + w * np.eye(theta.shape[0]), lr * g + w * theta
                    )
            self._learnable_pars.update_values(theta + dtheta)
            return None

        # if solver is not None, then we run the QP update. We can either pre-compute
        # the factorization of the hessian, or pass it directly to the solver
        if no_hessian:
            if no_decay:
                J = lr * g
            else:
                J = (lr * g + w * theta) / (1 + w)
            H = np.eye(theta.shape[0])
        else:
            L = cholesky_added_multiple_identities(H, maxiter=self.cho_maxiter)
            if no_decay:
                if self.cho_before_update:
                    J = cho_solve((L, True), lr * g)
                    H = np.eye(theta.shape[0])
                else:
                    J = lr * g
                    H = L @ L.T
            else:
                J = lr * g + w * theta
                H = L @ L.T + w * np.eye(theta.shape[0])
                if self.cho_before_update:
                    J = np.linalg.solve(H, J)
                    H = np.eye(theta.shape[0])
        lbx, ubx = self._get_update_bounds(theta)
        sol = solver(p=np.concatenate((J, H.T), None), lbx=lbx, ubx=ubx)
        dtheta = np.clip(sol["x"].full()[:, 0], lbx, ubx)
        self._learnable_pars.update_values(theta + dtheta)
        stats = solver.stats()
        return None if stats["success"] else stats["return_status"]
