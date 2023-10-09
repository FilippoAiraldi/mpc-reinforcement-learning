from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.linalg import cho_solve

from mpcrl.core.learning_rate import LearningRate, LrType
from mpcrl.core.schedulers import Scheduler
from mpcrl.optim.gradient_based_optimizer import GradientBasedOptimizer
from mpcrl.util.math import cholesky_added_multiple_identities


class GradientDescent(GradientBasedOptimizer):
    """Gradient descent optimizer."""

    def __init__(
        self,
        learning_rate: Union[LrType, Scheduler[LrType], LearningRate[LrType]],
        max_percentage_update: float = float("+inf"),
        weight_decay: float = 0.0,
        cho_before_update: bool = False,
        cho_maxiter: int = 1000,
        cho_solve_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Instantiates the optimizer.

        Parameters
        ----------
        learning_rate : float/array, scheduler or LearningRate
            The learning rate of the optimizer. A float/array can be passed in case the
            learning rate must stay constant; otherwise, a scheduler can be passed which
            will be stepped `on_update` by default. Otherwise, a `LearningRate` object
            can be passed, allowing to specify both the scheduling and stepping
            strategies of this fundamental hyper-parameter.
        max_percentage_update : float, optional
            A positive float that specifies the maximum percentage change the learnable
            parameters can experience in each update. For example,
            `max_percentage_update=0.5` means that the parameters can be updated by up
            to 50% of their current value. By default, it is set to `+inf`.
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
        """
        super().__init__(learning_rate, max_percentage_update)
        self.weight_decay = weight_decay
        self.cho_before_update = cho_before_update
        self.cho_maxiter = cho_maxiter
        if cho_solve_kwargs is None:
            cho_solve_kwargs = {"check_finite": False}
        self.cho_solve_kwargs = cho_solve_kwargs

    def update(
        self,
        gradient: npt.NDArray[np.floating],
        hessian: Optional[npt.NDArray[np.floating]] = None,
    ) -> tuple[npt.NDArray[np.floating], Optional[str]]:
        if self._update_solver is None:
            if hessian is None:
                return self._do_1st_order_unconstrained_update(gradient), None
            return self._do_2nd_order_unconstrained_update(gradient, hessian), None
        if hessian is None:
            return self._do_1st_order_constrained_update(gradient)
        return self._do_2nd_order_constrained_update(gradient, hessian)

    def _do_1st_order_unconstrained_update(self, g: np.ndarray) -> np.ndarray:
        """Computes a 1st order gradient descent update, with no constraints and,
        optionally, with weight decay."""
        theta = self.learnable_parameters.value
        lr = self.learning_rate.value
        w = self.weight_decay
        dtheta = -lr * g if w <= 0.0 else -(lr * g + w * theta) / (1 + w)
        return theta + dtheta

    def _do_2nd_order_unconstrained_update(
        self, g: np.ndarray, H: np.ndarray
    ) -> np.ndarray:
        """Computes a 2nd order gradient descent update, with no constraints and,
        optionally, with weight decay."""
        theta = self.learnable_parameters.value
        lr = self.learning_rate.value
        w = self.weight_decay
        L = cholesky_added_multiple_identities(H, maxiter=self.cho_maxiter)

        if w <= 0.0:
            dtheta = -lr * cho_solve((L, True), g)
        else:
            dtheta = -np.linalg.solve(
                L @ L.T + w * np.eye(theta.shape[0]), lr * g + w * theta
            )
        return theta + dtheta

    def _do_1st_order_constrained_update(
        self, g: np.ndarray
    ) -> tuple[np.ndarray, Optional[str]]:
        """Computes a 1st order gradient descent update, with constraints and,
        optionally, with weight decay."""
        theta = self.learnable_parameters.value
        lr = self.learning_rate.value
        w = self.weight_decay
        solver = self._update_solver

        G = lr * g if w <= 0.0 else (lr * g + w * theta) / (1 + w)
        H = np.eye(theta.shape[0])
        lbx, ubx = self._get_update_bounds(theta)
        sol = solver(h=H, g=G, lbx=lbx, ubx=ubx)
        dtheta = sol["x"].full().reshape(-1)
        stats = solver.stats()
        return theta + dtheta, None if stats["success"] else stats["return_status"]

    def _do_2nd_order_constrained_update(
        self, g: np.ndarray, H: np.ndarray
    ) -> tuple[np.ndarray, Optional[str]]:
        """Computes a 2nd order gradient descent update, with constraints and,
        optionally, with weight decay."""
        theta = self.learnable_parameters.value
        lr = self.learning_rate.value
        w = self.weight_decay
        solver = self._update_solver

        L = cholesky_added_multiple_identities(H, maxiter=self.cho_maxiter)
        if w <= 0.0:
            if self.cho_before_update:
                G = cho_solve((L, True), lr * g)
                H = np.eye(theta.shape[0])
            else:
                G = lr * g
                H = L @ L.T
        else:
            G = lr * g + w * theta
            H = L @ L.T + w * np.eye(theta.shape[0])
            if self.cho_before_update:
                G = np.linalg.solve(H, G)
                H = np.eye(theta.shape[0])
        lbx, ubx = self._get_update_bounds(theta)
        sol = solver(h=H, g=G, lbx=lbx, ubx=ubx)
        dtheta = sol["x"].full().reshape(-1)
        stats = solver.stats()
        return theta + dtheta, None if stats["success"] else stats["return_status"]
