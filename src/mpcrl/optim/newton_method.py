from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.linalg import cho_solve

from ..core.schedulers import Scheduler
from ..util.math import cholesky_added_multiple_identities
from .gradient_based_optimizer import GradientBasedOptimizer, LrType, SymType


class NetwonMethod(GradientBasedOptimizer[SymType, LrType]):
    """Netwon Method."""

    _order = 2
    _hessian_sparsity = "dense"

    def __init__(
        self,
        learning_rate: Union[LrType, Scheduler[LrType]],
        weight_decay: float = 0.0,
        cho_before_update: bool = False,
        cho_maxiter: int = 1000,
        cho_solve_kwargs: Optional[dict[str, Any]] = None,
        hook: Literal["on_update", "on_episode_end", "on_timestep_end"] = "on_update",
        max_percentage_update: float = float("+inf"),
    ) -> None:
        """Instantiates the optimizer.

        Parameters
        ----------
        learning_rate : float/array, scheduler
            The learning rate of the optimizer. A float/array can be passed in case the
            learning rate must stay constant; otherwise, a scheduler can be passed which
            will be stepped `on_update` by default (see `hook` argument).
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
            Only relevant if the update is  constrained. By default, `False`.
        cho_maxiter : int, optional
            Maximum number of iterations in the Cholesky's factorization with additive
            multiples of the identity to ensure positive definiteness of the hessian. By
            default, `1000`.
        cho_solve_kwargs : kwargs for `scipy.linalg.cho_solve`, optional
            The optional kwargs to be passed to `scipy.linalg.cho_solve` to solve linear
            systems with the hessian's Cholesky decomposition. If `None`, it is
            equivalent to `cho_solve_kwargs = {'check_finite': False }`.
        hook : {'on_update', 'on_episode_end', 'on_timestep_end'}, optional
            Specifies when to step the optimizer's learning rate's scheduler to decay
            its value (see `step` method also). This allows to vary the rate over the
            learning iterations. The options are:
             - `on_update` steps the learning rate after each agent's update
             - `on_episode_end` steps the learning rate after each episode's end
             - `on_timestep_end` steps the learning rate after each env's timestep.

            By default, 'on_update' is selected.
        max_percentage_update : float, optional
            A positive float that specifies the maximum percentage change the learnable
            parameters can experience in each update. For example,
            `max_percentage_update=0.5` means that the parameters can be updated by up
            to 50% of their current value. By default, it is set to `+inf`. If
            specified, the update becomes constrained and has to be solved as a QP,
            which is inevitably slower than its unconstrained counterpart.
        """
        if cho_before_update:
            self._hessian_sparsity = "diag"
        super().__init__(learning_rate, hook, max_percentage_update)
        self.weight_decay = weight_decay
        self.cho_before_update = cho_before_update
        self.cho_maxiter = cho_maxiter
        if cho_solve_kwargs is None:
            cho_solve_kwargs = {"check_finite": False}
        self.cho_solve_kwargs = cho_solve_kwargs

    def _second_order_update(
        self, gradient: npt.NDArray[np.floating], hessian: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], Optional[str]]:
        theta = self.learnable_parameters.value
        lr = self.lr_scheduler.value
        w = self.weight_decay
        cho_kw = self.cho_solve_kwargs
        L = cholesky_added_multiple_identities(hessian, maxiter=self.cho_maxiter)

        # if unconstrained, apply the update directly; otherwise, solve the QP
        solver = self._update_solver
        if solver is None:
            dtheta = _nm_unconstrained(theta, gradient, L, lr, w, cho_kw)
            return theta + dtheta, None
        H, G = _nm_constrained(
            theta, gradient, hessian, L, lr, w, self.cho_before_update, cho_kw
        )
        lbx, ubx = self._get_update_bounds(theta)
        sol = solver(h=H, g=G, lbx=lbx, ubx=ubx)
        dtheta = np.asarray(sol["x"].elements())
        stats = solver.stats()
        return theta + dtheta, None if stats["success"] else stats["return_status"]


def _nm_unconstrained(
    theta: np.ndarray,
    g: np.ndarray,
    L: np.ndarray,
    lr: LrType,
    weight_decay: float,
    cho_solve_kwargs: dict,
) -> np.ndarray:
    """Computes the update's change according to the Netwon's Method."""
    if weight_decay <= 0.0:
        return -lr * cho_solve((L, True), g, **cho_solve_kwargs)
    return -np.linalg.solve(
        L @ L.T + weight_decay * np.eye(theta.shape[0]), lr * g + weight_decay * theta
    )


def _nm_constrained(
    theta: np.ndarray,
    g: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    lr: LrType,
    weight_decay: float,
    cho_before_update: bool,
    cho_solve_kwargs: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the update's change according to the Netwon's Method, which is solved
    numerically due to the presence of constraints."""
    if weight_decay <= 0.0:
        if cho_before_update:
            G = cho_solve((L, True), lr * g, **cho_solve_kwargs)
            H = np.eye(theta.shape[0])
        else:
            G = lr * g
            H = L @ L.T
    else:
        G = lr * g + weight_decay * theta
        H = L @ L.T + weight_decay * np.eye(theta.shape[0])
        if cho_before_update:
            G = np.linalg.solve(H, G)
            H = np.eye(theta.shape[0])
    return H, G
