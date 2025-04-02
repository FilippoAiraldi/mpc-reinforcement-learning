from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.linalg import cho_solve

from ..core.schedulers import Scheduler
from ..util.math import cholesky_added_multiple_identity
from .gradient_based_optimizer import GradientBasedOptimizer, LrType, SymType


class NewtonMethod(GradientBasedOptimizer[SymType, LrType]):
    r"""Second-order gradient-based Newton's method.

    In constrast to the first-order methods, the Newton's method uses also the Hessian
    of the loss function to compute the update. The unconstrained update is given by

    .. math:: \theta \gets \theta - \alpha H^{-1} g.

    However, we do not directly use the provided Hessian, but rather its Cholesky
    decomposition after having ensured it is positive semi-definite via
    :meth:`cholesky_added_multiple_identity`. As usual, weight decay can be added, but
    for sake of simplicity it is not included in the formula above. In case there are
    constraints on the learnable parameters, the update is solved as a Quadratic
    Programming (QP) problem, which is slower than the unconstrained counterpart. This
    QP takes the form

    .. math::

        \begin{aligned}
            \min_{\Delta\theta} & \quad \frac{1}{2} \Delta\theta^\top H \Delta\theta + \alpha g^\top \Delta\theta \\
            \text{s.t.} & \quad \theta_{\text{lower}} \leq \theta + \Delta\theta \leq \theta_{\text{upper}}
        \end{aligned}

    if ``cho_before_update=False``; otherwise, the objective is
    :math:`\frac{1}{2} \lVert \Delta\theta \rVert_2^2 + \alpha (H^{-1} g)^\top \Delta\theta`.

    Parameters
    ----------
    learning_rate : float or array or :class:`mpcrl.core.schedulers.Scheduler`
        The learning rate of the optimizer. It can be:

        - a float, in case the learning rate must stay constant and is the same for all
          learnable parameters

        - an array, in case the learning rate must stay constant but is different for
          each parameter (should have the same size as the number of learnable
          parameters)

        - a :class:`mpcrl.core.schedulers.Scheduler`, in case the learning rate can vary
          during the learning process (usually, it is set to decay). See the ``hook``
          argument for more details on when this scheduler is stepped.
    weight_decay : float, optional
        A positive float that specifies the decay of the learnable parameters in the
        form of an L2 regularization term. By default, it is set to ``0.0``, so no
        decay/regularization takes place.
    cho_before_update : bool, optional
        Whether to perform a Cholesky's factorization of the hessian in preparation
        of each update. If ``False``, the quadratic form in the QP objective hosts the
        Hessian matrix; else if ``True``, the linear system :math:`H^{-1} g` is first
        solved via Cholesky's factorization, and the QP update's Hessian is downgraded
        to an identity matrix. Only relevant if the update is constrained. By
        default, ``False``.
    cho_maxiter : int, optional
        Maximum number of iterations in the Cholesky's factorization with additive
        multiples of the identity to ensure positive definiteness of the hessian. By
        default, ``1000``.
    cho_solve_kwargs : kwargs for :func:`scipy.linalg.cho_solve`, optional
        The optional kwargs to be passed to :func:`scipy.linalg.cho_solve` to solve
        linear systems with the Hessian's Cholesky decomposition. If ``None``, it is set
        by default to ``cho_solve_kwargs = {"check_finite": False }``. Only relevant if
        no weight decay is given.
    hook : {"on_update", "on_episode_end", "on_timestep_end"}, optional
        Specifies when to step the optimizer's learning rate's scheduler to decay
        its value. This allows to vary the rate over the learning iterations. The
        options are:

        - ``"on_update"`` steps the learning rate after each agent's update

        - ``"on_episode_end"`` steps the learning rate after each episode's end

        - ``"on_timestep_end"`` steps the learning rate after each env's timestep.

        By default, ``"on_update"`` is selected.
    max_percentage_update : float, optional
        A positive float that specifies the maximum percentage change the learnable
        parameters can experience in each update. For example,
        ``max_percentage_update=0.5`` means that the parameters can be updated by up
        to 50% of their current value. By default, it is set to ``+inf``.
        If specified, the update becomes constrained and has to be solved as a QP, which
        is inevitably slower than its unconstrained counterpart (a linear system).
    bound_consistency : bool, optional
        A boolean that, if ``True``, forces the learnable parameters to lie in their
        bounds when updated. This is done via :func:`numpy.clip`. Only beneficial if
        numerical issues arise during updates, e.g., due to the QP solver not being able
        to guarantee bounds.
    """

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
        bound_consistency: bool = False,
    ) -> None:
        if cho_before_update:
            self._hessian_sparsity = "diag"
        super().__init__(learning_rate, hook, max_percentage_update, bound_consistency)
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
        L = cholesky_added_multiple_identity(hessian, maxiter=self.cho_maxiter)

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
    """Computes the update's change according to the Newton's Method."""
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
    """Computes the update's change according to the Newton's Method, which is solved
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
