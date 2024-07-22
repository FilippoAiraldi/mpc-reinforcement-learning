from typing import Literal, Optional, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from ..core.schedulers import Scheduler
from .gradient_based_optimizer import GradientBasedOptimizer, LrType, SymType


class GradientDescent(GradientBasedOptimizer[SymType, LrType]):
    r"""First-order Gradient descent optimizer, based on
    :cite:`sutskever_importance_2013` and :class:`torch.optim.SGD`.

    In its basic formulation, this optimizer updates the parameters as

    .. math:: \theta \gets \theta - \alpha g,

    where :math:`\theta` are the learnable parameters, :math:`\alpha` is the learning
    rate (could be extended to the case this is a vector of rates), and :math:`g` is the
    gradient of the loss function w.r.t. the parameters. If momentum or weight decay are
    used, the gradient :math:`g` is modified before using it, but the update rule
    remains the same. However, when considering a constrained parameter space, we need
    to solve a Quadratic Programming (QP) problem to ensure the parameters stay within
    their bounds. For gradient descent, the QP problem is

    .. math::

        \begin{aligned}
            \min_{\Delta\theta} & \quad \frac{1}{2} \lVert \Delta\theta \rVert_2^2 + \alpha g^\top \Delta\theta \\
            \text{s.t.} & \quad \theta_{\text{lower}} \leq \theta + \Delta\theta \leq \theta_{\text{upper}}
        \end{aligned}

    followed by the update :math:`\theta \gets \theta + \Delta\theta`.

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
    momentum : float, optional
        A positive float that specifies the momentum factor. By default, it is set
        to ``0.0``, so no momentum is used.
    dampening : float, optional
        A positive float that specifies the dampening factor for the momentum. By
        default, it is set to ``0.0``, so no dampening is used.
    nesterov : bool, optional
        A boolean that specifies whether to use Nesterov momentum. By default, it is
        set to ``False``.
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

    _order = 1
    _hessian_sparsity = "diag"
    # In GD, the hessian is at most diagonal, i.e., in case we have constraints.

    def __init__(
        self,
        learning_rate: Union[LrType, Scheduler[LrType]],
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        hook: Literal["on_update", "on_episode_end", "on_timestep_end"] = "on_update",
        max_percentage_update: float = float("+inf"),
        bound_consistency: bool = False,
    ) -> None:
        super().__init__(learning_rate, hook, max_percentage_update, bound_consistency)
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self._momentum_buffer = None

    def _first_order_update(
        self, gradient: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], Optional[str]]:
        theta = self.learnable_parameters.value

        # compute candidate update
        dtheta, self._momentum_buffer = _gd(
            theta,
            gradient,
            self._momentum_buffer,
            self.weight_decay,
            self.momentum,
            self.lr_scheduler.value,
            self.dampening,
            self.nesterov,
        )

        # if unconstrained, apply the update directly; otherwise, solve the QP
        solver = self._update_solver
        if solver is None:
            return theta + dtheta, None
        lbx, ubx = self._get_update_bounds(theta)
        sol = solver(h=cs.DM.eye(theta.shape[0]), g=-dtheta, lbx=lbx, ubx=ubx)
        dtheta = np.asarray(sol["x"].elements())
        stats = solver.stats()
        return theta + dtheta, None if stats["success"] else stats["return_status"]


def _gd(
    theta: np.ndarray,
    g: np.ndarray,
    momentum_buffer: Optional[np.ndarray],
    weight_decay: float,
    momentum: float,
    lr: LrType,
    dampening: float,
    nesterov: bool,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Computes the update's change according to the gradient descent algorithm."""
    if weight_decay > 0.0:
        g += weight_decay * theta
    if momentum > 0.0:
        if momentum_buffer is None:
            momentum_buffer = g.copy()
        else:
            momentum_buffer = momentum * momentum_buffer + (1 - dampening) * g
        if nesterov:
            g += momentum * momentum_buffer
        else:
            g = momentum_buffer
    dtheta = -lr * g
    return dtheta, momentum_buffer
