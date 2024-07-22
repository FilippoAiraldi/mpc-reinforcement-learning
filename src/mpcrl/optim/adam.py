from typing import Literal, Optional, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from ..core.parameters import LearnableParametersDict, SymType
from ..core.schedulers import Scheduler
from .gradient_based_optimizer import GradientBasedOptimizer, LrType


class Adam(GradientBasedOptimizer[SymType, LrType]):
    r"""First-order gradient-based Adam and AdamW optimizers, based on
    :cite:`kingma_adam_2014` and :class:`torch.optim.Adam`, and
    :cite:`loshchilov_decoupled_2017` and :class:`torch.optim.AdamW`, respectively.
    AMSGrad is also supported :cite:`reddi_convergence_2019`.

    While there is an abundance of resources to understand the Adam optimizer and its
    variants, how do we address the constraints on the learnable parameters? Similarly
    to :class:`mpcrl.optim.GradientDescent`, once the update rule has calculated the
    change :math:`g` in the parameters (in an unconstrained fashion), we force the
    constraints via the Quadratic Programming (QP) problem

    .. math::

        \begin{aligned}
            \min_{\Delta\theta} & \quad \frac{1}{2} \lVert \Delta\theta \rVert_2^2 + \alpha g^\top \Delta\theta \\
            \text{s.t.} & \quad \theta_{\text{lower}} \leq \theta + \Delta\theta \leq \theta_{\text{upper}}
        \end{aligned}

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
    betas : tuple of 2 floats, optional
        Coefficients used for computing running averages of gradient and its square.
        By default, they are set to ``(0.9, 0.999)``.
    eps : float, optional
        Term added to the denominator to improve numerical stability. By default, it
        is set to ``1e-8``.
    weight_decay : float, optional
        A positive float that specifies the decay of the learnable parameters in the
        form of an L2 regularization term. By default, it is set to ``0.0``, so no
        decay/regularization takes place.
    decoupled_weight_decay : bool, optional
        If ``False``, the optimizer is _Adam_. Otherwise, it is _AdamW_. By default, it
        is ``False``.
    amsgrad : bool, optional
        If ``True``, uses the AMSGrad variant. By default, it is ``False``.
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
    # In Adam, hessian is at most diagonal, i.e., in case we have constraints

    def __init__(
        self,
        learning_rate: Union[LrType, Scheduler[LrType]],
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
        amsgrad: bool = False,
        hook: Literal["on_update", "on_episode_end", "on_timestep_end"] = "on_update",
        max_percentage_update: float = float("+inf"),
        bound_consistency: bool = False,
    ) -> None:
        super().__init__(learning_rate, hook, max_percentage_update, bound_consistency)
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.decoupled_weight_decay = decoupled_weight_decay
        self.amsgrad = amsgrad

    def set_learnable_parameters(self, pars: LearnableParametersDict[SymType]) -> None:
        super().set_learnable_parameters(pars)
        # initialize also running averages
        n = pars.size
        self._exp_avg = np.zeros(n, dtype=float)
        self._exp_avg_sq = np.zeros(n, dtype=float)
        self._max_exp_avg_sq = np.zeros(n, dtype=float) if self.amsgrad else None
        self._step = 0

    def _first_order_update(
        self, gradient: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], Optional[str]]:
        theta = self.learnable_parameters.value

        # compute candidate update
        weight_decay = self.weight_decay
        lr = self.lr_scheduler.value
        if weight_decay > 0.0:
            if self.decoupled_weight_decay:  # i.e., AdamW
                theta = theta * (1 - weight_decay * lr)
            else:
                gradient = gradient + weight_decay * theta
        self._step += 1
        dtheta, self._exp_avg, self._exp_avg_sq, self._max_exp_avg_sq = _adam(
            self._step,
            gradient,
            self._exp_avg,
            self._exp_avg_sq,
            lr,
            self.beta1,
            self.beta2,
            self.eps,
            self._max_exp_avg_sq,
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


def _adam(
    step: int,
    g: np.ndarray,
    exp_avg: np.ndarray,
    exp_avg_sq: np.ndarray,
    lr: LrType,
    beta1: float,
    beta2: float,
    eps: float,
    max_exp_avg_sq: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Computes the update's change according to Adam algorithm."""
    exp_avg = beta1 * exp_avg + (1 - beta1) * g
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * np.square(g)

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    step_size = lr / bias_correction1
    bias_correction2_sqrt = np.sqrt(bias_correction2)

    if max_exp_avg_sq is None:
        denom = np.sqrt(exp_avg_sq) / bias_correction2_sqrt + eps
    else:  # i.e., AMSGrad
        max_exp_avg_sq = np.maximum(max_exp_avg_sq, exp_avg_sq)
        denom = np.sqrt(max_exp_avg_sq) / bias_correction2_sqrt + eps
    dtheta = -step_size * (exp_avg / denom)
    return dtheta, exp_avg, exp_avg_sq, max_exp_avg_sq
