from typing import Literal, Optional, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from ..core.parameters import LearnableParametersDict, SymType
from ..core.schedulers import Scheduler
from .gradient_based_optimizer import GradientBasedOptimizer, LrType


class Adam(GradientBasedOptimizer[SymType, LrType]):
    """Adam and AdamW optimizers, based on [1,2] and [3,4], respectively. AMSGrad is
    also supported [5].

    References
    ----------
    [1] Kingma, D.P. and Ba, J., 2014. Adam: A method for stochastic optimization.
        arXiv preprint arXiv:1412.6980.
    [2] Loshchilov, I. and Hutter, F., 2017. Decoupled weight decay regularization.
        arXiv preprint arXiv:1711.05101.
    [3] Adam - PyTorch 2.1 documentation.
        https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    [4] AdamW - PyTorch 2.1 documentation.
        https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    [5] Reddi, S.J., Kale, S. and Kumar, S., 2019. On the convergence of adam and
        beyond. arXiv preprint arXiv:1904.09237.
    """

    _order = 1
    _hessian_sparsity = "diag"
    """In Adam, hessian is at most diagonal, i.e., in case we have constraints."""

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
    ) -> None:
        """Instantiates the optimizer.

        Parameters
        ----------
        learning_rate : float/array, scheduler
            The learning rate of the optimizer. A float/array can be passed in case the
            learning rate must stay constant; otherwise, a scheduler can be passed which
            will be stepped `on_update` by default (see `hook` argument).
        betas : tuple of 2 floats, optional
            Coefficients used for computing running averages of gradient and its square.
            By default, they are set to `(0.9, 0.999)`.
        eps : float, optional
            Term added to the denominator to improve numerical stability. By default, it
            is set to `1e-8`.
        weight_decay : float, optional
            A positive float that specifies the decay of the learnable parameters in the
            form of an L2 regularization term. By default, it is set to `0.0`, so no
            decay/regularization takes place.
        decoupled_weight_decay : bool, optional
            If `False`, the optimizer is Adam. Otherwise, it is `AdamW`. By default, it
            is `False`.
        amsgrad : bool, optional
            If `True`, uses the AMSGrad variant. By default, it is `False`.
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
        super().__init__(learning_rate, hook, max_percentage_update)
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
