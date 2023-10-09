from typing import Optional, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from mpcrl.core.learning_rate import LearningRate, LrType
from mpcrl.core.schedulers import Scheduler
from mpcrl.optim.gradient_based_optimizer import GradientBasedOptimizer


class GradientDescent(GradientBasedOptimizer):
    """First-order Gradient descent optimizer, based on [1,2].

    References
    ----------
    [1] Sutskever, I., Martens, J., Dahl, G. and Hinton, G., 2013, May. On the
        importance of initialization and momentum in deep learning. In International
        conference on machine learning (pp. 1139-1147). PMLR.
    [2] SGD - PyTorch 2.1 documentation.
        https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    """

    _hessian_sparsity = "diag"
    """In GD, the hessian is at most diagonal, i.e., in case we have constraints."""

    def __init__(
        self,
        learning_rate: Union[LrType, Scheduler[LrType], LearningRate[LrType]],
        max_percentage_update: float = float("+inf"),
        weight_decay: float = 0.0,
        #
        momentum: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
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
            to 50% of their current value. By default, it is set to `+inf`. If
            specified, the update becomes constrained and has to be solved as a QP,
            which is inevitably slower than its unconstrained counterpart.
        weight_decay : float, optional
            A positive float that specifies the decay of the learnable parameters in the
            form of an L2 regularization term. By default, it is set to `0.0`, so no
            decay/regularization takes place.
        momentum : float, optional
            A positive float that specifies the momentum factor. By default, it is set
            to `0.0`, so no momentum is used.
        dampening : float, optional
            A positive float that specifies the dampening factor for the momentum. By
            default, it is set to `0.0`, so no dampening is used.
        """
        super().__init__(learning_rate, max_percentage_update)
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self._momentum_buffer = None

    def update(
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
            self.learning_rate.value,
            self.dampening,
            self.nesterov,
        )

        # if unconstrained, apply the update directly; otherwise, solve the QP
        solver = self._update_solver
        if solver is None:
            return theta + dtheta, None
        lbx, ubx = self._get_update_bounds(theta)
        sol = solver(h=cs.DM.eye(theta.shape[0]), g=-dtheta, lbx=lbx, ubx=ubx)
        dtheta = sol["x"].full().reshape(-1)
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
    if weight_decay >= 0.0:
        g += weight_decay * theta
    if momentum >= 0.0:
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
