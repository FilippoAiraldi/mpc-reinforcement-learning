from abc import ABC
from typing import Any, Generic, Optional, TypeVar

import numpy as np

from mpcrl.agents.agent import SymType
from mpcrl.agents.learning_agent import LearningAgent
from mpcrl.core.learning_rate import LrType
from mpcrl.optim.gradient_based_optimizer import GradientBasedOptimizer

ExpType = TypeVar("ExpType")


class RlLearningAgent(
    LearningAgent[SymType, ExpType], ABC, Generic[SymType, ExpType, LrType]
):
    """Base class for learning agents that employe gradient-based RL strategies to
    learn/improve the MPC policy."""

    def __init__(
        self, discount_factor: float, optimizer: GradientBasedOptimizer, **kwargs: Any
    ) -> None:
        """Instantiates the RL learning agent.

        Parameters
        ----------
        discount_factor : float
            In RL, the factor that discounts future rewards in favor of immediate
            rewards. Usually denoted as `\\gamma`. Should be a number in (0, 1).
        optimizer : GradientBasedOptimizer
            A gradient-based optimizer (e.g., `mpcrl.optim.GradientDescent`) to compute
            the updates of the learnable parameters, based on the current gradient-based
            RL algorithm.
        kwargs
            Additional arguments to be passed to `LearningAgent`.
        """
        self.discount_factor = discount_factor
        self.optimizer = optimizer
        super().__init__(**kwargs)
        self.optimizer.set_learnable_parameters(self._learnable_pars)

    def establish_callback_hooks(self) -> None:
        super().establish_callback_hooks()
        lr = self.optimizer.learning_rate
        lr_hook = lr.hook
        if lr_hook is not None:
            self.hook_callback(repr(lr), lr_hook, lr.step)

    def _do_gradient_update(
        self, gradient: np.ndarray, hessian: Optional[np.ndarray] = None
    ) -> Optional[str]:
        """Internal utility to call the optimizer and perform the gradient update."""
        theta_new, status = (
            self.optimizer.update(gradient)
            if hessian is None
            else self.optimizer.update(gradient, hessian)
        )
        # theta_new = np.clip(theta_new, self._learnable_pars.lb, self._learnable_pars.ub)
        self._learnable_pars.update_values(theta_new)
        return status
