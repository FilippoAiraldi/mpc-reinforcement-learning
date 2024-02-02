from abc import ABC
from typing import Any, Generic, TypeVar

from ...optim.gradient_based_optimizer import GradientBasedOptimizer, LrType
from .agent import SymType
from .learning_agent import LearningAgent

ExpType = TypeVar("ExpType")


class RlLearningAgent(
    LearningAgent[SymType, ExpType], ABC, Generic[SymType, ExpType, LrType]
):
    """Base class for learning agents that employe gradient-based RL strategies to
    learn/improve the MPC policy."""

    def __init__(
        self,
        discount_factor: float,
        optimizer: GradientBasedOptimizer[SymType, LrType],
        **kwargs: Any,
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
        optim = self.optimizer
        optimizer_hook = optim.hook
        if optimizer_hook is not None:
            self.hook_callback(repr(optim), optimizer_hook, optim.step)
