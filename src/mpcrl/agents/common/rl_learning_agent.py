from abc import ABC
from typing import Any, Generic, TypeVar

from ...optim.gradient_based_optimizer import GradientBasedOptimizer, LrType
from .agent import SymType
from .learning_agent import LearningAgent

ExpType = TypeVar("ExpType")


class RlLearningAgent(
    LearningAgent[SymType, ExpType], ABC, Generic[SymType, ExpType, LrType]
):
    r"""Base abstract class for learning agents that employe gradient-based RL
    strategies to learn/improve the MPC policy. The only difference with the
    :class:`LearningAgent` is that this class accepts the RL task's discount factor and
    a gradient-based optimizer that dictates how the learnable parameters are updated.

    Parameters
    ----------
    discount_factor : float
        In RL, the factor that discounts future rewards in favor of immediate rewards.
        Usually denoted as :math:`\gamma`. It should satisfy :math:`\gamma \in (0, 1]`.
    optimizer : GradientBasedOptimizer
        A gradient-based optimizer (e.g., :class:`optim.GradientDescent`) to
        compute the updates of the learnable parameters, based on the current
        gradient-based RL algorithm.
    kwargs
        Additional arguments to be passed to :class:`LearningAgent`.
    """

    def __init__(
        self,
        discount_factor: float,
        optimizer: GradientBasedOptimizer[SymType, LrType],
        **kwargs: Any,
    ) -> None:
        self.discount_factor = discount_factor
        self.optimizer = optimizer
        super().__init__(**kwargs)
        self.optimizer.set_learnable_parameters(self._learnable_pars)

    def _establish_callback_hooks(self) -> None:
        super()._establish_callback_hooks()
        optim = self.optimizer
        optimizer_hook = optim.hook
        if optimizer_hook is not None:
            self._hook_callback(repr(optim), optimizer_hook, optim.step)
