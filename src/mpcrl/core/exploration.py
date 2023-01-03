from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from mpcrl.core.random import np_random
from mpcrl.core.schedulers import Scheduler


class ExplorationStrategy(ABC):
    """Base class for exploration strategies such as greedy, epsilon-greeyd, etc."""

    __slots__: Tuple[str, ...] = ()

    @abstractmethod
    def can_explore(self) -> bool:
        """Computes whether, according to the exploration strategy, the agent should
        explore or not at the current instant.

        Returns
        -------
        bool
            `True` if the agent should explore according to this strategy; otherwise,
            `False`.
        """

    @abstractmethod
    def step(self) -> None:
        """Updates the exploration strength and/or probability, in case the strategy
        supports them (usually, by decaying them over time)."""

    @abstractmethod
    def perturbation(self, *args: Any, **kwargs: Any) -> npt.NDArray[np.double]:
        """Returns a random perturbation."""

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__class__.__name__


class NoExploration(ExplorationStrategy):
    """Strategy where no exploration is allowed at any time or, in other words, the
    policy is always deterministic (only based on the current state, and not perturbed).
    """

    def can_explore(self) -> bool:
        return False

    def step(self) -> None:
        return

    def perturbation(self, *args: Any, **kwargs: Any) -> npt.NDArray[np.double]:
        raise NotImplementedError(
            f"Perturbation not implemented in {self.__class__.__name__}"
        )


class GreedyExploration(ExplorationStrategy):
    """Fully greedy strategy for perturbing the policy, thus inducing exploration. This
    strategy always perturbs randomly the policy."""

    __slots__ = ("strength_scheduler", "np_random")

    def __init__(
        self,
        strength: Union[Scheduler[npt.NDArray[np.double]], npt.NDArray[np.double]],
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the greedy exploration strategy.

        Parameters
        ----------
        strength : scheduler or array/supports-algebraic-operations
            The strength of the exploration. If passed in the form of an
            `mpcrl.schedulers.Scheduler`, then the strength can be scheduled to
            decay/increase every time `exploration.step` is called. If an array or
            something other than a scheduler is passed, then this quantity will get
            wrapped in a base scheduler which will kept it constant.
        seed : int or None, optional
            Number to seed the RNG engine used for randomizing the exploration. By
            default, `None`.
        """
        super().__init__()
        if not isinstance(strength, Scheduler):
            strength = Scheduler(strength)
        self.strength_scheduler = strength
        self.np_random = np_random(seed)

    @property
    def strength(self) -> npt.NDArray[np.double]:
        """Gets the current strength of the exploration strategy."""
        return self.strength_scheduler.value

    def can_explore(self) -> bool:
        return True

    def step(self) -> None:
        """Updates the exploration strength according to its scheduler."""
        self.strength_scheduler.step()

    def perturbation(
        self, method: str, *args: Any, **kwargs: Any
    ) -> npt.NDArray[np.double]:
        """Returns a random perturbation.

        Parameters
        ----------
        method : str
            The name of a method from the ones available to `numpy.random.Generator`,
            e.g., 'random', 'uniform', 'beta', etc.
        args, kwargs
            Args and kwargs with which to call such method.

        Returns
        -------
        array
            An array representing the perturbation.
        """
        return (
            getattr(self.np_random, method)(*args, **kwargs)
            * self.strength_scheduler.value
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(stn={self.strength_scheduler.value})"


class EpsilonGreedyExploration(GreedyExploration):
    """Epsilon-greedy strategy for perturbing the policy, thus inducing exploration.
    This strategy only occasionally perturbs randomly the policy."""

    __slots__ = ("epsilon_scheduler",)

    def __init__(
        self,
        epsilon: Union[Scheduler[float], float],
        strength: Union[Scheduler[npt.NDArray[np.double]], npt.NDArray[np.double]],
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the epsilon-greedy exploration strategy.

        Parameters
        ----------
        epsilon : scheduler or float
            The probability to explore. Should be in range [0, 1]. If passed in the form
            of an `mpcrl.schedulers.Scheduler`, then the probability can be scheduled to
            decay/increase every time `exploration.step` is called. If an array or
            something other than a scheduler is passed, then this quantity will get
            wrapped in a base scheduler which will kept it constant.
        strength : scheduler or array/supports-algebraic-operations
            The strength of the exploration. Can be scheduled, see `epsilon`.
        seed : int or None, optional
            Number to seed the RNG engine used for randomizing the exploration. By
            default, `None`.
        """
        super().__init__(strength, seed)
        if not isinstance(epsilon, Scheduler):
            epsilon = Scheduler(epsilon)
        self.epsilon_scheduler = epsilon

    @property
    def epsilon(self) -> float:
        """Gets the current probability of the exploration strategy."""
        return self.epsilon_scheduler.value

    def can_explore(self) -> bool:
        return self.np_random.random() <= self.epsilon_scheduler.value

    def step(self) -> None:
        """Updates the exploration probability and strength according to their
        schedulers."""
        self.strength_scheduler.step()
        self.epsilon_scheduler.step()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(eps={self.epsilon_scheduler.value},"
            f"stn={self.strength_scheduler.value})"
        )
