from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt

from mpcrl.core.random import np_random


class ExplorationStrategy(ABC):
    """Base class for exploration strategies such as greedy, epsilon-greeyd, etc."""

    __slots__: List[str] = []

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
    def decay(self) -> None:
        """Updats the exploration strength and/or probability, in case the strategy
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

    def decay(self) -> None:
        return

    def perturbation(self, *args: Any, **kwargs: Any) -> npt.NDArray[np.double]:
        return np.nan  # type: ignore


class GreedyExploration(ExplorationStrategy):
    """Fully greedy strategy for perturbing the policy, thus inducing exploration. This
    strategy always perturbs randomly the policy."""

    __slots__ = (
        "strength",
        "strength_decay_rate",
        "np_random",
    )

    def __init__(
        self,
        strength: npt.NDArray[np.double],
        strength_decay_rate: npt.NDArray[np.double],
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the greedy exploration strategy.

        Parameters
        ----------
        strength : npt.NDArray[np.double]
            The strength of the exploration.
        strength_decay_rate : float
            Multiplicative rate at which the exploration strength decays over time, by
            calling the method `decay`. Should be smaller than 1 for a decreasing
            exploration strategy.
        seed : int or None, optional
            Number to seed the RNG engine used for randomizing the exploration. By
            default, `None`.
        """
        super().__init__()
        self.strength = strength
        self.strength_decay_rate = strength_decay_rate
        self.np_random = np_random(seed)[0]

    def can_explore(self) -> bool:
        return True

    def decay(self) -> None:
        self.strength *= self.strength_decay_rate  # type: ignore

    def perturbation(  # type: ignore
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
        return getattr(self.np_random, method)(*args, **kwargs) * self.strength

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(stn={self.strength},"
            f"stn_decay={self.strength_decay_rate})"
        )


class EpsilonGreedyExploration(GreedyExploration):
    """Epsilon-greedy strategy for perturbing the policy, thus inducing exploration.
    This strategy only occasionally perturbs randomly the policy."""

    __slots__ = (
        "epsilon",
        "epsilon_decay_rate",
    )

    def __init__(
        self,
        epsilon: float,
        strength: npt.NDArray[np.double],
        epsilon_decay_rate: float,
        strength_decay_rate: Optional[npt.NDArray[np.double]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the epsilon-greedy exploration strategy.

        Parameters
        ----------
        epsilon : float
            The probability to explore. Should be in range (0, 1), extrema excluded.
        strength : npt.NDArray[np.double]
            The strength of the exploration.
        epsilon_decay_rate : float
            Multiplicative rate at which the exploration probability `epsilon` decays
            over time, by calling the method `decay`. Should be smaller than 1 for a
            decreasing exploration strategy.
        strength_decay_rate : float, optional
            Multiplicative rate at which the exploration strength decays over time, by
            calling the method `decay`. Should be smaller than 1 for a decreasing
            exploration strategy. By default, if `None`, it is equal to the epsilon
            decay rate.
        seed : int or None, optional
            Number to seed the RNG engine used for randomizing the exploration. By
            default, `None`.
        """
        if strength_decay_rate is None:
            strength_decay_rate = epsilon_decay_rate  # type: ignore
        super().__init__(strength, strength_decay_rate, seed)  # type: ignore
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate

    def can_explore(self) -> bool:
        return self.np_random.random() > self.strength  # type: ignore

    def decay(self) -> None:
        super().decay()  # decays only the strength
        self.epsilon *= self.epsilon_decay_rate

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(eps={self.epsilon},stn={self.strength},"
            f"eps_decay={self.epsilon_decay_rate},stn_decay={self.strength_decay_rate})"
        )
