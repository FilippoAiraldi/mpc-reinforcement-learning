from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt

from ..util.seeding import RngType
from .schedulers import NoScheduling, Scheduler


class ExplorationStrategy(ABC):
    """Base class for exploration strategies such as greedy, epsilon-greeyd, etc."""

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
    def step(self, *args: Any, **kwargs: Any) -> None:
        """Updates the exploration strength and/or probability, in case the strategy
        supports them (usually, by decaying them over time)."""

    @abstractmethod
    def perturbation(self, *args: Any, **kwargs: Any) -> npt.NDArray[np.floating]:
        """Returns a random perturbation."""

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        hook: Optional[str] = getattr(self, "hook", None)
        if hook is None:
            return self.__class__.__name__
        return f"{self.__class__.__name__}(hook='{hook}')"


class NoExploration(ExplorationStrategy):
    """Strategy where no exploration is allowed at any time or, in other words, the
    policy is always deterministic (only based on the current state, and not perturbed).
    """

    def can_explore(self) -> bool:
        return False

    def step(self, *_, **__) -> None:
        return

    def perturbation(self, *args: Any, **kwargs: Any) -> npt.NDArray[np.floating]:
        raise NotImplementedError(
            f"Perturbation not implemented in {self.__class__.__name__}"
        )


class GreedyExploration(ExplorationStrategy):
    """Fully greedy strategy for perturbing the policy, thus inducing exploration. This
    strategy always perturbs randomly the policy."""

    def __init__(
        self,
        strength: Union[Scheduler[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        hook: Literal["on_update", "on_episode_end", "on_timestep_end"] = "on_update",
        seed: RngType = None,
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
        hook : {'on_update', 'on_episode_end', 'on_timestep_end'}, optional
            Specifies to which callback to hook, i.e., when to step the exploration's
            schedulers (if any) to, e.g., decay the chances of exploring or the
            perturbation strength (see `step` method also). The options are:
             - `on_update` steps the exploration after each agent's update
             - `on_episode_end` steps the exploration after each episode's end
             - `on_timestep_end` steps the exploration after each env's timestep.

            By default, 'on_update' is selected.
        seed : None, int, array_like[ints], SeedSequence, BitGenerator, Generator
            Number to seed the RNG engine used for randomizing the exploration. By
            default, `None`.
        """
        super().__init__()
        self._hook = hook
        if not isinstance(strength, Scheduler):
            strength = NoScheduling[npt.NDArray[np.floating]](strength)
        self.strength_scheduler = strength
        self.reset(seed)

    @property
    def hook(self) -> Optional[str]:
        """Specifies to which callback to hook, i.e., when to step the exploration's
        schedulers (if any) to, e.g., decay the chances of exploring or the perturbation
        strength (see `step` method also). Can be `None` in case no hook is needed."""
        # return hook only if the strength scheduler requires to be stepped
        return None if isinstance(self.strength_scheduler, NoScheduling) else self._hook

    @property
    def strength(self) -> npt.NDArray[np.floating]:
        """Gets the current strength of the exploration strategy."""
        return self.strength_scheduler.value

    def reset(self, seed: RngType = None) -> None:
        """Resets the exploration RNG."""
        self.np_random = np.random.default_rng(seed)

    def can_explore(self) -> bool:
        return True

    def step(self, *_, **__) -> None:
        """Updates the exploration strength according to its scheduler."""
        self.strength_scheduler.step()

    def perturbation(
        self, method: str, *args: Any, **kwargs: Any
    ) -> npt.NDArray[np.floating]:
        """Returns a random perturbation.

        Parameters
        ----------
        method : str
            The name of a method from the ones available to `numpy.random.Generator`,
            e.g., 'random', 'normal', etc.
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
        hook = self.hook
        hookstr = "None" if hook is None else f"'{hook}'"
        return (
            f"{self.__class__.__name__}(stn={self.strength_scheduler.value},"
            f"hook={hookstr})"
        )


class EpsilonGreedyExploration(GreedyExploration):
    """Epsilon-greedy strategy for perturbing the policy, thus inducing exploration.
    This strategy only occasionally perturbs randomly the policy."""

    def __init__(
        self,
        epsilon: Union[Scheduler[float], float],
        strength: Union[Scheduler[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        hook: Literal["on_update", "on_episode_end", "on_timestep_end"] = "on_update",
        seed: RngType = None,
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
        hook : {'on_update', 'on_episode_end', 'on_timestep_end'}, optional
            Specifies to which callback to hook, i.e., when to step the exploration's
            schedulers (if any) to, e.g., decay the chances of exploring or the
            perturbation strength (see `step` method also). The options are:
             - `on_update` steps the exploration after each agent's update
             - `on_episode_end` steps the exploration after each episode's end
             - `on_timestep_end` steps the exploration after each env's timestep.

            By default, 'on_update' is selected.
        seed : None, int, array_like[ints], SeedSequence, BitGenerator, Generator
            Number to seed the RNG engine used for randomizing the exploration. By
            default, `None`.
        """
        super().__init__(strength, hook, seed)
        if not isinstance(epsilon, Scheduler):
            epsilon = NoScheduling[float](epsilon)
        self.epsilon_scheduler = epsilon

    @property
    def hook(self) -> Optional[str]:
        # return hook only if the strength or epislon scheduler requires to be stepped
        return (
            None
            if isinstance(self.strength_scheduler, NoScheduling)
            and isinstance(self.epsilon_scheduler, NoScheduling)
            else self._hook
        )

    @property
    def epsilon(self) -> float:
        """Gets the current probability of the exploration strategy."""
        return self.epsilon_scheduler.value

    def can_explore(self) -> bool:
        return self.np_random.random() <= self.epsilon_scheduler.value

    def step(self, *_, **__) -> None:
        """Updates the exploration probability and strength according to their
        schedulers."""
        self.strength_scheduler.step()
        self.epsilon_scheduler.step()

    def __repr__(self) -> str:
        hook = self.hook
        hookstr = "None" if hook is None else f"'{hook}'"
        return (
            f"{self.__class__.__name__}(eps={self.epsilon_scheduler.value},"
            f"stn={self.strength_scheduler.value},hook={hookstr})"
        )


class StepWiseExploration(ExplorationStrategy):
    """Wrapper exploration class that enables a base exploration strategy to change only
    every N steps, thus yielding a step-wise strategy with steps of the given length.
    This is useful when, e.g., the exploration strategy must be kept constant across
    time for a number of steps.

    Note
    ----
    This exploration wrapper modifies the exploration chance and magnitude of the
    wrapped base strategy as well as the step behaviour, i.e., the decay of the base
    exploration's schedulers (if any) is enlarged by the step size factor. This is
    because the number of calls to the base exploration's `step` method is reduced by
    a factor of the step size.
    """

    def __init__(
        self,
        base_exploration: ExplorationStrategy,
        step_size: int,
        stepwise_decay: bool = True,
    ) -> None:
        """Creates a step-wise exploration strategy wrapepr.

        Parameters
        ----------
        base_exploration : ExplorationStrategy
            The base exploration strategy to be made step-wise.
        step_size : int
            Size of each step.
        stepwise_decay : bool, optional
            Enables the decay `step` to also be step-wise, i.e., applied only every N
            steps.
        """
        super().__init__()
        self.base_exploration = base_exploration
        self.step_size = step_size
        self._explore_counter = 0
        self._step_counter = 0
        self._stepwise_decay = stepwise_decay

    @property
    def hook(self) -> Optional[str]:
        """Returns the hook of the base exploration strategy, if any."""
        return getattr(self.base_exploration, "hook", None)

    def can_explore(self) -> bool:
        # since this method is called at every timestep (when deterministic=False), we
        # decide here if the base exploration is frozen or not, i.e., if we are at the
        # new step or not
        self._explore_counter %= self.step_size
        if self._explore_counter == 0:
            self._cached_can_explore = self._cached_perturbation = None
        self._explore_counter += 1

        if self._cached_can_explore is not None:
            return self._cached_can_explore
        self._cached_can_explore = self.base_exploration.can_explore()
        return self._cached_can_explore

    def step(self, *_, **__) -> None:
        if not self._stepwise_decay:
            return self.base_exploration.step()
        self._step_counter %= self.step_size
        if self._step_counter == 0:
            self.base_exploration.step()
        self._step_counter += 1

    def perturbation(self, *args: Any, **kwargs: Any) -> npt.NDArray[np.floating]:
        if self._cached_perturbation is not None:
            return self._cached_perturbation
        self._cached_perturbation = self.base_exploration.perturbation(*args, **kwargs)
        return self._cached_perturbation

    def __repr__(self) -> str:
        clsn = self.__class__.__name__
        bclsn = self.base_exploration.__class__.__name__
        return f"{clsn}(base={bclsn},step_size={self.step_size})"
