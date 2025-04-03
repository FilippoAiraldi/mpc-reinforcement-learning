r"""Exploration is a fundamental concept in Reinforcement Learning. Without it, often
the learning algorithms converge to very suboptimal solutions, or don't even work.

This submodule contains base classes and implementations for exploration strategies in
the context of MPC-based RL. These classes allow the agent to draw perturbations to
apply then to the MPC's optimal action, thus inducing exploration. Mathematically
speaking, this can be achieved in two distinct ways, or modes:

- **additive**: this is the simplest way to apply perturbations. When the MPC solver
  provides the optimal action to take in the environment's current state :math:`s` by
  solving the state value function problem :math:`\min_{u} V(s)`, before applying the
  action to the environment, the agent will draw a perturbation :math:`p` from the
  exploration stratey and apply the action :math:`\tilde{u} = u + p` to the environment.

- **gradient-based**: this is a more sophisticated way to apply perturbations. We can
  induce exploration more safely by modifying the objective of the state value function
  as :math:`\min_{u} V(s) + p^\top u_0`, where :math:`p` is the random perturbation and
  :math:`u_0` is the first action in the NLP problem. This way, we can perturb the
  gradient of the solution based on the scale of the first action.

See :ref:`user_guide_exploration` for a more thorough explanation. In any case,
whichever mode is selected, all the modifications and perturbations are taken care of
automatically by the agent and the exploration strategy."""

from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt

from ..util.seeding import RngType
from .schedulers import NoScheduling, Scheduler


class ExplorationStrategy(ABC):
    """Base abstract class for exploration strategies.

    Parameters
    ----------
    hook : {"on_update", "on_episode_end", "on_timestep_end"}, optional
        Specifies to which callback to hook onto, i.e., when to step the exploration's
        schedulers (if any) to, e.g., decay the chances of exploring or the perturbation
        strength (see :meth:`step` also). The options are

        - ``"on_update"``, which steps the exploration after each agent's update
        - ``"on_episode_end"``, which steps the exploration after each episode ends
        - ``"on_timestep_end"``, which steps the exploration after each env's timestep.

        By default, ``"on_update"`` is selected.
    mode : {"gradient-based", "additive"} optional
        Mode of application of explorative perturbations to the MPC. If ``"additive"``,
        then the drawn pertubation is added to the optimal action computed by the
        MPC solver. By default, ``"gradient-based"`` is selected, and in this mode the
        pertubations enter  directly in the MPC objective and is multiplied by the first
        action, thus affecting its gradient.
    """

    def __init__(
        self,
        hook: Literal["on_update", "on_episode_end", "on_timestep_end"] = "on_update",
        mode: Literal["gradient-based", "additive"] = "gradient-based",
    ) -> None:
        super().__init__()
        self._hook = hook
        self._mode = mode

    @property
    def hook(
        self,
    ) -> Optional[Literal["on_update", "on_episode_end", "on_timestep_end"]]:
        """Gets which callback the exploration is hooked on, i.e., when to step the
        exploration's schedulers (if any) to, e.g., decay the chances of exploring or
        the perturbation strength (see :meth:`step` also). Can be ``None`` in case no
        hook is needed."""
        return self._hook

    @property
    def mode(self) -> Literal["gradient-based", "additive"]:
        """Gets the mode of application of explorative perturbations to the MPC."""
        return self._mode

    @abstractmethod
    def can_explore(self) -> bool:
        """Computes whether, according to the exploration strategy, the agent should
        explore or not now, at the current instant.

        Returns
        -------
        bool
            ``True`` if the agent should explore according to this strategy; otherwise,
            ``False``.
        """

    @abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> None:
        """Steps (i.e., decays or increases) any scheduler that this class holds, e.g.,
        exploration's strength and chances."""

    @abstractmethod
    def perturbation(self, *args: Any, **kwargs: Any) -> npt.NDArray[np.floating]:
        """Returns a random perturbation."""

    def reset(self, _: RngType = None) -> None:
        """Resets the exploration status, in case it is non-deterministic."""

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hook={self.hook},mode={self.mode})"


class NoExploration(ExplorationStrategy):
    """Strategy where no exploration is allowed at any time or, in other words, the
    policy is always deterministic (only based on the current state, and not perturbed).

    Notes
    -----
    This is a special kind of :class:`ExplorationStrategy`, the only one without any
    :attr:`hook` and :attr:`mode`.
    """

    def __init__(self) -> None:
        super().__init__()
        del self._hook, self._mode

    @property
    def hook(self) -> None:
        """Returns ``None``, since no exploration is allowed."""
        return None

    @property
    def mode(self) -> None:
        """Returns ``None``, since no exploration is allowed."""
        return None

    def can_explore(self) -> bool:
        return False

    def step(self, *_, **__) -> None:
        """Does nothing, since no exploration is allowed."""
        return

    def perturbation(self, *args: Any, **kwargs: Any) -> npt.NDArray[np.floating]:
        raise NotImplementedError(
            f"Perturbation not implemented in {self.__class__.__name__}"
        )

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class GreedyExploration(ExplorationStrategy):
    """Fully greedy strategy that always perturbs randomly the MPC policy.

    Parameters
    ----------
    strength : scheduler or array/supports-algebraic-operations
        The strength of the exploration. If passed in the form of an
        :class:`mpcrl.schedulers.Scheduler`, then the strength can be scheduled to decay
        or increase every time :meth:`step` is called. Otherwise, it is kept constant.
    hook : {"on_update", "on_episode_end", "on_timestep_end"}, optional
        Specifies to which callback to hook onto, i.e., when to step the exploration's
        schedulers (if any) to, e.g., decay the chances of exploring or the perturbation
        strength (see :meth:`step` also). The options are

        - ``"on_update"``, which steps the exploration after each agent's update
        - ``"on_episode_end"``, which steps the exploration after each episode ends
        - ``"on_timestep_end"``, which steps the exploration after each env's timestep.

        By default, ``"on_update"`` is selected.
    mode : {"gradient-based", "additive"} optional
        Mode of application of explorative perturbations to the MPC. If ``"additive"``,
        then the drawn pertubation is added to the optimal action computed by the
        MPC solver. By default, ``"gradient-based"`` is selected, and in this mode the
        pertubations enter  directly in the MPC objective and is multiplied by the first
        action, thus affecting its gradient.
    seed : None, int, array_like of ints, SeedSequence, BitGenerator, Generator
        Number to seed the :class:`numpy.random.Generator` used for randomizing the
        exploration. By default, ``None``.
    """

    def __init__(
        self,
        strength: Union[Scheduler[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        hook: Literal["on_update", "on_episode_end", "on_timestep_end"] = "on_update",
        mode: Literal["gradient-based", "additive"] = "gradient-based",
        seed: RngType = None,
    ) -> None:
        super().__init__(hook, mode)
        if not isinstance(strength, Scheduler):
            strength = NoScheduling[npt.NDArray[np.floating]](strength)
        self.strength_scheduler = strength
        self.reset(seed)

    @property
    def hook(
        self,
    ) -> Optional[Literal["on_update", "on_episode_end", "on_timestep_end"]]:
        # return hook only if the strength scheduler requires to be stepped
        return None if isinstance(self.strength_scheduler, NoScheduling) else self._hook

    def reset(self, seed: RngType = None) -> None:
        self.np_random = np.random.default_rng(seed)

    def can_explore(self) -> bool:
        return True

    def step(self, *_, **__) -> None:
        """Steps (i.e., decays or increases) the exploration strength according to its
        scheduler."""
        self.strength_scheduler.step()

    def perturbation(
        self, method: str, *args: Any, **kwargs: Any
    ) -> npt.NDArray[np.floating]:
        """Returns a random perturbation.

        Parameters
        ----------
        method : str
            The name of a method from the ones available to
            :class:`numpy.random.Generator`,
            e.g., ``"random"`` for :func:`numpy.random.Generator.random`, ``"normal"``
            for :func:`numpy.random.Generator.random`, etc.
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
        stn = self.strength_scheduler.value
        return f"{self.__class__.__name__}(stn={stn},hook={self.hook},mode={self.mode})"


class EpsilonGreedyExploration(GreedyExploration):
    """Epsilon-greedy strategy for perturbing the policy, which only occasionally
    perturbs randomly the MPC policy.

    Parameters
    ----------
    epsilon : scheduler or float
        The probability to explore. Should be in range ``[0, 1]``. If passed in the form
        of an :class:`mpcrl.schedulers.Scheduler`, then the probability can be scheduled
        to decay or increase every time :meth:`step` is called. Otherwise, it is kept
        constant.
    strength : scheduler or array/supports-algebraic-operations
        The strength of the exploration. If passed in the form of an
        :class:`mpcrl.schedulers.Scheduler`, then the strength can be scheduled to decay
        or increase every time :meth:`step` is called. Otherwise, it is kept constant.
    hook : {"on_update", "on_episode_end", "on_timestep_end"}, optional
        Specifies to which callback to hook onto, i.e., when to step the exploration's
        schedulers (if any) to, e.g., decay the chances of exploring or the perturbation
        strength (see :meth:`step` also). The options are

        - ``"on_update"``, which steps the exploration after each agent's update
        - ``"on_episode_end"``, which steps the exploration after each episode ends
        - ``"on_timestep_end"``, which steps the exploration after each env's timestep.

        By default, ``"on_update"`` is selected.
    mode : {"gradient-based", "additive"} optional
        Mode of application of explorative perturbations to the MPC. If ``"additive"``,
        then the drawn pertubation is added to the optimal action computed by the
        MPC solver. By default, ``"gradient-based"`` is selected, and in this mode the
        pertubations enter  directly in the MPC objective and is multiplied by the first
        action, thus affecting its gradient.
    seed : None, int, array_like of ints, SeedSequence, BitGenerator, Generator
        Number to seed the :class:`numpy.random.Generator` used for randomizing the
        exploration. By default, ``None``.
    """

    def __init__(
        self,
        epsilon: Union[Scheduler[float], float],
        strength: Union[Scheduler[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        hook: Literal["on_update", "on_episode_end", "on_timestep_end"] = "on_update",
        mode: Literal["gradient-based", "additive"] = "gradient-based",
        seed: RngType = None,
    ) -> None:
        super().__init__(strength, hook, mode, seed)
        if not isinstance(epsilon, Scheduler):
            epsilon = NoScheduling[float](epsilon)
        self.epsilon_scheduler = epsilon

    @property
    def hook(
        self,
    ) -> Optional[Literal["on_update", "on_episode_end", "on_timestep_end"]]:
        # return hook only if the strength or epsilon scheduler requires to be stepped
        return (
            None
            if isinstance(self.strength_scheduler, NoScheduling)
            and isinstance(self.epsilon_scheduler, NoScheduling)
            else self._hook
        )

    def can_explore(self) -> bool:
        return self.np_random.random() <= self.epsilon_scheduler.value

    def step(self, *_, **__) -> None:
        """Steps (i.e., decays or increases) the exploration strength and probability
        according to their schedulers."""
        self.strength_scheduler.step()
        self.epsilon_scheduler.step()

    def __repr__(self) -> str:
        clsn = self.__class__.__name__
        eps = self.epsilon_scheduler.value
        stn = self.strength_scheduler.value
        return f"{clsn}(eps={eps},stn={stn},hook={self.hook},mode={self.mode})"


class OrnsteinUhlenbeckExploration(ExplorationStrategy):
    """Exploration based on the Ornstein-Uhlenbeck Brownian motion with friction.

    Inspired by :class:`stable_baselines3.common.noise.OrnsteinUhlenbeckActionNoise`.

    Parameters
    ----------
    mean : scheduler or array/supports-algebraic-operations
        Mean of the stochastic process. Should have the same shape as the action.
    sigma : scheduler or array/supports-algebraic-operations
        Standard deviation of the stochastic process. Should have the same shape as
        the action.
    theta : float, optional
        Coefficient of attraction of the process towards mean, by default ``0.15``.
    dt : float, optional
        Time step of the process, by default ``1.0``.
    initial_noise : array-like, optional
        A default initial noise. By default ``None``, in which case it is set to zero.
    hook : {"on_update", "on_episode_end", "on_timestep_end"}, optional
        Specifies to which callback to hook onto, i.e., when to step the exploration's
        schedulers (if any) to, e.g., decay the chances of exploring or the perturbation
        strength (see :meth:`step` also). The options are

        - ``"on_update"``, which steps the exploration after each agent's update
        - ``"on_episode_end"``, which steps the exploration after each episode ends
        - ``"on_timestep_end"``, which steps the exploration after each env's timestep.

        By default, ``"on_update"`` is selected.
    mode : {"gradient-based", "additive"} optional
        Mode of application of explorative perturbations to the MPC. If ``"additive"``,
        then the drawn pertubation is added to the optimal action computed by the
        MPC solver. By default, ``"gradient-based"`` is selected, and in this mode the
        pertubations enter  directly in the MPC objective and is multiplied by the first
        action, thus affecting its gradient.
    seed : None, int, array_like of ints, SeedSequence, BitGenerator, Generator
        Number to seed the :class:`numpy.random.Generator` used for randomizing the
        exploration. By default, ``None``.
    """

    def __init__(
        self,
        mean: Union[Scheduler[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        sigma: Union[Scheduler[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        theta: float = 0.15,
        dt: float = 1.0,
        initial_noise: Optional[npt.ArrayLike] = None,
        hook: Literal["on_update", "on_episode_end", "on_timestep_end"] = "on_update",
        mode: Literal["gradient-based", "additive"] = "gradient-based",
        seed: RngType = None,
    ) -> None:
        super().__init__(hook, mode)
        if not isinstance(mean, Scheduler):
            mean = NoScheduling[npt.NDArray[np.floating]](mean)
        self.mean_scheduler = mean
        if not isinstance(sigma, Scheduler):
            sigma = NoScheduling[npt.NDArray[np.floating]](sigma)
        self.sigma_scheduler = sigma
        self.theta = theta
        self.dt = dt
        self._dtheta_dt = theta * dt
        self._sqrt_dt = np.sqrt(dt)
        self.initial_noise = initial_noise
        self.reset(seed)

    @property
    def hook(
        self,
    ) -> Optional[Literal["on_update", "on_episode_end", "on_timestep_end"]]:
        # return hook only if the mean or sigma scheduler requires to be stepped
        return (
            None
            if isinstance(self.mean_scheduler, NoScheduling)
            and isinstance(self.sigma_scheduler, NoScheduling)
            else self._hook
        )

    def reset(self, seed: RngType = None) -> None:
        self.np_random = np.random.default_rng(seed)
        self._prev_noise = (
            np.zeros_like(self.mean_scheduler.value)
            if self.initial_noise is None
            else np.asarray(self.initial_noise)
        )

    def can_explore(self) -> bool:
        return True

    def step(self, *_, **__) -> None:
        """Updates (i.e., decays or increases) the mean and standard deviation of the
        perturbation according to their schedulers."""
        self.mean_scheduler.step()
        self.sigma_scheduler.step()

    def perturbation(
        self, *_: Any, size: Union[int, tuple[int, ...]], **__: Any
    ) -> npt.NDArray[np.floating]:
        sigma = self.sigma_scheduler.value
        noise = (
            self._prev_noise
            + self._dtheta_dt * (self.mean_scheduler.value - self._prev_noise)
            + self._sqrt_dt * (sigma * self.np_random.normal(size=size))
        )
        self._prev_noise = noise
        return noise

    def __repr__(self) -> str:
        clsn = self.__class__.__name__
        mean = self.mean_scheduler.value
        sigma = self.sigma_scheduler.value
        return (
            f"{clsn}(mean={mean},sigma={sigma},theta={self.theta},dt={self.dt},"
            f"hook={self.hook},mode={self.mode})"
        )


class StepWiseExploration(ExplorationStrategy):
    """Wrapper-like exploration that keeps the wrapped base exploration strategy
    constants for a number of steps, thus creating a piecewise exploration.

    This class takes in another exploration instance, and allows it to change only every
    ``N`` steps, thus yielding a step-wise strategy with steps of the given length. This
    is useful when, e.g., the exploration strategy must be kept constant across time for
    a number of steps.

    Parameters
    ----------
    base_exploration : ExplorationStrategy
        The base exploration strategy to be made step-wise.
    step_size : int
        Size of each step.
    stepwise_decay : bool, optional
        Enables the decay :meth:`step` to also be step-wise, i.e., applied only every
        ``N`` steps.

    Notes
    -----
    Be carefull that this exploration wrapper ends up modifying the exploration chance
    and magnitude (if any) of the wrapped base strategy as well as the step behaviour,
    i.e., the frequency of the decay/increment of the base exploration's schedulers
    (again, if any) is enlarged by the step size factor. This is because the number of
    calls to the base exploration's :meth:`step` method is reduced by a factor of the
    step size.
    """

    def __init__(
        self,
        base_exploration: ExplorationStrategy,
        step_size: int,
        stepwise_decay: bool = True,
    ) -> None:
        super().__init__()
        del self._hook, self._mode
        self.base_exploration = base_exploration
        self.step_size = step_size
        self._explore_counter = 0
        self._step_counter = 0
        self._stepwise_decay = stepwise_decay

    @property
    def hook(
        self,
    ) -> Optional[Literal["on_update", "on_episode_end", "on_timestep_end"]]:
        """Returns the hook of the base exploration strategy, if any."""
        return self.base_exploration.hook

    @property
    def mode(self) -> Literal["gradient-based", "additive"]:
        """Returns the mode of the base exploration strategy."""
        return self.base_exploration.mode

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
        h = self.hook
        m = self.mode
        return f"{clsn}(base={bclsn},step_size={self.step_size},hook={h},mode={m})"
