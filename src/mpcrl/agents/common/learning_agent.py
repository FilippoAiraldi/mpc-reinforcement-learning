from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from typing import Any, Callable, Generic, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt
from gymnasium import Env

from ...core.callbacks import LearningAgentCallbackMixin
from ...core.experience import ExperienceReplay
from ...core.exploration import ExplorationStrategy
from ...core.parameters import LearnableParametersDict
from ...core.update import UpdateStrategy
from ...util.seeding import RngType, mk_seed
from .agent import ActType, Agent, ObsType, SymType, _update_dicts

ExpType = TypeVar("ExpType")


class LearningAgent(
    Agent[SymType], LearningAgentCallbackMixin, ABC, Generic[SymType, ExpType]
):
    """Base class for a learning agent with MPC as policy provider where the main method
    `update`, which is called to update the learnable parameters of the MPC according to
    the underlying learning methodology (e.g., Bayesian Optimization, RL, etc.) is
    abstract and must be implemented by inheriting classes.

    Aside of `update`, this class implements also the basic structure for both on-policy
    and off-policy learning, which require further ad-hoc implementations. For on-policy
    learning, the `train` method should be run, which requires the implementation of
    `train_one_episode`. For off-policy learning, run `train_offpolicy` and implement
    `train_one_rollout` instead. At least one of the two implementations is required in
    the inheriting class, depending on the learning algorithm. Some algorithms might
    support both.

    Note: this class makes no assumptions on the learning methodology used to update the
    MPC's learnable parameters."""

    def __init__(
        self,
        update_strategy: Union[int, UpdateStrategy],
        learnable_parameters: LearnableParametersDict[SymType],
        exploration: Optional[ExplorationStrategy] = None,
        experience: Union[None, int, ExperienceReplay[ExpType]] = None,
        **kwargs: Any,
    ) -> None:
        """Instantiates the learning agent.

        Parameters
        ----------
        update_strategy : UpdateStrategy or int
            The strategy used to decide which frequency to update the mpc parameters
            with. If an `int` is passed, then the default strategy that updates every
            `n` env's steps is used (where `n` is the argument passed); otherwise, an
            instance of `UpdateStrategy` can be passed to specify these in more details.
        learnable_parameters : LearnableParametersDict
            A special dict containing the learnable parameters of the MPC, together with
            their bounds and values. This dict is complementary with `fixed_parameters`,
            which contains the MPC parameters that are not learnt by the agent.
        exploration : ExplorationStrategy, optional
            Exploration strategy for inducing exploration in the MPC policy. By default
            `None`, in which case `NoExploration` is used in the fixed-MPC agent.
        experience : int or ExperienceReplay, optional
            The container for experience replay memory. If `None` is passed, then a
            memory with length 1 is created, i.e., it keeps only the latest memory
            transition. If an integer `n` is passed, then a memory with the length `n`
            is created and with sample size `n`.
        kwargs
            Additional arguments to be passed to `Agent`.
        """
        Agent.__init__(self, **kwargs)
        LearningAgentCallbackMixin.__init__(self)
        self._raises: bool = True
        self._learnable_pars = learnable_parameters
        if experience is None:
            experience = ExperienceReplay(maxlen=1)
        elif isinstance(experience, int):
            experience = ExperienceReplay(maxlen=experience, sample_size=experience)
        self._experience = experience
        if exploration is not None:
            self._exploration = exploration
        if not isinstance(update_strategy, UpdateStrategy):
            update_strategy = UpdateStrategy(update_strategy, "on_timestep_end")
        self._update_strategy = update_strategy
        self._updates_enabled = True
        self.establish_callback_hooks()

    @property
    def experience(self) -> ExperienceReplay[ExpType]:
        """Gets the experience replay memory of the agent."""
        return self._experience

    @property
    def update_strategy(self) -> UpdateStrategy:
        """Gets the update strategy of the agent."""
        return self._update_strategy

    @property
    def learnable_parameters(self) -> LearnableParametersDict[SymType]:
        """Gets the parameters of the MPC that can be learnt by the agent."""
        return self._learnable_pars

    def reset(self, seed: RngType = None) -> None:
        """Resets agent's internal variables, exploration and experience's RNG"""
        super().reset(seed)
        self.experience.reset(seed)

    def store_experience(self, item: ExpType) -> None:
        """Stores the given item in the agent's memory for later experience replay.

        Parameters
        ----------
        item : experience-type
            Item to be stored in memory.
        """
        self._experience.append(item)

    def evaluate(self, *args: Any, **kwargs: Any) -> npt.NDArray[np.floating]:
        self._updates_enabled = False
        return super().evaluate(*args, **kwargs)

    def train(
        self,
        env: Env[ObsType, ActType],
        episodes: int,
        seed: RngType = None,
        raises: bool = True,
        env_reset_options: Optional[dict[str, Any]] = None,
    ) -> npt.NDArray[np.floating]:
        """On-policy training of the agent on an environment.

        Parameters
        ----------
        env : Env[ObsType, ActType]
            A gym environment where to train the agent in.
        episodes : int
            Number of training episodes.
        seed : None, int, array_like[ints], SeedSequence, BitGenerator, Generator
            Agent's and each env's RNG seed.
        raises : bool, optional
            If `True`, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised.
        env_reset_options : dict, optional
            Additional information to specify how the environment is reset at each
            training episode (optional, depending on the specific environment).

        Returns
        -------
        array of doubles
            The cumulative returns for each training episode.

        Raises
        ------
        MpcSolverError or MpcSolverWarning
            Raises the error or the warning (depending on `raises`) if any of the MPC
            solvers fail.
        UpdateError or UpdateWarning
            Raises the error or the warning (depending on `raises`) if the update fails.
        """
        rng = np.random.default_rng(seed)
        self.reset(rng)
        self._updates_enabled = True
        self._raises = raises
        returns = np.zeros(episodes, float)

        self.on_training_start(env)
        for episode in range(episodes):
            state, _ = env.reset(seed=mk_seed(rng), options=env_reset_options)
            self.on_episode_start(env, episode, state)
            r = self.train_one_episode(env, episode, state, raises)
            self.on_episode_end(env, episode, r)
            returns[episode] = r

        self.on_training_end(env, returns)
        return returns

    def train_one_episode(
        self,
        env: Env[ObsType, ActType],
        episode: int,
        init_state: ObsType,
        raises: bool = True,
    ) -> float:
        """On-policy training of the agent on an environment for one episode.

        Parameters
        ----------
        env : Env[ObsType, ActType]
            A gym environment where to train the agent in.
        episode : int
            Number of the current training episode.
        init_state : observation type
            Initial state/observation of the environment.
        raises : bool, optional
            If `True`, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised.

        Returns
        -------
        float
            The cumulative rewards for this training episode.

        Raises
        ------
        MpcSolverError or MpcSolverWarning
            Raises the error or the warning (depending on `raises`) if any of the MPC
            solvers fail.
        UpdateError or UpdateWarning
            Raises the error or the warning (depending on `raises`) if the update fails.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `train_one_episode` for "
            "on-policy learning."
        )

    def train_offpolicy(
        self,
        episode_rollouts: Iterable[Iterable[Any]],
        seed: RngType = None,
        raises: bool = True,
        eval_frequency: Optional[int] = None,
        eval_env_factory: Callable[[], Optional[Env[ObsType, ActType]]] = None,
        eval_kwargs: Optional[dict[str, Any]] = None,
        eval_at_start: bool = True,
    ) -> Optional[npt.NDArray[np.floating]]:
        """Off-policy training of the agent on an environment.

        Parameters
        ----------
        episode_rollouts : iterable of iterables of any
            An iterable, i.e., sequence, of episodical rollouts generated off-policy.
            Each rollout is itself a sequence of transitions, e.g., SARSA tuples. In
            general, these can be of different types, depending on the specific learning
            algorithm.
        seed : None, int, array_like[ints], SeedSequence, BitGenerator, Generator
            Agent's (and each evaluation env's, if applicable) RNG seed.
        raises : bool, optional
            If `True`, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised.
        eval_frequency : int, optional
            Frequency of evaluation of the agent's performance during off-policy
            training. If `None`, no evaluation is performed. If an integer is passed,
            then also `eval_env_factory` must be given as argument.
        eval_env_factory : callable, optional
            A callable that returns a new environment instance to be used for
            evaluation, if `eval_frequency` is given. Otherwise, it is unused.
        eval_kwargs : dict, optional
            Additional keyword arguments to be passed to the agent's `evaluate` method.
            Must contain the `episode` key (which is a required argument to `evaluate`),
            and must not contain the `env`, `seed` or `raises` key, as these are already
            handled by this method automatically.
        eval_at_start : bool, optional
            If `True`, an evaluation round is also performed before the training starts.

        Returns
        -------
        2d-array of doubles or None
            The cumulative returns for each evaluation episode, if evaluation is
            enabled; otherwise, `None`.

        Raises
        ------
        ValueError
            Raises if `eval_frequency` is given but `eval_env_factory` is not callable.
        MpcSolverError or MpcSolverWarning
            Raises the error or the warning (depending on `raises`) if any of the MPC
            solvers fail.
        UpdateError or UpdateWarning
            Raises the error or the warning (depending on `raises`) if the update fails.
        """
        if eval_frequency is not None:
            if not callable(eval_env_factory):
                raise ValueError(
                    "If `eval_frequency` is given, `eval_env_factory` must be callable."
                )
            returns = []
            if eval_kwargs is None:
                eval_kwargs = {}
            do_eval = lambda: self.evaluate(
                eval_env_factory(), seed=rng, raises=raises, **eval_kwargs
            )
        else:
            returns = np.empty(())

        rng = np.random.default_rng(seed)
        self.reset(rng)
        self._raises = raises
        env_proxy = "off-policy"

        if eval_frequency is not None and eval_at_start:
            returns.append(do_eval())

        self._updates_enabled = True
        self.on_training_start(env_proxy)
        for episode, rollout in enumerate(episode_rollouts):
            self.on_episode_start(env_proxy, episode, float("nan"))
            self.train_one_rollout(rollout, episode, raises)
            self.on_episode_end(env_proxy, episode, float("nan"))
            if eval_frequency is not None and (episode + 1) % eval_frequency == 0:
                returns.append(do_eval())
                self._updates_enabled = True  # updates must be re-enabled after eval

        returns = np.asarray(returns, float)
        self.on_training_end(env_proxy, returns)
        if eval_frequency is not None:
            return returns

    def train_one_rollout(
        self, rollout: Iterable[Any], episode: int, raises: bool = True
    ) -> None:
        """Train the agent in an off-policy manner on the given rollout.

        Parameters
        ----------
        rollout : iterable of any
            Rollout, i.e., a sequence of transitions generated off-policy, e.g., SARSA
            tuples. In general, these can be of different types, depending on the
            specific learning algorithm.
        raises : bool, optional
            If `True`, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised.

        Raises
        ------
        MpcSolverError or MpcSolverWarning
            Raises the error or the warning (depending on `raises`) if any of the MPC
            solvers fail.
        UpdateError or UpdateWarning
            Raises the error or the warning (depending on `raises`) if the update fails.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `train_offpolicy` for "
            "off-policy learning."
        )

    @abstractmethod
    def update(self) -> Optional[str]:
        """Updates the learnable parameters of the MPC according to the agent's learning
        algorithm.

        Returns
        -------
        errormsg : str or None
            In case the update fails, an error message is returned to be raised as error
            or warning; otherwise, `None` is returned.
        """

    def establish_callback_hooks(self) -> None:
        super().establish_callback_hooks()
        # hook exploration (only if necessary)
        exploration_hook: Optional[str] = getattr(self._exploration, "hook", None)
        if exploration_hook is not None:
            self.hook_callback(
                repr(self._exploration), exploration_hook, self._exploration.step
            )
        # hook updates (always necessary)
        assert self._update_strategy.hook in {
            "on_episode_end",
            "on_timestep_end",
        }, "Updates can be hooked only to `episode_end` or `on_timestep_end`."
        func: Callable[..., None] = (
            (lambda _, e, __: self._check_and_perform_update(e, None))
            if self._update_strategy.hook == "on_episode_end"
            else (lambda _, e, t: self._check_and_perform_update(e, t))
        )
        self.hook_callback(
            repr(self._update_strategy),
            self._update_strategy.hook,
            func,
        )

    def _check_and_perform_update(self, episode: int, timestep: Optional[int]) -> None:
        """Internal utility to check if an update is due and perform it."""
        if not self._updates_enabled or not self._update_strategy.can_update():
            return
        update_msg = self.update()
        if update_msg is not None:
            self.on_update_failure(episode, timestep, update_msg, self._raises)
        self.on_update()

    def _get_parameters(
        self,
    ) -> Union[None, dict[str, npt.ArrayLike], Collection[dict[str, npt.ArrayLike]]]:
        """Internal utility to retrieve parameters of the MPC in order to solve it.
        `LearningAgent` returns both fixed and learnable parameters."""
        learnable_pars = self._learnable_pars.value_as_dict
        fixed_pars = self._fixed_pars
        if fixed_pars is None:
            return learnable_pars  # type: ignore[return-value]
        if isinstance(fixed_pars, dict):
            fixed_pars.update(learnable_pars)
            return fixed_pars
        return tuple(
            _update_dicts(self._fixed_pars, learnable_pars)  # type: ignore[arg-type]
        )
