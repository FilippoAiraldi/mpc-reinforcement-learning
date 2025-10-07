from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from typing import Any, Callable, Generic, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt
from gymnasium import Env
from gymnasium.spaces import Box

from ...core.callbacks import LearningAgentCallbackMixin
from ...core.experience import ExperienceReplay
from ...core.parameters import LearnableParametersDict
from ...core.update import UpdateStrategy
from ...util.seeding import RngType, mk_seed
from .agent import ActType, Agent, ObsType, SymType, _update_dicts

ExpType = TypeVar("ExpType")


class LearningAgent(
    Agent[SymType], LearningAgentCallbackMixin, ABC, Generic[SymType, ExpType]
):
    r"""Base abstract class for a learning agent with MPC as policy provider. The main
    method :meth:`update`, which is called to update the learnable parameters of the MPC
    according to the underlying learning methodology (e.g., Bayesian Optimization, RL,
    etc.), is abstract and must be implemented by inheriting classes based on their
    learning algorithm.

    Aside of :meth:`update`, this class also provides also the basic skeleton for both
    on-policy and off-policy learning, which require further ad-hoc implementations. For
    on-policy learning, the :meth:`train` method should be run, which requires the
    implementation of :meth:`train_one_episode`. For off-policy learning, run
    :meth:`train_offpolicy` and implement :meth:`train_one_rollout` instead. At least
    one of the two implementations is required in the inheriting class, depending on the
    learning algorithm. Some algorithms might support both.

    Parameters
    ----------
    update_strategy : UpdateStrategy or int
        The strategy used to decide which frequency to update the mpc parameters with.
        If an ``int`` is passed, then the default strategy that updates every ``n``
        env's steps is used (where ``n`` is the argument passed); otherwise, an instance
        of :class:`core.update.UpdateStrategy` can be passed to specify the
        desired strategy in more details.
    learnable_parameters : :class:`core.parameters.LearnableParametersDict`
        A special dict containing the learnable parameters of the MPC (usually referred
        to as :math:`\theta`), together with their bounds and values. This dict is
        complementary to :attr:`fixed_parameters`, which contains the MPC parameters
        that are not learnt by the agent.
    experience : int or ExperienceReplay, optional
        The container for experience replay memory. If ``None`` is passed, then a memory
        with unitary length is created, i.e., it keeps only the latest memory
        transition. If an integer ``n`` is passed, then a memory with the length ``n``
        is created and with sample size ``n``. Otherwise, pass an instance of
        :class:`core.experience.ExperienceReplay` to specify the requirements in more
        details.
    kwargs
        Additional arguments to be passed to :class:`Agent`.

    Notes
    -----
    This class makes no assumptions on the learning methodology used to update the MPC's
    learnable parameters. This could be either gradient-based or gradient-free, but the
    logic implemented in this class should be largely remain untouched.
    """

    def __init__(
        self,
        update_strategy: Union[int, UpdateStrategy],
        learnable_parameters: LearnableParametersDict[SymType],
        experience: Union[None, int, ExperienceReplay[ExpType]] = None,
        **kwargs: Any,
    ) -> None:
        Agent.__init__(self, **kwargs)
        LearningAgentCallbackMixin.__init__(self)
        self._learnable_pars = self._reorder_learnable_parameters(learnable_parameters)
        if experience is None:
            experience = ExperienceReplay(maxlen=1)
        elif isinstance(experience, int):
            experience = ExperienceReplay(maxlen=experience, sample_size=experience)
        self._experience = experience
        if not isinstance(update_strategy, UpdateStrategy):
            update_strategy = UpdateStrategy(update_strategy, "on_timestep_end")
        self._update_strategy = update_strategy
        self._raises: bool
        self._is_training = False
        self._establish_callback_hooks()

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
        super().reset(seed)
        self.experience.reset(seed)

    def store_experience(self, item: ExpType) -> None:
        """Stores the given item in the agent's :attr:`experience` for later usage in
        updating the parametrization.

        Parameters
        ----------
        item : ExpType
            Item to be stored in memory.
        """
        self._experience.append(item)

    def evaluate(self, *args: Any, **kwargs: Any) -> npt.NDArray[np.floating]:
        self._is_training = False
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
            The gym environment where to train the agent on.
        episodes : int
            Number of training episodes.
        seed : None, int, array_like of ints, SeedSequence, BitGenerator, Generator
            Seed for the agent's and env's random number generator. By default ``None``.
        raises : bool, optional
            If ``True``, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised.
        env_reset_options : dict, optional
            Additional information to specify how the environment is reset at each
            evalution episode (optional, depending on the specific environment).

        Returns
        -------
        array of doubles
            The cumulative returns for each training episode.

        Raises
        ------
        MpcSolverError or MpcSolverWarning
            Raises the error or the warning (depending on ``raises``) if any of the MPC
            solvers fail.
        UpdateError or UpdateWarning
            Raises the error or the warning (depending on ``raises``) if the update
            fails.
        """
        if hasattr(env, "action_space"):
            assert isinstance(env.action_space, Box), "Env action space must be a Box,"
        rng = np.random.default_rng(seed)
        self.reset(rng)
        self._is_training = True
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
        """On-policy training of the agent on an environment for one single episode.

        Parameters
        ----------
        env : Env[ObsType, ActType]
            The gym environment where to train the agent on.
        episode : int
            Number of the current training episode.
        init_state : observation type
            Initial state/observation of the environment.
        raises : bool, optional
            If ``True``, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised.

        Returns
        -------
        float
            The cumulative rewards for this training episode.

        Raises
        ------
        MpcSolverError or MpcSolverWarning
            Raises the error or the warning (depending on ``raises``) if any of the MPC
            solvers fail.
        UpdateError or UpdateWarning
            Raises the error or the warning (depending on ``raises``) if the update
            fails.
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
    ) -> None:
        """Off-policy training of the agent on an environment.

        Parameters
        ----------
        episode_rollouts : iterable of iterables of any
            An iterable of episodical rollouts generated in an
            off-policy fashion. Each rollout is itself a sequence of transitions, e.g.,
            SARSA tuples. In other words, `episode_rollouts` is a sequence of sequences
            of tuples. However, in general, its nature and the tuples' can widely differ
            from learning algorithm to learning algorithm.
        seed : None, int, array_like of ints, SeedSequence, BitGenerator, Generator
            Seed for the agent's random number generator. By default ``None``.
        raises : bool, optional
            If ``True``, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised.

        Raises
        ------
        MpcSolverError or MpcSolverWarning
            Raises the error or the warning (depending on ``raises``) if any of the MPC
            solvers fail.
        UpdateError or UpdateWarning
            Raises the error or the warning (depending on ``raises``) if the update
            fails.
        """
        rng = np.random.default_rng(seed)
        self.reset(rng)
        self._raises = raises
        env_proxy = "off-policy"

        self._is_training = True
        self.on_training_start(env_proxy)
        for episode, rollout in enumerate(episode_rollouts):
            self.on_episode_start(env_proxy, episode, float("nan"))
            self.train_one_rollout(rollout, episode, raises)
            self.on_episode_end(env_proxy, episode, float("nan"))

        self.on_training_end(env_proxy, np.empty(0))

    def train_one_rollout(
        self, rollout: Iterable[Any], episode: int, raises: bool = True
    ) -> None:
        """Train the agent in an off-policy manner on the given rollout.

        Parameters
        ----------
        rollout : iterable of any
            Rollout, i.e., a sequence of transitions generated off-policy, e.g., SARSA
            tuples.  However, in general, these tuples can be of different nature,
            depending on the specific learning algorithm.
        raises : bool, optional
            If ``True``, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised.

        Raises
        ------
        MpcSolverError or MpcSolverWarning
            Raises the error or the warning (depending on ``raises``) if any of the MPC
            solvers fail.
        UpdateError or UpdateWarning
            Raises the error or the warning (depending on ``raises``) if the update
            fails.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `train_offpolicy` for "
            "off-policy learning."
        )

    @abstractmethod
    def update(self) -> Optional[str]:
        r"""Updates the learnable parameters (usually referred to as :math:`\theta`) of
        the MPC according to the agent's learning algorithm.

        Returns
        -------
        errormsg : str or None
            In case the update fails, an error message is returned to be raised as error
            or warning; otherwise, `None` is returned.
        """

    def _reorder_learnable_parameters(
        self, learnable_parameters: LearnableParametersDict[SymType]
    ) -> LearnableParametersDict[SymType]:
        """Reorders the learnable parameters of the MPC according to their creation
        order."""
        reordered = [
            learnable_parameters.pop(name)
            for name in self.V.parameters.keys()
            if name in learnable_parameters
        ]
        assert not learnable_parameters, (
            "Not all learnable parameters could be reordered. "
            "Please check for spurious learnable parameters in `learnable_parameters`."
        )
        learnable_parameters.update(reordered)
        return learnable_parameters

    def _establish_callback_hooks(self) -> None:
        super()._establish_callback_hooks()
        # hook exploration (only if necessary)
        exploration_hook = self._exploration.hook
        if exploration_hook is not None:
            self._hook_callback(
                repr(self._exploration), exploration_hook, self._exploration.step
            )
        # hook updates (always necessary)
        update_hook = self._update_strategy.hook
        func: Callable[..., None] = (
            (lambda _, e, __: self._check_and_perform_update(e, None))
            if update_hook == "on_episode_end"
            else (lambda _, e, t: self._check_and_perform_update(e, t))
        )
        self._hook_callback(repr(self._update_strategy), update_hook, func)

    def _check_and_perform_update(self, episode: int, timestep: Optional[int]) -> None:
        """Internal utility to check if an update is due and perform it."""
        if not self._is_training or not self._update_strategy.can_update():
            return
        update_msg = self.update()
        if update_msg is not None:
            self.on_update_failure(episode, timestep, update_msg, self._raises)
        self.on_update()

    def _get_parameters(
        self,
    ) -> Union[None, dict[str, npt.ArrayLike], Collection[dict[str, npt.ArrayLike]]]:
        """Internal utility to retrieve parameters of the MPC in order to solve it.
        :class:`LearningAgent` returns both fixed and learnable parameters."""
        learnable_pars = self._learnable_pars.value_as_dict
        fixed_pars = self._fixed_pars
        if fixed_pars is None:
            return learnable_pars
        if isinstance(fixed_pars, dict):
            fixed_pars.update(learnable_pars)
            return fixed_pars
        return tuple(_update_dicts(self._fixed_pars, learnable_pars))
