from abc import ABC, abstractmethod
from functools import wraps
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Optional,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt
from csnlp.wrappers import Mpc

from mpcrl.agents.agent import ActType, Agent, ObsType, SymType, _update_dicts
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.util.types import GymEnvLike

ExpType = TypeVar("ExpType")

_STEP_ON_METHODS = {
    "agent-update": "on_update",
    "env-step": "on_env_step",
    "ep-start": "on_episode_start",
}
"""Mapping between `step_on` type and target method to wrap."""


class LearningAgent(Agent[SymType], ABC, Generic[SymType, ExpType]):
    """Base class for a learning agent with MPC as policy provider where the main method
    `update`, which is called to update the learnable parameters of the MPC according to
    the underlying learning methodology (e.g., Bayesian Optimization, RL, etc.) is
    abstract and must be implemented by inheriting classes.

    Note: this class makes no assumptions on the learning methodology used to update the
    MPC's learnable parameters."""

    __slots__ = (
        "_experience",
        "_learnable_pars",
        "experience_sample_size",
        "experience_sample_include_last",
    )

    def __init__(
        self,
        mpc: Mpc[SymType],
        learnable_parameters: LearnableParametersDict[SymType],
        fixed_parameters: Union[
            None, Dict[str, npt.ArrayLike], Collection[Dict[str, npt.ArrayLike]]
        ] = None,
        exploration: Optional[ExplorationStrategy] = None,
        experience: Optional[ExperienceReplay[ExpType]] = None,
        experience_sample_size: Union[int, float] = 1,
        experience_sample_include_last: Union[int, float] = 0,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        step_on: Literal["agent-update", "env-step", "ep-start"] = "agent-update",
        name: Optional[str] = None,
    ) -> None:
        """Instantiates the learning agent.

        Parameters
        ----------
        mpc : Mpc[casadi.SX or MX]
            The MPC controller used as policy provider by this agent. The instance is
            modified in place to create the approximations of the state function `V(s)`
            and action value function `Q(s,a)`, so it is recommended not to modify it
            further after initialization of the agent. Moreover, some parameter and
            constraint names will need to be created, so an error is thrown if these
            names are already in use in the mpc. These names are under the attributes
            `perturbation_parameter`, `action_parameter` and `action_constraint`.
        learnable_parameters : LearnableParametersDict
            A special dict containing the learnable parameters of the MPC, together with
            their bounds and values. This dict is complementary with `fixed_parameters`,
            which contains the MPC parameters that are not learnt by the agent.
        fixed_parameters : dict[str, array_like] or collection of, optional
            A dict (or collection of dict, in case of `csnlp.MultistartNlp`) whose keys
            are the names of the MPC parameters and the values are their corresponding
            values. Use this to specify fixed parameters, that is, non-learnable. If
            `None`, then no fixed parameter is assumed.
        exploration : ExplorationStrategy, optional
            Exploration strategy for inducing exploration in the MPC policy. By default
            `None`, in which case `NoExploration` is used in the fixed-MPC agent.
        experience : ExperienceReplay, optional
            The container for experience replay memory. If `None` is passed, then a
            memory wtih length 1 is created, i.e., it keeps only the latest memoery
            transition.
        experience_sample_size : int or float, optional
            Size (or percentage of replay `maxlen`) of the experience replay items to
            draw when performing an update. By default, one item per sampling is drawn.
        experience_sample_include_last : int or float, optional
            Size (or percentage of sample size) dedicated to including the latest
            experience transitions. By default, 0, i.e., no last item is included.
        warmstart: 'last' or 'last-successful', optional
            The warmstart strategy for the MPC's NLP. If 'last-successful', the last
            successful solution is used to warm start the solver for the next iteration.
            If 'last', the last solution is used, regardless of success or failure.
        step_on : {'agent-update', 'env-step', 'ep-start'}, optional
            Specifies to the algorithm when to step its schedulers (e.g., for learning
            rate and/or exploration decay), either after
             - each agent's update ('agent-update')
             - each environment's step ('env-step')
             - each episode's start ('ep-start').
            By default, 'agent-update' is selected.
        name : str, optional
            Name of the agent. If `None`, one is automatically created from a counter of
            the class' instancies.
        """
        super().__init__(mpc, fixed_parameters, exploration, warmstart, name)
        self._learnable_pars = learnable_parameters
        self._experience = (
            ExperienceReplay(maxlen=1) if experience is None else experience
        )
        self.experience_sample_size = experience_sample_size
        self.experience_sample_include_last = experience_sample_include_last
        self._decorate_method_with_step(step_on)

    @property
    def experience(self) -> ExperienceReplay[ExpType]:
        """Gets the experience replay memory of the agent."""
        return self._experience

    @property
    def learnable_parameters(self) -> LearnableParametersDict[SymType]:
        """Gets the parameters of the MPC that can be learnt by the agent."""
        return self._learnable_pars

    def step(self) -> None:
        """Steps the exploration strength/chance for the agent (usually, this decays
        over time)."""
        self._exploration.step()

    def store_experience(self, item: ExpType) -> None:
        """Stores the given item in the agent's memory for later experience replay.

        Parameters
        ----------
        item : experience-type
            Item to be stored in memory.
        """
        self._experience.append(item)

    def sample_experience(self) -> Iterator[ExpType]:
        """Samples the experience memory.

        Yields
        ------
        sample : iterator of experience-type
            An iterator over the sampled experience.
        """
        return self.experience.sample(
            self.experience_sample_size, self.experience_sample_include_last
        )

    def on_training_start(self, env: GymEnvLike[ObsType, ActType]) -> None:
        """Callback called at the beginning of the training process.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent is being trained on.
        """

    def on_training_end(self, env: GymEnvLike[ObsType, ActType]) -> None:
        """Callback called at the end of the training process.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent has been trained on.
        """

    def on_episode_start(self, env: GymEnvLike[ObsType, ActType], episode: int) -> None:
        """Callback called at the beginning of each episode in the training process.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent is being trained on.
        episode : int
            Number of the training episode.
        """

    def on_episode_end(self, env: GymEnvLike[ObsType, ActType], episode: int) -> None:
        """Callback called at the end of each episode in the training process.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent is being trained on.
        episode : int
            Number of the training episode.
        """

    def on_env_step(
        self, env: GymEnvLike[ObsType, ActType], episode: int, timestep: int
    ) -> None:
        """Callback called after each `env.step`.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent is being trained on.
        episode : int
            Number of the training episode.
        timestep : int
            Time instant of the current training episode.
        """

    def on_update(self) -> None:
        """Callaback called after each `agent.update`. Use this callback for, e.g.,
        decaying exploration probabilities or learning rates."""

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

    @staticmethod
    @abstractmethod
    def train(
        agent: "LearningAgent[SymType, ExpType]",
        env: GymEnvLike[ObsType, ActType],
        episodes: int,
        update_frequency: int,
        seed: Union[None, int, Iterable[int]] = None,
        raises: bool = True,
        env_reset_options: Optional[Dict[str, Any]] = None,
    ) -> npt.NDArray[np.double]:
        """Train the agent on an environment.

        Parameters
        ----------
        agent : LearningAgent or inheriting
            The agent to train.
        env : GymEnvLike[ObsType, ActType]
            A gym-like environment where to train the agent in.
        episodes : int
            Number of training episodes.
        update_frequency : int
            The frequency of timesteps (i.e., every `env.step`) at which to perform
            updates to the learning parameters.
        seed : int or iterable of ints, optional
            Each env's seed for RNG.
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

    def _decorate_method_with_step(
        self, step_on: Literal["agent-update", "env-step", "ep-start"]
    ) -> None:
        """Internal decorator to call `step` each time the selected method is called."""
        methodname = _STEP_ON_METHODS.get(step_on)
        if methodname is None:
            raise ValueError(f"Unrecognized step strategy `step_on`; got {step_on}.")

        def get_decorator(method: Callable) -> Callable:
            @wraps(method)
            def wrapper(*args, **kwargs):
                out = method(*args, **kwargs)
                self.step()
                return out

            return wrapper

        setattr(self, methodname, get_decorator(getattr(self, methodname)))

    def _get_parameters(
        self,
    ) -> Union[None, Dict[str, npt.ArrayLike], Collection[Dict[str, npt.ArrayLike]]]:
        """Internal utility to retrieve parameters of the MPC in order to solve it.
        `LearningAgent` returns both fixed and learnable parameters."""
        learnable_pars = self._learnable_pars.value_as_dict
        fixed_pars = self._fixed_pars
        if fixed_pars is None:
            return learnable_pars  # type: ignore
        if isinstance(fixed_pars, dict):
            fixed_pars.update(learnable_pars)
            return fixed_pars
        return tuple(_update_dicts(self._fixed_pars, learnable_pars))  # type: ignore
