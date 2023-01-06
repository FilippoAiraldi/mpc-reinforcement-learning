from abc import ABC, abstractmethod
from contextlib import contextmanager
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

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp.wrappers import Mpc

from mpcrl.agents.agent import ActType, Agent, ObsType, SymType, _update_dicts
from mpcrl.core.callbacks import LearningAgentCallbacks
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.core.schedulers import Scheduler
from mpcrl.util.iters import bool_cycle
from mpcrl.util.random import generate_seeds
from mpcrl.util.types import GymEnvLike

ExpType = TypeVar("ExpType")


class LearningAgent(
    Agent[SymType], LearningAgentCallbacks, ABC, Generic[SymType, ExpType]
):
    """Base class for a learning agent with MPC as policy provider where the main method
    `update`, which is called to update the learnable parameters of the MPC according to
    the underlying learning methodology (e.g., Bayesian Optimization, RL, etc.) is
    abstract and must be implemented by inheriting classes.

    Note: this class makes no assumptions on the learning methodology used to update the
    MPC's learnable parameters."""

    __slots__ = ("_experience", "_learnable_pars")

    def __init__(
        self,
        mpc: Mpc[SymType],
        learnable_parameters: LearnableParametersDict[SymType],
        fixed_parameters: Union[
            None, Dict[str, npt.ArrayLike], Collection[Dict[str, npt.ArrayLike]]
        ] = None,
        exploration: Optional[ExplorationStrategy] = None,
        experience: Optional[ExperienceReplay[ExpType]] = None,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        stepping: Literal["on_update", "on_episode_start", "on_env_step"] = "on_update",
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
            memory with length 1 is created, i.e., it keeps only the latest memory
            transition.
        warmstart: 'last' or 'last-successful', optional
            The warmstart strategy for the MPC's NLP. If 'last-successful', the last
            successful solution is used to warm start the solver for the next iteration.
            If 'last', the last solution is used, regardless of success or failure.
        stepping : {'on_update', 'on_episode_start', 'on_env_step'}, optional
            Specifies to the algorithm when to step its schedulers (e.g., for learning
            rate and/or exploration decay), either after 1) each agent's update, if
            'on_update'; 2) each episode's start, if 'on_episode_start'; 3) each
            environment's step, if 'on_env_step'. By default, 'on_update' is selected.
        name : str, optional
            Name of the agent. If `None`, one is automatically created from a counter of
            the class' instancies.
        """
        Agent.__init__(
            self,
            mpc=mpc,
            fixed_parameters=fixed_parameters,
            exploration=exploration,
            warmstart=warmstart,
            name=name,
        )
        LearningAgentCallbacks.__init__(self)
        self._learnable_pars = learnable_parameters
        self._experience = (
            ExperienceReplay(maxlen=1) if experience is None else experience
        )
        self._decorate_method_with_step(stepping)

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
        # prepare for training start
        update_cycle = bool_cycle(update_frequency)
        returns = np.zeros(episodes, dtype=float)
        agent.on_training_start(env)

        for episode, current_seed in zip(range(episodes), generate_seeds(seed)):
            agent.on_episode_start(env, episode)
            agent.reset()
            state, _ = env.reset(seed=current_seed, options=env_reset_options)
            returns[episode] = agent.train_one_episode(
                agent=agent,
                env=env,
                episode=episode,
                init_state=state,
                update_cycle=update_cycle,
                raises=raises,
            )
            agent.on_episode_end(env, episode, returns[episode])

        agent.on_training_end(env, returns)
        return returns

    @staticmethod
    @abstractmethod
    def train_one_episode(
        agent: "LearningAgent[SymType, ExpType]",
        env: GymEnvLike[ObsType, ActType],
        episode: int,
        init_state: ObsType,
        update_cycle: Iterator[bool],
        raises: bool = True,
    ) -> float:
        """Train the agent on an environment for one episode.

        Parameters
        ----------
        agent : LearningAgent or inheriting
            The agent to train.
        env : GymEnvLike[ObsType, ActType]
            A gym-like environment where to train the agent in.
        episode : int
            Number of the current training episode.
        init_state : observation type
            Initial state/observation of the environment.
        update_cycle : itertools.cycle of bool
            Update cycle. When this iterator returns true, then an update should be
            performed. Should be an infinite iterator to avoid exceptions.
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

    @contextmanager
    def fullstate(self) -> Iterator[None]:
        with super().fullstate(), self._learnable_pars.fullstate():
            yield

    @contextmanager
    def pickleable(self) -> Iterator[None]:
        with super().pickleable(), self._learnable_pars.pickleable():
            yield

    def _decorate_method_with_step(self, methodname: str) -> None:
        """Internal decorator to call `step` each time the selected method is called."""

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
            return learnable_pars  # type: ignore[return-value]
        if isinstance(fixed_pars, dict):
            fixed_pars.update(learnable_pars)
            return fixed_pars
        return tuple(
            _update_dicts(self._fixed_pars, learnable_pars)  # type: ignore[arg-type]
        )


class RlLearningAgent(LearningAgent[SymType, ExpType], ABC):
    """Base class for learning agents that employe gradient-based RL strategies to
    learn/improve the MPC policy."""

    __slots__ = ("_learning_rate_scheduler", "discount_factor", "_update_solver")

    def __init__(
        self,
        mpc: Mpc[SymType],
        discount_factor: float,
        learning_rate: Union[
            Scheduler[npt.NDArray[np.double]],
            Scheduler[float],
            npt.NDArray[np.double],
            float,
        ],
        learnable_parameters: LearnableParametersDict[SymType],
        fixed_parameters: Union[
            None, Dict[str, npt.ArrayLike], Collection[Dict[str, npt.ArrayLike]]
        ] = None,
        exploration: Optional[ExplorationStrategy] = None,
        experience: Optional[ExperienceReplay[ExpType]] = None,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        stepping: Literal["on_update", "on_episode_start", "on_env_step"] = "on_update",
        name: Optional[str] = None,
    ) -> None:
        """Instantiates the RL learning agent.

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
        discount_factor : float
            In RL, the factor that discounts future rewards in favor of immediate
            rewards. Usually denoted as `\\gamma`. Should be a number in (0, 1).
        learning_rate : Scheduler of array or float
            The learning rate of the algorithm, in general, a small number. A scheduler
            can be passed so that the learning rate is decayed after every step (see
            `stepping`). The rate can be a single float, or an array of rates for each
            parameter.
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
            memory with length 1 is created, i.e., it keeps only the latest memory
            transition.
        warmstart: 'last' or 'last-successful', optional
            The warmstart strategy for the MPC's NLP. If 'last-successful', the last
            successful solution is used to warm start the solver for the next iteration.
            If 'last', the last solution is used, regardless of success or failure.
        stepping : {'on_update', 'on_episode_start', 'on_env_step'}, optional
            Specifies to the algorithm when to step its schedulers (e.g., for learning
            rate and/or exploration decay), either after 1) each agent's update, if
            'on_update'; 2) each episode's start, if 'on_episode_start'; 3) each
            environment's step, if 'on_env_step'. By default, 'on_update' is selected.
        name : str, optional
            Name of the agent. If `None`, one is automatically created from a counter of
            the class' instancies.
        """
        super().__init__(
            mpc=mpc,
            learnable_parameters=learnable_parameters,
            fixed_parameters=fixed_parameters,
            exploration=exploration,
            experience=experience,
            warmstart=warmstart,
            stepping=stepping,
            name=name,
        )
        self._learning_rate_scheduler = (
            learning_rate
            if isinstance(learning_rate, Scheduler)
            else Scheduler(learning_rate)
        )
        self.discount_factor = discount_factor
        self._update_solver = self._init_update_solver()

    @property
    def learning_rate(self) -> Union[float, npt.NDArray[np.double]]:
        """Gets the learning rate of the Q-learning agent."""
        return self._learning_rate_scheduler.value

    def _init_update_solver(self) -> Optional[cs.Function]:
        """Internal utility to initialize the update solver, in particular, a QP solver.
        If the update is unconstrained, then no solver is initialized, i.e., `None` is
        returned."""
        lb = self._learnable_pars.lb
        ub = self._learnable_pars.ub
        if np.isneginf(lb).all() and np.isposinf(ub).all():
            return None

        n_p = self._learnable_pars.size
        theta: cs.SX = cs.SX.sym("theta", n_p, 1)
        theta_new: cs.SX = cs.SX.sym("theta+", n_p, 1)
        dtheta = theta_new - theta
        p: cs.SX = cs.SX.sym("p", n_p, 1)
        qp = {
            "x": theta_new,
            "f": 0.5 * cs.dot(dtheta, dtheta) + cs.dot(p, dtheta),
            "p": cs.vertcat(theta, p),
        }
        opts = {"print_iter": False, "print_header": False}
        return cs.qpsol(f"qpsol_{self.name}", "qrqp", qp, opts)


# TODO:
# - max update percentage
# - updater on steps or on episodes
