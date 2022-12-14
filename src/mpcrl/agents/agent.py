from contextlib import contextmanager
from typing import (
    Any,
    Collection,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from warnings import warn

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Solution, wrappers
from csnlp.util.io import SupportsDeepcopyAndPickle
from csnlp.wrappers import Mpc
from typing_extensions import TypeAlias

from mpcrl.core.errors import MpcSolverError, MpcSolverWarning
from mpcrl.core.exploration import ExplorationStrategy, NoExploration
from mpcrl.core.random import make_seeds
from mpcrl.util.named import Named
from mpcrl.util.types import GymEnvLike

SymType = TypeVar("SymType", cs.SX, cs.MX)
ActType: TypeAlias = Union[npt.ArrayLike, Dict[str, npt.ArrayLike]]
ObsType: TypeAlias = Union[npt.ArrayLike, Dict[str, npt.ArrayLike]]


def _update_dicts(sinks: Iterable[Dict], source: Dict) -> Iterator[Dict]:
    """Internal utility for updating dicts `sinks` with one `source`."""
    for sink in sinks:
        sink.update(source)
        yield sink


def _get_action(mpc: Mpc, sol: Solution) -> cs.DM:
    """Internal utility to get the first optimal MPC action from a solution."""
    return cs.vertcat(*(sol.vals[u][:, 0] for u in mpc.actions))


def _raise_or_warn_mpc_failure(msg: str, raises: bool) -> None:
    """Internal utility to raise errors or warnings with a message for MPC failures."""
    if raises:
        raise MpcSolverError(msg)
    else:
        warn(msg, MpcSolverWarning)


class Agent(Named, SupportsDeepcopyAndPickle, Generic[SymType]):
    """Simple MPC-based agent with a fixed (i.e., non-learnable) MPC controller.

    In this agent, the MPC controller is used as policy provider, as well as to provide
    the value function `V(s)` and quality function `Q(s,a)`, where `s` and `a` are the
    state and action of the environment, respectively. However, this class does not use
    any RL method to improve its MPC policy."""

    __slots__ = (
        "_V",
        "_Q",
        "_fixed_pars",
        "_exploration",
        "_last_solution",
        "_store_last_successful",
    )
    cost_perturbation_method = "normal"
    cost_perturbation_parameter = "cost_perturbation"
    init_action_parameter = init_action_constraint = "a_init"

    def __init__(
        self,
        mpc: Mpc[SymType],
        fixed_parameters: Union[
            None, Dict[str, npt.ArrayLike], Collection[Dict[str, npt.ArrayLike]]
        ] = None,
        exploration: Optional[ExplorationStrategy] = None,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        name: Optional[str] = None,
    ) -> None:
        """Instantiates an agent with an MPC controller.

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
        fixed_parameters : dict[str, array_like] or collection of, optional
            A dict (or collection of dict, in case of `csnlp.MultistartNlp`) whose keys
            are the names of the MPC parameters and the values are their corresponding
            values. Use this to specify fixed parameters, that is, non-learnable. If
            `None`, then no fixed parameter is assumed.
        exploration : ExplorationStrategy, optional
            Exploration strategy for inducing exploration in the MPC policy. By default
            `None`, in which case `NoExploration` is used in the fixed-MPC agent.
        warmstart: 'last' or 'last-successful', optional
            The warmstart strategy for the MPC's NLP. If 'last-successful', the last
            successful solution is used to warm start the solver for the next iteration.
            If 'last', the last solution is used, regardless of success or failure.
        name : str, optional
            Name of the agent. If `None`, one is automatically created from a counter of
            the class' instancies.

        Raises
        ------
        ValueError
            Raises if the given mpc has no control action as optimization variable; or
            if the required parameter and constraint names are already specified in the
            mpc.
        """
        Named.__init__(self, name)
        SupportsDeepcopyAndPickle.__init__(self)
        self._V, self._Q = self._setup_V_and_Q(mpc)
        self._fixed_pars = fixed_parameters
        self._exploration = NoExploration() if exploration is None else exploration
        self._store_last_successful = warmstart == "last-successful"
        self.reset()

    @property
    def unwrapped(self) -> "Agent":
        """Gets the underlying wrapped instance of an agent."""
        return self

    def is_wrapped(self, *args: Any, **kwargs: Any) -> bool:
        """Gets whether the agent instance is wrapped or not by the wrapper type."""
        return False

    @property
    def V(self) -> Mpc[SymType]:
        """Gets the MPC function approximation of the state value function `V(s)`."""
        return self._V

    @property
    def Q(self) -> Mpc[SymType]:
        """Gets the MPC function approximation of the action value function `Q(s,a)`."""
        return self._Q

    @property
    def fixed_parameters(
        self,
    ) -> Union[None, Dict[str, npt.ArrayLike], Collection[Dict[str, npt.ArrayLike]]]:
        """Gets the fixed parameters of the MPC controller (i.e., non-learnable). Can be
        an iterable in the case of `csnlp.MultistartNpl`, where multiple parameters can
        be specified, one for each scenario in the MPC scheme."""
        return self._fixed_pars

    @property
    def exploration(self) -> ExplorationStrategy:
        """Gets the exploration strategy used within this agent."""
        return self._exploration

    @contextmanager
    def fullstate(self) -> Iterator[None]:
        with super().fullstate(), self._Q.fullstate(), self._V.fullstate():
            yield

    @contextmanager
    def pickleable(self) -> Iterator[None]:
        with super().pickleable(), self._Q.pickleable(), self._V.pickleable():
            yield

    def reset(self) -> None:
        """Resets the agent's internal variables."""
        self._last_solution: Optional[Solution[SymType]] = None

    def solve_mpc(
        self,
        mpc: Mpc[SymType],
        state: Union[npt.ArrayLike, Dict[str, npt.ArrayLike]],
        action: Union[None, npt.ArrayLike, Dict[str, npt.ArrayLike]] = None,
        perturbation: Optional[npt.ArrayLike] = None,
        vals0: Union[
            None, Dict[str, npt.ArrayLike], Iterable[Dict[str, npt.ArrayLike]]
        ] = None,
    ) -> Solution:
        """Solves the agent's specific MPC optimal control problem.

        Parameters
        ----------
        mpc : Mpc
            The MPC problem to solve, either `Agent.V` or `Agent.Q`.
        state : array_like or dict[str, array_like]
            A 1D array representing the value of all states of the MPC, concatenated.
            Otherwise, a dict whose keys are the names of each state, and values are
            their numerical values.
        action : array_like or dict[str, array_like], optional
            Same for `state`, for the action. Only valid if evaluating the action value
            function `Q(s,a)`. For this reason, it can be `None` for `V(s)`.
        perturbation : array_like, optional
            The cost perturbation used to induce exploration in `V(s)`. Can be `None`
            for `Q(s,a)`.
        vals0 : dict[str, array_like] or iterable of, optional
            A dict (or an iterable of dict, in case of `csnlp.MultistartNlp`), whose
            keys are the names of the MPC variables, and values are the numerical
            initial values of each variable. Use this to warm-start the MPC. If `None`,
            and a previous solution (possibly, successful) is available, the MPC solver
            is automatically warm-started

        Returns
        -------
        Solution
            The solution of the MPC.
        """
        # convert state keys into initial state keys (with "_0")
        # convert action dict to vector if not None
        if isinstance(state, dict):
            x0_dict = {f"{k}_0": v for k, v in state.items()}
        else:
            mpcstates = mpc.states
            cumsizes = np.cumsum([s.shape[0] for s in mpcstates.values()][:-1])
            states = np.split(state, cumsizes)
            x0_dict = {f"{k}_0": v for k, v in zip(mpcstates.keys(), states)}
        if action is None:
            u0_vec = None
        elif isinstance(action, dict):
            u0_vec = cs.vertcat(*(action[k] for k in mpc.actions.keys()))
        else:
            u0_vec = action

        # merge (initial) state, action and perturbation in unique dict
        additional_pars = x0_dict
        if u0_vec is not None:
            additional_pars[self.init_action_parameter] = u0_vec
        if self.cost_perturbation_parameter in mpc.parameters:
            additional_pars[self.cost_perturbation_parameter] = (
                0 if perturbation is None else perturbation
            )

        # create pars and vals0
        pars = self._get_parameters()
        if pars is None:
            pars = additional_pars
        elif isinstance(pars, dict):
            pars.update(additional_pars)
        else:  # iterable of dict
            pars = _update_dicts(pars, additional_pars)  # type: ignore
        if vals0 is None and self._last_solution is not None:
            vals0 = self._last_solution.vals

        # solve and store solution
        sol = mpc(pars=pars, vals0=vals0)
        if not self._store_last_successful or sol.success:
            self._last_solution = sol
        return sol

    def state_value(
        self,
        state: Union[npt.ArrayLike, Dict[str, npt.ArrayLike]],
        deterministic: bool = False,
        vals0: Union[
            None, Dict[str, npt.ArrayLike], Iterable[Dict[str, npt.ArrayLike]]
        ] = None,
    ) -> Solution:
        """Computes the state value function `V(s)` approximated by the MPC.

        Parameters
        ----------
        state : array_like or dict[str, array_like]
            The state `s` at which to evaluate the value function `V(s)`. Can either be
            a dict of state names and values, or a single concatenated column vector of
            all the states.
        deterministic : bool, optional
            If `True`, the cost of the MPC is perturbed according to the exploration
            strategy to induce some exploratory behaviour. Otherwise, no perturbation is
            performed. By default, `deterministic=False`.
        vals0 : dict[str, array_like] or iterable of, optional
            A dict (or an iterable of dict, in case of `csnlp.MultistartNlp`), whose
            keys are the names of the MPC variables, and values are the numerical
            initial values of each variable. Use this to warm-start the MPC. If `None`,
            and a previous solution (possibly, successful) is available, the MPC solver
            is automatically warm-started

        Returns
        -------
        Solution
            The solution of the MPC approximating `V(s)` at the given state.
        """
        if deterministic or not self._exploration.can_explore():
            perturbation = None
        else:
            perturbation = self._exploration.perturbation(
                self.cost_perturbation_method,
                size=self.V.parameters[self.cost_perturbation_parameter].shape,
            )
        return self.solve_mpc(self._V, state, perturbation=perturbation, vals0=vals0)

    def action_value(
        self,
        state: Union[npt.ArrayLike, Dict[str, npt.ArrayLike]],
        action: Union[npt.ArrayLike, Dict[str, npt.ArrayLike]],
        vals0: Union[
            None, Dict[str, npt.ArrayLike], Iterable[Dict[str, npt.ArrayLike]]
        ] = None,
    ) -> Solution:
        """Computes the action value function `Q(s,a)` approximated by the MPC.

        Parameters
        ----------
        state : array_like or dict[str, array_like]
            The state `s` at which to evaluate the value function `Q(s,a)`. Can either
            be a dict of state names and values, or a single concatenated column vector
            of all the states.
        action : array_like or dict[str, array_like]
            The action `a` at which to evaluate the value function `Q(s,a)`. Can either
            be a dict of action names and values, or a single concatenated column vector
            of all the actions.
        vals0 : dict[str, array_like] or iterable of, optional
            A dict (or an iterable of dict, in case of `csnlp.MultistartNlp`), whose
            keys are the names of the MPC variables, and values are the numerical
            initial values of each variable. Use this to warm-start the MPC. If `None`,
            and a previous solution (possibly, successful) is available, the MPC solver
            is automatically warm-started

        Returns
        -------
        Solution
            The solution of the MPC approximating `Q(s,a)` at given state and action.
        """
        return self.solve_mpc(self._Q, state, action=action, vals0=vals0)

    def evaluate(
        self,
        env: GymEnvLike[ObsType, ActType],
        episodes: int,
        deterministic: bool = True,
        seed: Union[None, int, Iterable[int]] = None,
        raises: bool = True,
    ) -> npt.NDArray[np.double]:
        """Evaluates the agent in a given environment.

        Note: after solving `V(s)` for the current state `s`, the action is computed and
        passed to the environment as the concatenation of the first optimal action
        variables of the MPC (see `csnlp.Mpc.actions`).

        Parameters
        ----------
        env : gym env
            A gym-like environment where to test the agent in.
        episodes : int
            Number of evaluation episodes.
        deterministic : bool, optional
            Whether the agent should act deterministically; by default, `True`.
        seed : int or iterable of ints, optional
            Each env's seed for RNG.
        raises : bool, optional
            If `True`, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised.

        Returns
        -------
        array of doubles
            The cumulative returns (one return per evaluation episode).

        Raises
        ------
            Raises if the MPC optimization solver fails and `warns_on_exception=False`.
        """
        returns = np.zeros(episodes)

        for episode, current_seed in zip(range(episodes), make_seeds(seed)):
            self.reset()
            state, _ = env.reset(seed=current_seed)
            truncated, terminated = False, False

            while not (truncated or terminated):
                sol = self.state_value(state, deterministic)
                if not sol.success:
                    _raise_or_warn_mpc_failure(
                        f"Solver failed with status: {sol.status}.", raises
                    )
                action = _get_action(self._V, sol)

                state, r, truncated, terminated, _ = env.step(action)
                returns[episode] += r
        return returns

    def _setup_V_and_Q(self, mpc: Mpc[SymType]) -> Tuple[Mpc[SymType], Mpc[SymType]]:
        """Internal utility to setup the function approximators for the value function
        V(s) and the quality function Q(s,a)."""
        na = mpc.na
        if na <= 0:
            raise ValueError(f"Expected Mpc with na>0; got na={na} instead.")

        V, Q = mpc, mpc.copy()
        V.unwrapped.name += "_V"
        Q.unwrapped.name += "_Q"
        a0 = Q.nlp.parameter(self.init_action_parameter, (na, 1))
        perturbation = V.nlp.parameter(self.cost_perturbation_parameter, (na, 1))

        actions = mpc.actions
        u0 = cs.vertcat(*(actions[k][:, 0] for k in actions.keys()))
        f = V.nlp.f
        if mpc.is_wrapped(wrappers.NlpScaling):
            f = mpc.scale(f)

        V.nlp.minimize(f + cs.dot(perturbation, u0))
        Q.nlp.constraint(self.init_action_constraint, u0, "==", a0)
        return V, Q

    def _get_parameters(
        self,
    ) -> Union[None, Dict[str, npt.ArrayLike], Collection[Dict[str, npt.ArrayLike]]]:
        """Internal utility to retrieve parameters of the MPC in order to solve it.
        `Agent` has no learnable parameter, so only fixed parameters are returned."""
        return self._fixed_pars
