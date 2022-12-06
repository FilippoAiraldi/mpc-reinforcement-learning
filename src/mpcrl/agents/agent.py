from contextlib import contextmanager
from copy import deepcopy
from typing import (
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

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Solution
from csnlp.wrappers import Mpc

from mpcrl.core.exploration import ExplorationStrategy, NoExploration
from mpcrl.util.named import Named

T = TypeVar("T", cs.SX, cs.MX)


def _update_dicts(sinks: Iterable[Dict], source: Dict) -> Iterator[Dict]:
    """Internal utility for updating dicts `sinks` with one `source`."""
    for sink in sinks:
        sink.update(source)
        yield sink


class Agent(Named, Generic[T]):
    """Simple MPC-based agent with a fixed (i.e., non-learnable) MPC controller.

    In this agent, the MPC controller is used as policy provider, as well as to provide
    the value function `V(s)` and quality function `Q(s,a)`, where `s` and `a` are the
    state and action of the environment, respectively. However, this class does not use
    any RL method to improve its MPC policy."""

    cost_perturbation_method = "normal"
    cost_perturbation_parameter = "cost_perturbation"
    init_action_parameter = init_action_constraint = "a_init"

    def __init__(
        self,
        mpc: Mpc[T],
        fixed_parameters: Dict[str, npt.ArrayLike] = None,
        exploration: ExplorationStrategy = None,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        name: str = None,
    ) -> None:
        """Instantiates an agent with an MPC controller.

        Parameters
        ----------
        mpc : Mpc[casadi.SX or MX]
            The MPC controller used as policy provider by this agent. The instance is
            modified in place, so it is recommended not to modify it further. Moreover,
            some parameter and constraint names will need to be created, so an error is
            thrown if these names are already in use in the mpc. These names are under
            the attributes `perturbation_parameter`, `action_parameter` and
            `action_constraint`.
        fixed_pars : dict[str, array_like], optional
            A dict containing whose keys are the names of the MPC parameters and the
            values are their corresponding values. Use this to specify fixed parameters,
            that is, parameters that are non-learnable. If `None`, then no fixed
            parameter is assumed.
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
        super().__init__(name)
        self._V, self._Q = self._setup_V_and_Q(mpc)
        self._fixed_pars = {} if fixed_parameters is None else fixed_parameters
        self._exploration = NoExploration() if exploration is None else exploration
        self._last_solution: Optional[Solution[T]] = None
        self._store_last_successful = warmstart == "last-successful"

    @property
    def unwrapped(self) -> "Agent":
        """Gets the underlying wrapped instance of an agent."""
        return self

    @property
    def V(self) -> Mpc[T]:
        """Gets the MPC function approximation of the state value function `V(s)`."""
        return self._V

    @property
    def Q(self) -> Mpc[T]:
        """Gets the MPC function approximation of the action value function `Q(s,a)`."""
        return self._Q

    @property
    def fixed_parameters(self) -> Dict[str, npt.ArrayLike]:
        """Gets the fixed parameters of the MPC controller (i.e., non-learnable)."""
        return self._fixed_pars

    @property
    def exploration(self) -> ExplorationStrategy:
        """Gets the exploration strategy used within this agent."""
        return self._exploration

    def copy(self) -> "Agent":
        """Creates a deepcopy of this Agent's instance.

        Returns
        -------
        Agent
            A new instance of the agent.
        """
        with self._Q.fullstate(), self._V.fullstate():
            return deepcopy(self)

    @contextmanager
    def pickleable(self) -> Iterator[None]:
        """Context manager that makes the agent and its function approximators
        pickleable."""
        with self._Q.pickleable(), self._V.pickleable():
            yield

    def solve_mpc(
        self,
        mpc: Mpc[T],
        state: Union[npt.ArrayLike, Dict[str, npt.ArrayLike]],
        action: Union[npt.ArrayLike, Dict[str, npt.ArrayLike]] = None,
        pars: Union[
            None, Dict[str, npt.ArrayLike], Iterable[Dict[str, npt.ArrayLike]]
        ] = None,
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
        pars : dict[str, array_like] or iterable of, optional
            A dict (or an iterable of dict, in case of `csnlp.MultistartNlp`), whose
            keys are the names of the MPC parameters, and values are the numerical
            values of each parameter. Pass `None` in case the MPC has no parameter.
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
        # convert state keys into initial state keys (with "_0") and, if not None,
        # convert action dict to vector
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

        # merge (initial) state and action in unique dict,
        additional_pars = x0_dict
        if u0_vec is not None:
            additional_pars[self.init_action_parameter] = u0_vec

        # create pars and vals0
        if pars is None:
            pars = additional_pars
        elif isinstance(pars, dict):
            pars.update(additional_pars)
        else:  # iterable of dict
            pars = _update_dicts(pars, additional_pars)
        if vals0 is None and self._last_solution is not None:
            vals0 = self._last_solution.vals

        # solve and store solution
        sol = mpc(pars=pars, vals0=vals0)
        if not self._store_last_successful or sol.success:
            self._last_solution = sol
        return sol

    def _setup_V_and_Q(self, mpc: Mpc[T]) -> Tuple[Mpc[T], Mpc[T]]:
        """Internal utility to setup the function approximators for the value function
        V(s) and the quality function Q(s,a)."""
        na = mpc.na
        if na <= 0:
            raise ValueError(f"Expected Mpc with na>0; got na={na} instead.")
        V, Q = mpc, mpc.copy()
        actions = mpc.actions
        u0 = cs.vertcat(*(actions[k][:, 0] for k in actions.keys()))
        perturbation = V.nlp.parameter(self.cost_perturbation_par, (na, 1))
        V.nlp.minimize(V.nlp.f + cs.dot(perturbation, u0))
        a0 = Q.nlp.parameter(self.init_action_par, (na, 1))
        Q.nlp.constraint(self.init_action_con, u0, "==", a0)
        return V, Q
