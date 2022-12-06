from contextlib import contextmanager
from copy import deepcopy
from itertools import repeat
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


class Agent(Named, Generic[T]):
    """Simple MPC-based agent with a fixed (i.e., non-learnable) MPC controller.

    In this agent, the MPC controller is used as policy provider, as well as to provide
    the value function `V(s)` and quality function `Q(s,a)`, where `s` and `a` are the
    state and action of the environment, respectively. However, this class does not use
    any RL method to improve its MPC policy."""

    cost_perturbation_par = "cost_perturbation"
    init_action_par = init_action_con = "a_init"

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
            the attributes `cost_perturbation_par`, `init_action_par` and
            `init_action_con`.
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
