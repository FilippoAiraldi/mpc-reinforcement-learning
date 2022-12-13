from abc import ABC, abstractmethod
from typing import Any, Collection, Dict, Generic, Literal, Optional, TypeVar, Union

import casadi as cs
import numpy.typing as npt
from csnlp.wrappers import Mpc

from mpcrl.agents.agent import Agent
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.util.types import GymEnvLike, Tact, Tobs

Tsym = TypeVar("Tsym", cs.SX, cs.MX)
Texp = TypeVar("Texp")


class LearningAgent(Agent, ABC, Generic[Tsym, Texp]):
    """Base class for a learning agent with MPC as policy provider where the main method
    `update`, which is called to update the learnable parameters of the MPC according to
    the underlying learning methodology (e.g., Bayesian Optimization, RL, etc.) is
    abstract and must be implemented by inheriting classes.

    Note: this class makes no assumptions on the learning methodology used to update the
    MPC's learnable parameters."""

    def __init__(
        self,
        mpc: Mpc[Tsym],
        learnable_parameters: LearnableParametersDict[Tsym],
        fixed_parameters: Union[
            None, Dict[str, npt.ArrayLike], Collection[Dict[str, npt.ArrayLike]]
        ] = None,
        exploration: Optional[ExplorationStrategy] = None,
        experience: Optional[ExperienceReplay[Texp]] = None,
        warmstart: Literal["last", "last-successful"] = "last-successful",
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
        warmstart: 'last' or 'last-successful', optional
            The warmstart strategy for the MPC's NLP. If 'last-successful', the last
            successful solution is used to warm start the solver for the next iteration.
            If 'last', the last solution is used, regardless of success or failure.
        name : str, optional
            Name of the agent. If `None`, one is automatically created from a counter of
            the class' instancies.
        """
        super().__init__(mpc, fixed_parameters, exploration, warmstart, name)
        self._experience = (
            ExperienceReplay(maxlen=1) if experience is None else experience
        )
        self._learnable_pars = learnable_parameters

    @property
    def experience(self) -> ExperienceReplay[Texp]:
        """Gets the experience replay memory of the agent."""
        return self._experience

    @property
    def learnable_parameters(self) -> LearnableParametersDict[Tsym]:
        """Gets the parameters of the MPC that can be learnt by the agent."""
        return self._learnable_pars

    def store_experience(self, item: Texp) -> None:
        """Stores the given item in the agent's memory for later experience replay. If
        the agent has no memory, then the method does nothing.

        Parameters
        ----------
        item : Texp
            Item to be stored in memory.
        """
        if self._experience is not None:
            self._experience.append(item)