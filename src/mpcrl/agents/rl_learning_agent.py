from abc import ABC
from typing import Collection, Dict, Generic, Literal, Optional, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp.wrappers import Mpc

from mpcrl.agents.agent import SymType
from mpcrl.agents.learning_agent import LearningAgent
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy
from mpcrl.core.learning_rate import LearningRate, LrType
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.core.schedulers import NoScheduling, Scheduler
from mpcrl.core.update import UpdateStrategy

ExpType = TypeVar("ExpType")


class RlLearningAgent(
    LearningAgent[SymType, ExpType], ABC, Generic[SymType, ExpType, LrType]
):
    """Base class for learning agents that employe gradient-based RL strategies to
    learn/improve the MPC policy."""

    __slots__ = ("_learning_rate", "discount_factor", "_update_solver")

    def __init__(
        self,
        mpc: Mpc[SymType],
        update_strategy: Union[int, UpdateStrategy],
        discount_factor: float,
        learning_rate: Union[LrType, Scheduler[LrType], LearningRate[LrType]],
        learnable_parameters: LearnableParametersDict[SymType],
        fixed_parameters: Union[
            None, Dict[str, npt.ArrayLike], Collection[Dict[str, npt.ArrayLike]]
        ] = None,
        exploration: Optional[ExplorationStrategy] = None,
        experience: Optional[ExperienceReplay[ExpType]] = None,
        warmstart: Literal["last", "last-successful"] = "last-successful",
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
        update_strategy : UpdateStrategy or int
            The strategy used to decide which frequency to update the mpc parameters
            with. If an `int` is passed, then the default strategy that updates every
            `n` env's steps is used (where `n` is the argument passed); otherwise, an
            instance of `UpdateStrategy` can be passed to specify these in more details.
        discount_factor : float
            In RL, the factor that discounts future rewards in favor of immediate
            rewards. Usually denoted as `\\gamma`. Should be a number in (0, 1).
        learning_rate : float/array, scheduler or LearningRate
            The learning rate of the algorithm. A float/array can be passed in case the
            learning rate must stay constant; otherwise, a scheduler can be passed which
            will be stepped `on_update` by default. Otherwise, a LearningRate can be
            passed, allowing to specify both the scheduling and stepping strategies of
            the learning rate.
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
        if not isinstance(learning_rate, LearningRate):
            learning_rate = LearningRate(learning_rate)
        self._learning_rate: LearningRate[LrType] = learning_rate
        self.discount_factor = discount_factor
        super().__init__(
            mpc=mpc,
            update_strategy=update_strategy,
            learnable_parameters=learnable_parameters,
            fixed_parameters=fixed_parameters,
            exploration=exploration,
            experience=experience,
            warmstart=warmstart,
            name=name,
        )
        self._update_solver = self._init_update_solver()

    @property
    def learning_rate(self) -> LrType:
        """Gets the learning rate of the learning agent."""
        return self._learning_rate.value

    def establish_callback_hooks(self) -> None:
        super().establish_callback_hooks()
        if not isinstance(self._learning_rate.scheduler, NoScheduling):
            self.hook_callback(
                repr(self._learning_rate),
                self._learning_rate.hook,
                self._learning_rate.step,
            )

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
