from abc import ABC
from typing import Any, Collection, Generic, Literal, Optional, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp.wrappers import Mpc
from scipy.linalg import cho_solve

from mpcrl.agents.agent import SymType
from mpcrl.agents.learning_agent import LearningAgent
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy
from mpcrl.core.learning_rate import LearningRate, LrType
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.core.schedulers import Scheduler
from mpcrl.core.update import UpdateStrategy
from mpcrl.util.math import cholesky_added_multiple_identities

ExpType = TypeVar("ExpType")


class RlLearningAgent(
    LearningAgent[SymType, ExpType], ABC, Generic[SymType, ExpType, LrType]
):
    """Base class for learning agents that employe gradient-based RL strategies to
    learn/improve the MPC policy."""

    def __init__(
        self,
        mpc: Mpc[SymType],
        update_strategy: Union[int, UpdateStrategy],
        discount_factor: float,
        learning_rate: Union[LrType, Scheduler[LrType], LearningRate[LrType]],
        learnable_parameters: LearnableParametersDict[SymType],
        fixed_parameters: Union[
            None, dict[str, npt.ArrayLike], Collection[dict[str, npt.ArrayLike]]
        ] = None,
        exploration: Optional[ExplorationStrategy] = None,
        experience: Optional[ExperienceReplay[ExpType]] = None,
        max_percentage_update: float = float("+inf"),
        warmstart: Literal["last", "last-successful"] = "last-successful",
        cho_maxiter: int = 1000,
        cho_solve_kwargs: Optional[dict[str, Any]] = None,
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
        max_percentage_update : float, optional
            A positive float that specifies the maximum percentage the parameters can be
            changed during each update. For example, `max_percentage_update=0.5` means
            that the parameters can be updated by up to 50% of their current value. By
            default, it is set to `+inf`.
        warmstart: 'last' or 'last-successful', optional
            The warmstart strategy for the MPC's NLP. If 'last-successful', the last
            successful solution is used to warm start the solver for the next iteration.
            If 'last', the last solution is used, regardless of success or failure.
        cho_maxiter : int, optional
            Maximum number of iterations in the Cholesky's factorization with additive
            multiples of the identity to ensure positive definiteness of the hessian. By
            default, `1000`. Only used if the algorithm exploits the hessian.
        cho_solve_kwargs : kwargs for scipy.linalg.cho_solve, optional
            The optional kwargs to be passed to `scipy.linalg.cho_solve` to solve for
            the inversion of the hessian. If `None`, it is equivalent to
            `cho_solve_kwargs = {'check_finite': False }`. Only used if the algorithm
            exploits the hessian.
        name : str, optional
            Name of the agent. If `None`, one is automatically created from a counter of
            the class' instancies.
        """
        if max_percentage_update <= 0.0:
            raise ValueError("Max percentage update must be in range (0, +inf).")
        if not isinstance(learning_rate, LearningRate):
            learning_rate = LearningRate(learning_rate, "on_update")
        self._learning_rate: LearningRate[LrType] = learning_rate
        self.discount_factor = discount_factor
        self.cho_maxiter = cho_maxiter
        if cho_solve_kwargs is None:
            cho_solve_kwargs = {"check_finite": False}
        self.cho_solve_kwargs = cho_solve_kwargs
        self.max_percentage_update = max_percentage_update
        super().__init__(
            mpc,
            update_strategy,
            learnable_parameters,
            fixed_parameters,
            exploration,
            experience,
            warmstart,
            name,
        )
        self._update_solver = self._init_update_solver()

    @property
    def learning_rate(self) -> LrType:
        """Gets the learning rate of the learning agent."""
        return self._learning_rate.value

    def establish_callback_hooks(self) -> None:
        super().establish_callback_hooks()
        lr_hook = self._learning_rate.hook
        if lr_hook is not None:
            self.hook_callback(
                repr(self._learning_rate),
                lr_hook,
                lambda *_, **__: self._learning_rate.step(),
            )

    def _init_update_solver(self) -> Optional[cs.Function]:
        """Internal utility to initialize the update solver, in particular, a QP solver.
        If the update is unconstrained, then no solver is initialized, i.e., `None` is
        returned."""
        if (
            self.max_percentage_update == float("+inf")
            and np.isneginf(self._learnable_pars.lb).all()
            and np.isposinf(self._learnable_pars.ub).all()
        ):
            return None

        sym_type = cs.MX
        n_p = self._learnable_pars.size
        theta = sym_type.sym("theta", n_p, 1)
        theta_new = sym_type.sym("theta+", n_p, 1)
        dtheta = theta_new - theta
        g = sym_type.sym("g", n_p, 1)  # includes learning rate
        H = sym_type.sym("H", n_p, n_p)
        qp = {
            "x": theta_new,
            "f": 0.5 * dtheta.T @ H @ dtheta + g.T @ dtheta,
            "p": cs.veccat(theta, g, H),
        }
        opts = {"expand": True, "print_iter": False, "print_header": False}
        return cs.qpsol(f"qpsol_{self.name}", "qrqp", qp, opts)

    def _get_update_bounds(
        self,
        theta: npt.NDArray[np.floating],
        eps: float = 0.1,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Internal utility to retrieve the current bounds on the QP solver for an
        update. Only called if the update problem is not unconstrained, i.e., there are
        either some lb or ub, or a maximum percentage update"""
        lb = self._learnable_pars.lb
        ub = self._learnable_pars.ub
        perc = self.max_percentage_update
        if perc == float("+inf"):
            return lb, ub
        max_update_delta = np.maximum(np.abs(perc * theta), eps)
        lb = np.maximum(lb, theta - max_update_delta)
        ub = np.minimum(ub, theta + max_update_delta)
        return lb, ub

    def _do_gradient_update(
        self, g: npt.NDArray[np.floating], H: Optional[npt.NDArray[np.floating]]
    ) -> Optional[str]:
        """Internal utility to do the actual gradient update by either calling the QP
        solver or by updating the parameters maually."""
        solver = self._update_solver
        theta = self._learnable_pars.value  # current values of parameters
        lr = self.learning_rate
        lr_g = lr * g
        if H is None:
            p = lr_g
        else:
            L = cholesky_added_multiple_identities(H, maxiter=self.cho_maxiter)
            p = lr * cho_solve((L, True), g, **self.cho_solve_kwargs)

        if solver is None:
            self._learnable_pars.update_values(theta - p)
            return None

        H = np.eye(theta.shape[0]) if H is None else L @ L.T
        lb, ub = self._get_update_bounds(theta)
        params = np.concatenate((theta, lr_g, H), None)
        theta_new = solver(p=params, lbx=lb, ubx=ub, x0=theta - p)["x"].full()[:, 0]
        self._learnable_pars.update_values(np.clip(theta_new, lb, ub))
        stats = solver.stats()
        return None if stats["success"] else stats["return_status"]
