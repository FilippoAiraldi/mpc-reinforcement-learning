from typing import (
    Any,
    Collection,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    SupportsFloat,
    Tuple,
    Union,
)

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Solution
from csnlp.wrappers import Mpc, NlpSensitivity
from gymnasium import Env
from scipy.linalg import cho_solve
from typing_extensions import TypeAlias

from mpcrl.agents.agent import ActType, ObsType, SymType
from mpcrl.agents.rl_learning_agent import LrType, RlLearningAgent
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy
from mpcrl.core.learning_rate import LearningRate
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.core.schedulers import Scheduler
from mpcrl.core.update import UpdateStrategy
from mpcrl.util.math import cholesky_added_multiple_identities

ExpType: TypeAlias = Tuple[
    npt.NDArray[np.floating],  # gradient of Q(s,a)
    npt.NDArray[np.floating],  # (approximate) hessian of Q(s,a)
]


class LstdQLearningAgent(
    RlLearningAgent[SymType, ExpType, LrType], Generic[SymType, LrType]
):
    """Second-order Least-Squares Temporal Difference (LSTD) Q-learning agent, as first
    proposed in a simpler format in [1], and then in [2].

    The Q-learning agent uses an MPC controller as policy provider and function
    approximation, and adjusts its parametrization according to the temporal-difference
    error, with the goal of improving the policy, in an indirect fashion by learning the
    action value function.

    References
    ----------
    [1] Gros, S. and Zanon, M., 2019. Data-driven economic NMPC using reinforcement
        learning. IEEE Transactions on Automatic Control, 65(2), pp. 636-648.
    [2] Esfahani, H.N., Kordabad, A.B. and Gros, S., 2021, June. Approximate Robust NMPC
        using Reinforcement Learning. In 2021 European Control Conference (ECC), pp.
        132-137. IEEE.
    """

    __slots__ = (
        "_dQdtheta",
        "_d2Qdtheta2",
        "td_errors",
        "cho_maxiter",
        "cho_solve_kwargs",
    )

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
        max_percentage_update: float = float("+inf"),
        warmstart: Literal["last", "last-successful"] = "last-successful",
        hessian_type: Literal["approx", "full"] = "approx",
        record_td_errors: bool = False,
        cho_maxiter: int = 1000,
        cho_solve_kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Instantiates the LSTD Q-learning agent.

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
            transition. In the case of LSTD Q-learning, each memory item consists of the
            action value function's gradient and hessian computed at each (succesful)
            env's step.
        max_percentage_update : float, optional
            A positive float that specifies the maximum percentage the parameters can be
            changed during each update. For example, `max_percentage_update=0.5` means
            that the parameters can be updated by up to 50% of their current value. By
            default, it is set to `+inf`.
        warmstart: 'last' or 'last-successful', optional
            The warmstart strategy for the MPC's NLP. If 'last-successful', the last
            successful solution is used to warm start the solver for the next iteration.
            If 'last', the last solution is used, regardless of success or failure.
        hessian_type : 'approx' or 'full', optional
            The type of hessian to use in this second-order algorithm. If `approx`, an
            easier approximation of it is used; otherwise, the full hessian is computed
            but this is much more expensive.
        record_td_errors: bool, optional
            If `True`, the TD errors are recorded in the field `td_errors`, which
            otherwise is `None`. By default, does not record them.
        cho_maxiter : int, optional
            Maximum number of iterations in the Cholesky's factorization with additive
            multiples of the identity to ensure positive definiteness of the hessian. By
            default, `1000`.
        cho_solve_kwargs : kwargs for scipy.linalg.cho_solve, optional
            The optional kwargs to be passed to `scipy.linalg.cho_solve`. If `None`, it
            is equivalent to `cho_solve_kwargs = {'check_finite': False }`.
        name : str, optional
            Name of the agent. If `None`, one is automatically created from a counter of
            the class' instancies.
        """
        super().__init__(
            mpc=mpc,
            update_strategy=update_strategy,
            discount_factor=discount_factor,
            learning_rate=learning_rate,  # type: ignore[arg-type]
            learnable_parameters=learnable_parameters,
            fixed_parameters=fixed_parameters,
            exploration=exploration,
            experience=experience,
            max_percentage_update=max_percentage_update,
            warmstart=warmstart,
            name=name,
        )
        self._dQdtheta, self._d2Qdtheta2 = self._init_Q_derivatives(hessian_type)
        self.cho_maxiter = cho_maxiter
        if cho_solve_kwargs is None:
            cho_solve_kwargs = {"check_finite": False}
        self.cho_solve_kwargs = cho_solve_kwargs
        self.td_errors: Optional[List[float]] = [] if record_td_errors else None

    def store_experience(  # type: ignore[override]
        self, cost: SupportsFloat, solQ: Solution[SymType], solV: Solution[SymType]
    ) -> None:
        """Stores the gradient and hessian for the current transition in memoru.

        Parameters
        ----------
        cost : float
            The cost of this state transition.
        solQ : Solution[SymType]
            The solution to `Q(s,a)`.
        solV : Solution[SymType]
            The solution to `V(s+)`.
        """
        sol_values = solQ.all_vals
        dQ: npt.NDArray[np.floating] = self._dQdtheta(sol_values).full().reshape(-1, 1)
        ddQ: npt.NDArray[np.floating] = self._d2Qdtheta2(sol_values).full()
        td_error: float = cost + self.discount_factor * solV.f - solQ.f
        g = -td_error * dQ
        H = dQ @ dQ.T - td_error * ddQ
        if self.td_errors is not None:
            self.td_errors.append(td_error)
        return super().store_experience((g, H))

    def update(self) -> Optional[str]:
        sample = self.experience.sample()
        gradient, Hessian = (np.mean(tuple(o), axis=0) for o in zip(*sample))
        R = cholesky_added_multiple_identities(Hessian, maxiter=self.cho_maxiter)
        step = cho_solve((R, True), gradient, **self.cho_solve_kwargs).reshape(-1)
        return self._do_gradient_update(step)

    def train_one_episode(
        self,
        env: Env[ObsType, ActType],
        episode: int,
        init_state: ObsType,
        raises: bool = True,
    ) -> float:
        truncated = terminated = False
        timestep = 0
        rewards = 0.0
        state = init_state

        # solve for the first action
        action, solV = self.state_value(state, deterministic=False)
        if not solV.success:
            self.on_mpc_failure(episode, None, solV.status, raises)

        while not (truncated or terminated):
            # compute Q(s,a)
            solQ = self.action_value(state, action)

            # step the system with action computed at the previous iteration
            state, cost, truncated, terminated, _ = env.step(action)

            # compute V(s+)
            action, solV = self.state_value(state, deterministic=False)
            if solQ.success and solV.success:
                self.store_experience(cost, solQ, solV)
            else:
                self.on_mpc_failure(
                    episode, timestep, f"{solQ.status}/{solV.status}", raises
                )

            # increase counters
            rewards += float(cost)
            timestep += 1
            self.on_env_step(env, episode, timestep)  # better to call this at the end
        return rewards

    def _init_Q_derivatives(
        self, hessian_type: Literal["approx", "full"]
    ) -> Tuple[cs.Function, cs.Function]:
        """Internal utility to compute the derivative of Q(s,a) w.r.t. the learnable
        parameters, a.k.a., theta."""
        theta = cs.vertcat(*self._learnable_pars.sym.values())
        nlp = self._Q.nlp
        nlp_ = NlpSensitivity(nlp, target_parameters=theta)
        Lt = nlp_.jacobians["L-p"]  # a.k.a., dQdtheta
        Ltt = nlp_.hessians["L-pp"]  # a.k.a., approximated d2Qdtheta2
        if hessian_type == "approx":
            d2Qdtheta2 = Ltt
        elif hessian_type == "full":
            dydtheta, _ = nlp_.parametric_sensitivity(second_order=False)
            d2Qdtheta2 = dydtheta.T @ nlp_.jacobians["K-p"] + Ltt
        else:
            raise ValueError(f"Invalid type of hessian; got {hessian_type}.")

        # convert to functions (much faster runtime)
        input = cs.vertcat(nlp.p, nlp.x, nlp.lam_g, nlp.lam_h, nlp.lam_lbx, nlp.lam_ubx)
        dQdtheta_ = cs.Function("dQdtheta", [input], [Lt])
        d2Qdtheta2_ = cs.Function("d2Qdtheta2", [input], [d2Qdtheta2])
        assert (
            not dQdtheta_.has_free() and not d2Qdtheta2_.has_free()
        ), "Internal error in Q derivatives."
        return dQdtheta_, d2Qdtheta2_
