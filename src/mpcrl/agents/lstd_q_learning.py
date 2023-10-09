from collections.abc import Collection
from typing import Callable, Generic, Literal, Optional, SupportsFloat, Union

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Solution
from csnlp.wrappers import Mpc, NlpSensitivity
from gymnasium import Env
from typing_extensions import TypeAlias

from mpcrl.agents.agent import ActType, ObsType, SymType
from mpcrl.agents.rl_learning_agent import LrType, RlLearningAgent
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.core.update import UpdateStrategy
from mpcrl.optim.gradient_based_optimizer import GradientBasedOptimizer

ExpType: TypeAlias = tuple[
    npt.NDArray[np.floating],  # gradient of Bellman residuals w.r.t. theta
    npt.NDArray[np.floating],  # (approximate) hessian of Bellman residuals w.r.t. theta
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

    def __init__(
        self,
        mpc: Mpc[SymType],
        update_strategy: Union[int, UpdateStrategy],
        discount_factor: float,
        optimizer: GradientBasedOptimizer,
        learnable_parameters: LearnableParametersDict[SymType],
        fixed_parameters: Union[
            None, dict[str, npt.ArrayLike], Collection[dict[str, npt.ArrayLike]]
        ] = None,
        exploration: Optional[ExplorationStrategy] = None,
        experience: Union[None, int, ExperienceReplay[ExpType]] = None,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        hessian_type: Literal["none", "approx", "full"] = "approx",
        record_td_errors: bool = False,
        use_last_action_on_fail: bool = False,
        remove_bounds_on_initial_action: bool = False,
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
            rewards. Usually denoted as `\\gamma`. Should be a number in (0, 1].
        optimizer : GradientBasedOptimizer
            A gradient-based optimizer (e.g., `mpcrl.optim.GradientDescent`) to compute
            the updates of the learnable parameters, based on the current gradient-based
            RL algorithm.
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
        experience : int or ExperienceReplay, optional
            The container for experience replay memory. If `None` is passed, then a
            memory with length 1 is created, i.e., it keeps only the latest memory
            transition.  If an integer `n` is passed, then a memory with the length `n`
            is created and with sample size `n`.
            In the case of LSTD Q-learning, each memory item consists of the action
            value function's gradient and hessian computed at each (succesful) env's
            step.
        warmstart: 'last' or 'last-successful', optional
            The warmstart strategy for the MPC's NLP. If 'last-successful', the last
            successful solution is used to warm start the solver for the next iteration.
            If 'last', the last solution is used, regardless of success or failure.
        hessian_type : 'none', 'approx' or 'full', optional
            The type of hessian to use in this (potentially) second-order algorithm.
            If 'none', no second order information is used. If `approx`, an easier
            approximation of it is used; otherwise, the full hessian is computed but
            this is much more expensive. This option must be in accordance with the
            choice of `optimizer`, that is, if the optimizer does not use second order
            information, then this option must be set to `none`.
        record_td_errors: bool, optional
            If `True`, the TD errors are recorded in the field `td_errors`, which
            otherwise is `None`. By default, does not record them.
        use_last_action_on_fail : bool, optional
            When `True`, if the MPC solver fails in solving the state value function
            `V(s)`, the last successful action is returned. When `False`, the action
            from the last MPC iteration is returned instead. By default, `False`.
        remove_bounds_on_initial_action : bool, optional
            When `True`, the upper and lower bounds on the initial action are removed in
            the action-value function approximator Q(s,a) since the first action is
            constrained to be equal to the initial action. This is useful to avoid
            issues in the LICQ of the NLP. However, it can lead to numerical problems.
            By default, `False`.
        name : str, optional
            Name of the agent. If `None`, one is automatically created from a counter of
            the class' instancies.
        """
        super().__init__(
            mpc=mpc,
            update_strategy=update_strategy,
            discount_factor=discount_factor,
            learnable_parameters=learnable_parameters,
            optimizer=optimizer,
            fixed_parameters=fixed_parameters,
            exploration=exploration,
            experience=experience,
            warmstart=warmstart,
            use_last_action_on_fail=use_last_action_on_fail,
            remove_bounds_on_initial_action=remove_bounds_on_initial_action,
            name=name,
        )
        self.hessian_type = hessian_type
        self._sensitivity = self._init_sensitivity(hessian_type)
        self.td_errors: Optional[list[float]] = [] if record_td_errors else None

    def update(self) -> Optional[str]:
        sample = self.experience.sample()
        gradients = []
        hessians = []
        for g, H in sample:
            gradients.append(g)
            hessians.append(H)
        gradient = np.mean(gradients, 0)
        hessian = np.mean(hessians, 0) if self.hessian_type != "none" else None
        return self._do_gradient_update(gradient.reshape(-1), hessian)

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
        action, solV = self.state_value(state, False)
        if not solV.success:
            self.on_mpc_failure(episode, None, solV.status, raises)

        while not (truncated or terminated):
            # compute Q(s,a)
            solQ = self.action_value(state, action)

            # step the system with action computed at the previous iteration
            new_state, cost, truncated, terminated, _ = env.step(action)
            self.on_env_step(env, episode, timestep)

            # compute V(s+) and store transition
            new_action, solV = self.state_value(new_state, False)
            if not self._try_store_experience(cost, solQ, solV):
                self.on_mpc_failure(
                    episode, timestep, f"{solQ.status}/{solV.status}", raises
                )

            # increase counters
            state = new_state
            action = new_action
            rewards += float(cost)
            timestep += 1
            self.on_timestep_end(env, episode, timestep)
        return rewards

    def _init_sensitivity(
        self, hessian_type: Literal["none", "approx", "full"]
    ) -> Callable[[cs.DM], tuple[np.ndarray, np.ndarray]]:
        """Internal utility to compute the derivative of Q(s,a) w.r.t. the learnable
        parameters, a.k.a., theta."""
        theta = cs.vvcat(self._learnable_pars.sym.values())
        nlp = self._Q.nlp
        nlp_ = NlpSensitivity(nlp, theta)
        Lt = nlp_.jacobians["L-p"]  # a.k.a., dQdtheta
        Ltt = nlp_.hessians["L-pp"]  # a.k.a., approximated d2Qdtheta2
        if hessian_type == "none":
            d2Qdtheta2 = cs.DM.nan()
        elif hessian_type == "approx":
            d2Qdtheta2 = Ltt
        elif hessian_type == "full":
            dydtheta, _ = nlp_.parametric_sensitivity(second_order=False)
            d2Qdtheta2 = dydtheta.T @ nlp_.jacobians["K-p"] + Ltt
        else:
            raise ValueError(f"Invalid type of hessian; got {hessian_type}.")

        # convert to function (much faster runtime)
        x_lam_p = cs.vertcat(nlp.primal_dual, nlp.p)
        sensitivity = cs.Function(
            "Q_sensitivity",
            (x_lam_p,),
            (Lt, d2Qdtheta2),
            ("x_lam_p",),
            ("dQ", "d2Q"),
            {"post_expand": True, "cse": True},
        )

        # wrap to conveniently return numpy arrays
        def func(sol_values: cs.DM) -> tuple[np.ndarray, np.ndarray]:
            dQ, ddQ = sensitivity(sol_values)
            return dQ.full().reshape(-1, 1), ddQ.full()

        return func

    def _try_store_experience(
        self, cost: SupportsFloat, solQ: Solution[SymType], solV: Solution[SymType]
    ) -> bool:
        """Internal utility that tries to store the gradient and hessian for the current
        transition in memory, if both V and Q were successful; otherwise, does not store
        it. Returns whether it was successful or not."""
        if solQ.success and solV.success:
            sol_values = solQ.all_vals
            dQ, ddQ = self._sensitivity(sol_values)
            td_error = cost + self.discount_factor * solV.f - solQ.f
            g = -td_error * dQ
            H = (dQ @ dQ.T - td_error * ddQ) if self.hessian_type != "none" else np.nan
            self.store_experience((g, H))
            success = True
        else:
            td_error = np.nan
            success = False
        if self.td_errors is not None:
            self.td_errors.append(td_error)
        return success
