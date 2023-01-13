from itertools import chain, repeat
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
from csnlp.util.math import prod
from csnlp.wrappers import Mpc, NlpSensitivity
from scipy.linalg import lstsq
from typing_extensions import TypeAlias

from mpcrl.agents.agent import ActType, ObsType, SymType
from mpcrl.agents.rl_learning_agent import LrType, RlLearningAgent
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy
from mpcrl.core.learning_rate import LearningRate
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.core.schedulers import Scheduler
from mpcrl.core.update import UpdateStrategy
from mpcrl.util.math import monomial_powers
from mpcrl.util.types import GymEnvLike

ExpType: TypeAlias = Tuple[
    npt.NDArray[np.double],  # rollout's costs
    npt.NDArray[np.double],  # rollout's state feature vectors Phi(s)
    npt.NDArray[np.double],  # rollout's Psi(s,a)
    npt.NDArray[np.double],  # rollout's gradient of policy
]


class LstdDpgAgent(RlLearningAgent[SymType, ExpType, LrType], Generic[SymType, LrType]):
    """Least-Squares Temporal Difference (LSTD) Deterministic Policy Gradient (DPG)
    agent, as first introduced in [1] as its stochastic counterpart, and then refined in
    [2]. An application can be found in [3].

    The DPG agent uses an MPC controller as policy provider and function approximation,
    and adjusts its parametrization according to the temporal-difference error, with the
    goal of improving the policy, in a direct fashion by estimating the gradient of the
    policy and descending in its direction.

    References
    ----------
    [1] Gros, S. and Zanon, M., 2021, May. Reinforcement Learning based on MPC
        and the Stochastic Policy Gradient Method. In 2021 American Control
        Conference (ACC) (pp. 1947-1952). IEEE.
    [2] Gros, S. and Zanon, M., 2019. Towards Safe Reinforcement Learning Using NMPC and
        Policy Gradients: Part II - Deterministic Case. arXiv preprint arXiv:1906.04034.
    [3] Cai, W., Kordabad, A.B., Esfahani, H.N., Lekkas, A.M. and Gros, S., 2021,
        December. MPC-based reinforcement learning for a simplified freight mission of
        autonomous surface vehicles. In 2021 60th IEEE Conference on Decision and
        Control (CDC) (pp. 2990-2995). IEEE.
    """

    __slots__ = (
        "_dKdtheta",
        "_dKdy",
        "_dydu0",
        "_Phi",
        "rollout_length",
        "_current_rollout",
        "lstsq_kwargs",
        "policy_performances",
        "policy_gradients",
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
        warmstart: Literal["last", "last-successful"] = "last-successful",
        rollout_length: Optional[int] = None,
        record_policy_performance: bool = False,
        record_policy_gradient: bool = False,
        state_features: Optional[cs.Function] = None,
        lstsq_kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Instantiates the LSTD DPG agent.

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
            `n` episodes is used (where `n` is the argument passed); otherwise, an
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
            transition. In case of LSTD DPG, each memory item is obtain from a single
            rollout, and is a 4-tuple that contains: costs, state vector features (Phi),
            Psi (a temporary value), and the gradient of the policy.
        warmstart: 'last' or 'last-successful', optional
            The warmstart strategy for the MPC's NLP. If 'last-successful', the last
            successful solution is used to warm start the solver for the next iteration.
            If 'last', the last solution is used, regardless of success or failure.
        rollout_length : int, optional
            Time-step length of a closed-loop simulation, which defines a complete
            trajectory of the states, and is saved in the experience as a single item
            (since LSTD DPG needs to draw samples of trajectories). In case the env is
            episodic, it can be `None`, in which case the rollout length coincides with
            the episode's. In case the env is not episodic, i.e., it never terminates, a
            rollout length must be given in order to save the current trajectory as an
            atomic item in memory.
        record_policy_performance: bool, optional
            If `True`, the performance of each rollout is stored in the filed
            `policy_performances`, which otherwise is `None`. By default, does not
            record them.
        record_policy_gradient: bool, optional
            If `True`, the (estimated) policy gradient of each update is stored in the
            filed `policy_gradients`, which otherwise is `None`. By default, does not
            record them.
        state_features : casadi.Function, optional
            The state feature vector to be used in the linear approximation of the
            value function, which takes the form of
            ```
            V_v(s) = Phi(s)^T * v
            ```
            where `s` is the state, `v` are the weights, and `Phi(s)` is the state
            feature vector. This function is assumed to have one input and one output.
            By default, if not provided, it is designed as all monomials of the state
            with degrees <= 2 (see `LstdDpgAgent.monomials_state_features`).
        lstsq_kwargs : kwargs for scipy.linalg.lstsq, optional
            The optional kwargs to be passed to `scipy.linalg.lstsq`. If `None`, it
            is equivalent to
            ```
            {'cond': 1e-7, 'check_finite': False, 'lapack_driver': 'gelsy'}
            ```.
        name : str, optional
            Name of the agent. If `None`, one is automatically created from a counter of
            the class' instancies.
        """
        # change default update hook to 'on_episode_end'
        if not isinstance(update_strategy, UpdateStrategy):
            update_strategy = UpdateStrategy(update_strategy, "on_episode_end")
        super().__init__(
            mpc=mpc,
            update_strategy=update_strategy,
            discount_factor=discount_factor,
            learning_rate=learning_rate,  # type: ignore[arg-type]
            learnable_parameters=learnable_parameters,
            fixed_parameters=fixed_parameters,
            exploration=exploration,
            experience=experience,
            warmstart=warmstart,
            name=name,
        )
        # initialize derivatives and state feature vector
        self._dKdtheta, self._dKdy, self._dydu0 = self._init_dpg_derivatives()
        # TODO: check that monomial with power 0 makes Psi ill-posed
        self._Phi = (
            LstdDpgAgent.monomials_state_features(mpc.ns, mpc.sym_type.__name__, 0, 2)
            if state_features is None
            else state_features
        )
        # initialize others
        if lstsq_kwargs is None:
            lstsq_kwargs = {
                "check_finite": False,
                "lapack_driver": "gelsy",
                "cond": 1e-7,
            }
        self.lstsq_kwargs = lstsq_kwargs
        self.rollout_length = rollout_length
        self._rollout: List[
            Tuple[
                ObsType,
                ActType,
                SupportsFloat,
                ObsType,
                Solution[SymType],
            ]
        ] = []
        self.policy_performances: Optional[List[float]] = (
            [] if record_policy_performance else None
        )
        self.policy_gradients: Optional[List[npt.NDArray[np.double]]] = (
            [] if record_policy_gradient else None
        )

    def update(self) -> Optional[str]:
        sample = self.experience.sample()

        # congegrate sample's rollout into a unique least-squares problem
        L_, Phi_, Psi_, dpidtheta_ = [], [], [], []
        mask_phi = [False]
        for L, Phi, Psi, dpidtheta in sample:
            L_.append(L)
            Phi_.append(Phi)
            Psi_.append(Psi)
            mask_phi.extend(chain(repeat(True, L.shape[0]), (False,)))
            dpidtheta_.append(dpidtheta)
        L = np.concatenate(L_)
        Phi = np.concatenate(Phi_)
        Psi = np.concatenate(Psi_)
        dpidtheta = np.concatenate(dpidtheta_)

        # compute CAFA weights v
        Phi_diff = self.discount_factor * Phi[mask_phi[:-1]] - Phi[mask_phi[1:]]
        A_v = Phi[mask_phi[1:]].T @ -Phi_diff
        b_v = Phi[mask_phi[1:]].T @ L
        v = lstsq(A_v, b_v, **self.lstsq_kwargs)[0]

        # compute CAFA weights w
        A_w = Psi.T @ Psi
        b_w = Psi.T @ (L + Phi_diff @ v)
        w = lstsq(A_w, b_w, **self.lstsq_kwargs)[0]

        # compute policy gradient
        dJdtheta = (dpidtheta @ dpidtheta.transpose((0, 2, 1)) @ w).sum(0).reshape(-1)
        if self.policy_gradients is not None:
            self.policy_gradients.append(dJdtheta)

        # perform update
        p = self.learning_rate * dJdtheta
        theta = self._learnable_pars.value  # current values of parameters
        solver = self._update_solver
        if solver is None:
            self._learnable_pars.update_values(theta - p)
            return None

        sol = solver(
            p=np.concatenate((theta, p)),
            lbx=self._learnable_pars.lb,
            ubx=self._learnable_pars.ub,
            x0=theta - p,
        )
        self._learnable_pars.update_values(sol["x"].full().reshape(-1))
        stats = solver.stats()
        return None if stats["success"] else stats["return_status"]

    def train_one_episode(
        self,
        env: GymEnvLike[ObsType, ActType],
        episode: int,
        init_state: ObsType,
        raises: bool = True,
    ) -> float:
        truncated = terminated = False
        timestep = 0
        rewards = 0.0
        state = init_state

        while not (truncated or terminated):
            # compute V(s) (perturbed and not perturbed)
            action, sol = self.state_value(state, deterministic=False)
            action_opt, sol_opt = self.state_value(state, deterministic=True)

            # step the system with the action just computed
            state_new, cost, truncated, terminated, _ = env.step(action)

            # store transition in experience
            if sol.success and sol_opt.success:
                # NOTE: according to Cai et al. [3], the sensitivities should naively be
                # computed with the solution of unpertubed MPC (i.e., sol_opt).
                # According to Gros and Zanon [2], it is hinted that the perturbed
                # solution should be used instead (sol).
                self._rollout.append((state, action - action_opt, cost, state_new, sol))
            else:
                status = f"{sol.status}/{sol_opt.status}"
                self.on_mpc_failure(episode, timestep, status, raises)

            # increase counters
            state = state_new
            rewards += float(cost)
            timestep += 1

            # first, check if current rollout has reached its length, and only then
            # invoke on_env_step (as it might trigger an update)
            if (
                self.rollout_length is not None
                and len(self._rollout) >= self.rollout_length
            ):
                self._consolidate_rollout_into_memory()
            self.on_env_step(env, episode, timestep)

        # consolidate rollout at the end of episode, if no length was specified
        if self.rollout_length is None:
            self._consolidate_rollout_into_memory()
        return rewards

    @staticmethod
    def monomials_state_features(
        ns: int,
        sym_type: Literal["SX", "MX"],
        mindegree: int,
        maxdegree: int,
    ) -> cs.Function:
        """Gets the state feature vector composed of all monomials of degree in the
        given range.

        Parameters
        ----------
        ns : int
            Number of states.
        sym_type : {"MX" or "SX"}
            CasADi symbolic type to be used to build the function.
        mindegree : int
            Minimum degree of monomials (included).
        maxdegree : int
            Maximum degree of monomials (included).

        Returns
        -------
        casadi.Function
            A casadi function of the form `Phi(s)`, where `s` is the state and `Phi` the
            state feature vector.
        """
        s: SymType = getattr(cs, sym_type).sym("x", ns, 1)
        y = cs.vertcat(
            *(
                prod(s**p)
                for k in range(mindegree, maxdegree + 1)
                for p in monomial_powers(ns, k)
            )
        )
        return cs.Function("Phi", [s], [y], ["s"], ["Phi(s)"])

    def _init_dpg_derivatives(
        self,
    ) -> Tuple[cs.Function, cs.Function, npt.NDArray[np.double]]:
        """Internal utility to compute the derivatives w.r.t. the learnable parameters
        and other functions in order to estimate the policy gradient."""
        nlp = self._V.nlp
        y = nlp.primal_dual_vars()
        theta = cs.vertcat(*self._learnable_pars.sym.values())
        u0 = cs.vertcat(*self._V.first_actions.values())
        nlp_ = NlpSensitivity(nlp, target_parameters=theta)
        Kt = nlp_.jacobians["K-p"].T
        Ky = nlp_.jacobians["K-y"].T
        dydu0 = cs.evalf(cs.jacobian(u0, y)).full().T

        # convert derivatives to functions (much faster runtime)
        input = cs.vertcat(nlp.p, nlp.x, nlp.lam_g, nlp.lam_h, nlp.lam_lbx, nlp.lam_ubx)
        dKdtheta = cs.Function("dKdtheta", [input], [Kt])
        dKdy = cs.Function("dKdy", [input], [Ky])
        return dKdtheta, dKdy, dydu0

    def _consolidate_rollout_into_memory(self) -> None:
        """Internal utility to compact the current rollout into a single item in
        memory."""
        # convert to arrays
        S_, E_, L_ = [], [], []
        dKdtheta_, dKdy_ = [], []
        for s, e, cost, _, sol in self._rollout:
            S_.append(s)
            E_.append(e)
            L_.append(cost)
            all_val = sol.all_vals
            dKdtheta_.append(self._dKdtheta(all_val))
            dKdy_.append(self._dKdy(all_val))
        ns = self._V.ns
        na = self._V.na
        s_next_last = self._rollout[-1][3]
        N = len(S_)
        S = np.asarray(S_).reshape(N, ns)
        E = np.asarray(E_).reshape(N, na, 1)  # additional dim required for Psi
        L = np.asarray(L_).reshape(N, 1)
        dKdtheta = np.asarray(dKdtheta_)
        dKdy = np.asarray(dKdy_)

        # compute Phi (to avoid repeating computations, compute only the last Phi(s+))
        Phi = np.concatenate((self._Phi(S.T).full().T, self._Phi(s_next_last).T))

        # compute Psi
        ntheta = dKdtheta.shape[1]
        dydu0 = np.tile(self._dydu0, (N, 1, 1))
        dpidtheta = -dKdtheta @ np.linalg.solve(dKdy, dydu0)
        Psi = (dpidtheta @ E).reshape(N, ntheta)

        # save to memory and clear rollout
        super().store_experience((L, Phi, Psi, dpidtheta))  # type: ignore[arg-type]
        self._rollout.clear()
        if self.policy_performances is not None:
            self.policy_performances.append(L.sum())
