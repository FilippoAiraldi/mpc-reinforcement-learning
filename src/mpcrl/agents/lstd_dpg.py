from itertools import repeat
from typing import (
    Any,
    Callable,
    Collection,
    Generic,
    Iterator,
    Literal,
    Optional,
    SupportsFloat,
    Union,
)

import casadi as cs
import numba as nb
import numpy as np
import numpy.typing as npt
from csnlp.wrappers import Mpc, NlpSensitivity
from gymnasium import Env
from scipy.linalg import cho_solve, lstsq
from typing_extensions import TypeAlias

from mpcrl.agents.agent import ActType, ObsType, SymType
from mpcrl.agents.rl_learning_agent import LrType, RlLearningAgent
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.exploration import ExplorationStrategy
from mpcrl.core.learning_rate import LearningRate
from mpcrl.core.parameters import LearnableParametersDict
from mpcrl.core.schedulers import Scheduler
from mpcrl.core.update import UpdateStrategy
from mpcrl.util.math import cholesky_added_multiple_identities, monomials_basis_function

ExpType: TypeAlias = tuple[
    npt.NDArray[np.floating],  # rollout's costs
    npt.NDArray[np.floating],  # rollout's state feature vectors Phi(s)
    npt.NDArray[np.floating],  # rollout's Psi(s,a)
    npt.NDArray[np.floating],  # rollout's gradient of policy
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
        rollout_length: Optional[int] = None,
        record_policy_performance: bool = False,
        record_policy_gradient: bool = False,
        state_features: Optional[cs.Function] = None,
        lstsq_cond: Optional[float] = 1e-7,
        linsolver: Literal["csparse", "qr", "mldivide"] = "mldivide",
        hessian_type: Literal["none", "natural"] = "none",
        cho_maxiter: int = 1000,
        cho_solve_kwargs: Optional[dict[str, Any]] = None,
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
        max_percentage_update : float, optional
            A positive float that specifies the maximum percentage the parameters can be
            changed during each update. For example, `max_percentage_update=0.5` means
            that the parameters can be updated by up to 50% of their current value. By
            default, it is set to `+inf`.
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
            If `True`, the performance of each rollout is stored in the field
            `policy_performances`, which otherwise is `None`. By default, does not
            record them.
        record_policy_gradient: bool, optional
            If `True`, the (estimated) policy gradient of each update is stored in the
            field `policy_gradients`, which otherwise is `None`. By default, does not
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
            with degrees <= 2 (see `mpcrl.util.math.monomials_basis_function`).
        lstsq_cond : float, optional
            Conditional number to be passed to `scipy.linalg.lstsq`. By default, `None`.
        linsolver : "csparse" or "qr" or "mldivide", optional
            The type of linear solver to be used for solving the linear system derived
            from the KKT conditions and used to estimate the gradient of the policy. By
            default, `"mldivide"` is chosen.
        hessian_type : 'none' or 'natural', optional
            The type of hessian to use in this second-order algorithm. If `"none"`, no
            hessian is used (first-order). If `"natural"`, the hessian is approximated
            according to natural policy gradients.
        cho_maxiter : int, optional
            Maximum number of iterations in the Cholesky's factorization with additive
            multiples of the identity to ensure positive definiteness of the hessian. By
            default, `1000`. Only used if `hessian_type!='none'`.
        cho_solve_kwargs : kwargs for scipy.linalg.cho_solve, optional
            The optional kwargs to be passed to `scipy.linalg.cho_solve`. If `None`, it
            is equivalent to `cho_solve_kwargs = {"check_finite": False }`. Only used if
            `hessian_type!="none"`.
        name : str, optional
            Name of the agent. If `None`, one is automatically created from a counter of
            the class' instancies.
        """
        # change default update hook to 'on_episode_end'
        if not isinstance(update_strategy, UpdateStrategy):
            update_strategy = UpdateStrategy(update_strategy, "on_episode_end")
        super().__init__(
            mpc,
            update_strategy,
            discount_factor,
            learning_rate,  # type: ignore[arg-type]
            learnable_parameters,
            fixed_parameters,
            exploration,
            experience,
            max_percentage_update,
            warmstart,
            name,
        )
        # initialize derivatives, state feature vector and hessian approximation
        self._sensitivity = self._init_sensitivity(linsolver)
        self._Phi = (
            monomials_basis_function(mpc.ns, 0, 2)
            if state_features is None
            else state_features
        )
        self.hessian_type = hessian_type
        self.cho_maxiter = cho_maxiter
        if cho_solve_kwargs is None:
            cho_solve_kwargs = {"check_finite": False}
        self.cho_solve_kwargs = cho_solve_kwargs
        # initialize others
        self.lstsq_cond = lstsq_cond
        self.rollout_length = rollout_length or float("+inf")
        self._rollout: list[
            tuple[
                ObsType,
                ActType,
                SupportsFloat,
                ObsType,
                npt.NDArray[np.floating],
            ]
        ] = nb.typed.List()
        self.policy_performances: Optional[list[float]] = (
            [] if record_policy_performance else None
        )
        self.policy_gradients: Optional[list[npt.NDArray[np.floating]]] = (
            [] if record_policy_gradient else None
        )

    def update(self) -> Optional[str]:
        # sample and congregate sampled rollouts into a unique least-squares problem
        sample = self.experience.sample()
        L, Phi, Psi, dpidtheta, mask_phi = _congregate_rollouts_sample(sample)

        # compute CAFA weights v
        Phi_diff = self.discount_factor * Phi[mask_phi[:-1]] - Phi[mask_phi[1:]]
        Av = Phi[mask_phi[1:]].T @ -Phi_diff
        bv = Phi[mask_phi[1:]].T @ L
        v = lstsq(Av, bv, self.lstsq_cond, lapack_driver="gelsy", check_finite=False)[0]

        # compute CAFA weights w
        Aw = Psi.T @ Psi
        bw = Psi.T @ (L + Phi_diff @ v)
        w = lstsq(Aw, bw, self.lstsq_cond, lapack_driver="gelsy", check_finite=False)[0]

        # compute policy gradient
        dJdtheta = (dpidtheta @ dpidtheta.transpose((0, 2, 1))).sum(0) @ w
        if self.hessian_type == "natural":
            Hessian = (dpidtheta @ dpidtheta.transpose((0, 2, 1))).sum(0)
            R = cholesky_added_multiple_identities(Hessian, maxiter=self.cho_maxiter)
            step = cho_solve((R, True), dJdtheta, **self.cho_solve_kwargs)
        else:
            step = dJdtheta

        # perform update
        if self.policy_gradients is not None:
            self.policy_gradients.append(step)
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

        while not (truncated or terminated):
            # compute V(s) (perturbed and not perturbed)
            action, sol = self.state_value(state, False)
            action_opt, sol_opt = self.state_value(state, True)

            # step the system with the action just computed
            state_new, cost, truncated, terminated, _ = env.step(action)
            self.on_env_step(env, episode, timestep)

            # store transition in experience
            if sol.success and sol_opt.success:
                # NOTE: according to Cai et al. [3], the sensitivities should naively be
                # computed with the solution of unpertubed MPC (i.e., sol_opt).
                # According to Gros and Zanon [2], it is hinted that the perturbed
                # solution should be used instead (sol).
                exploration = (action - action_opt).full()
                sol_vals = sol.all_vals.full()
                self._rollout.append((state, exploration, cost, state_new, sol_vals))
            else:
                status = f"{sol.status}/{sol_opt.status}"
                self.on_mpc_failure(episode, timestep, status, raises)

            # increase counters
            state = state_new
            rewards += float(cost)
            timestep += 1

            # first, check if current rollout has reached its length, and only then
            # invoke on_timestep_end (as it might trigger an update)
            if len(self._rollout) >= self.rollout_length:
                self._consolidate_rollout_into_memory()
            self.on_timestep_end(env, episode, timestep)

        # consolidate rollout at the end of episode, if no length was specified
        if self.rollout_length == float("+inf"):
            self._consolidate_rollout_into_memory()
        return rewards

    def _init_sensitivity(
        self, linsolver_type: str
    ) -> Callable[[cs.DM, int], np.ndarray]:
        """Internal utility to compute the derivatives w.r.t. the learnable parameters
        and other functions in order to estimate the policy gradient."""
        nlp = self._V.nlp
        y = nlp.primal_dual
        theta = cs.vvcat(self._learnable_pars.sym.values())
        u0 = cs.vcat(self._V.first_actions.values())
        x_lam_p = cs.vertcat(nlp.primal_dual, nlp.p)

        # compute first bunch of derivatives
        nlp_ = NlpSensitivity(nlp, theta)
        Kt = nlp_.jacobians["K-p"].T
        Ky = nlp_.jacobians["K-y"].T
        dydu0 = cs.evalf(cs.jacobian(u0, y)).T

        # instantiate linear solver (must be MX)
        if nlp.sym_type is cs.SX:
            # convert SX to MX (so that we can use the linsolver)
            x_lam_p, x_lam_p_sx = cs.MX.sym("in", *x_lam_p.shape), x_lam_p
            Kt, Ky = cs.Function("sx2mx", (x_lam_p_sx,), (Kt, Ky))(x_lam_p)
        linsolver = (
            cs.solve  # cs.mldivide
            if linsolver_type == "mldivide"
            else cs.Linsol("linsolver", linsolver_type, Ky.sparsity()).solve
        )

        # compute sensitivity and convert to function (faster runtime)
        dpidtheta = -Kt @ linsolver(Ky, dydu0)
        sensitivity = cs.Function(
            "pi_sensitivity", (x_lam_p,), (dpidtheta,), ("x_lam_p",), ("dpidtheta",)
        )
        ntheta, na = dpidtheta.shape

        def func(sol_values: cs.DM, N: int) -> np.ndarray:
            # wrap to conveniently return arrays. Casadi does not support tensors with
            # >2 dims, so dpidtheta gets squished in the 3rd dim and needs reshaping
            return (
                sensitivity(sol_values.T)
                .full()
                .reshape(ntheta, na, N, order="F")
                .transpose((2, 0, 1))
            )

        return func

    def _consolidate_rollout_into_memory(self) -> None:
        """Internal utility to compact current rollout into a single item in memory."""
        # convert to arrays
        N, S, E, L, vals = _consolidate_rollout(self._rollout, self._V.ns, self._V.na)

        # compute Phi (to avoid repeating computations, compute only the last Phi(s+))
        s_next_last = self._rollout[-1][3].T  # type: ignore[union-attr]
        Phi = self._Phi(np.concatenate((S, s_next_last)).T).full().T

        # compute dpidtheta and Psi (casadi does not support tensors with more than 2
        # dims, so dpidtheta gets squished in the third dim and needs to be reshaped)
        dpidtheta = self._sensitivity(vals, N)
        Psi = (dpidtheta @ E).reshape(N, dpidtheta.shape[1])

        # save to memory and clear rollout
        self.store_experience((L, Phi, Psi, dpidtheta))
        self._rollout.clear()
        if self.policy_performances is not None:
            self.policy_performances.append(L.sum())


@nb.njit(cache=True, nogil=True, parallel=True)
def _consolidate_rollout(
    rollout: list[tuple[ObsType, ActType, float, ObsType, np.ndarray]],
    ns: int,
    na: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Internal utility to convert a rollout list to arrays."""
    N = len(rollout)
    S = np.empty((N, ns))
    E = np.empty((N, na, 1))  # additional dim required for Psi
    L = np.empty(N)
    sol_vals = np.empty((N, rollout[0][-1].size))
    for i in nb.prange(N):
        s, e, cost, _, sol_val = rollout[i]
        S[i] = s.reshape(-1)
        E[i] = e
        L[i] = cost
        sol_vals[i] = sol_val.reshape(-1)
    return N, S, E, L, sol_vals


def _congregate_rollouts_sample(
    sample: Iterator[ExpType],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[bool]]:
    """Internal utility to congregate a sample of rollouts into single arrays."""
    # jit does not seem to provide any speedup here
    L_, Phi_, Psi_, dpidtheta_, mask_phi = [], [], [], [], [False]
    for L, Phi, Psi, dpidtheta in sample:
        L_.append(L)
        Phi_.append(Phi)
        Psi_.append(Psi)
        dpidtheta_.append(dpidtheta)
        mask_phi.extend(repeat(True, L.shape[0]))
        mask_phi.append(False)
    L = np.concatenate(L_)
    Phi = np.concatenate(Phi_)
    Psi = np.concatenate(Psi_)
    dpidtheta = np.concatenate(dpidtheta_)
    return L, Phi, Psi, dpidtheta, mask_phi
