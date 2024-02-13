from collections.abc import Collection, Iterator
from typing import Callable, Generic, Literal, Optional, SupportsFloat, Union

import casadi as cs
import numba as nb
import numpy as np
import numpy.typing as npt
from csnlp.wrappers import Mpc, NlpSensitivity
from gymnasium import Env
from typing_extensions import TypeAlias

from ..core.experience import ExperienceReplay
from ..core.exploration import ExplorationStrategy
from ..core.parameters import LearnableParametersDict
from ..core.update import UpdateStrategy
from ..optim.gradient_based_optimizer import GradientBasedOptimizer
from ..util.math import monomials_basis_function
from .common.agent import ActType, ObsType, SymType
from .common.rl_learning_agent import LrType, RlLearningAgent

ExpType: TypeAlias = tuple[
    npt.NDArray[np.floating],  # rollout's costs
    npt.NDArray[np.floating],  # rollout's state feature vectors Phi(s)
    npt.NDArray[np.floating],  # rollout's Psi(s,a)
    npt.NDArray[np.floating],  # rollout's gradient of policy w.r.t. theta
    npt.NDArray[np.floating],  # rollout's CAFA weight v
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
        optimizer: GradientBasedOptimizer,
        learnable_parameters: LearnableParametersDict[SymType],
        fixed_parameters: Union[
            None, dict[str, npt.ArrayLike], Collection[dict[str, npt.ArrayLike]]
        ] = None,
        exploration: Optional[ExplorationStrategy] = None,
        experience: Union[None, int, ExperienceReplay[ExpType]] = None,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        rollout_length: Optional[int] = None,
        record_policy_performance: bool = False,
        record_policy_gradient: bool = False,
        state_features: Optional[cs.Function] = None,
        linsolver: Literal["csparse", "mldivide"] = "csparse",
        ridge_regression_regularization: float = 1e-6,
        use_last_action_on_fail: bool = False,
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
            transition. If an integer `n` is passed, then a memory with the length `n`
            is created and with sample size `n`.
            In case of LSTD DPG, each memory item is obtain from a single rollout, and
            is a 4-tuple that contains: costs, state vector features (Phi), Psi (a
            temporary value), and the gradient of the policy.
        max_percentage_update : float, optional
            A positive float that specifies the maximum percentage the parameters can be
            changed during each update. For example, `max_percentage_update=0.5` means
            that the parameters can be updated by up to 50% of their current value. By
            default, it is set to `+inf`.
        weight_decay : float, optional
            A positive float that specifies the decay of the learnable parameters in the
            form of an L2 regularization term. By default, it is set to `0.0`, so no
            decay/regularization takes place.
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
        linsolver : "csparse" or "mldivide", optional
            The type of linear solver to be used for solving the linear system derived
            from the KKT conditions and used to estimate the gradient of the policy. By
            default, `"csparse"` is chosen as the KKT matrix is most often sparse.
        ridge_regression_regularization : float, optional
            Ridge regression regularization used during the computations of the LSTD
            weights via least-squares. By default, `1e-6`.
        use_last_action_on_fail : bool, optional
            When `True`, if the MPC solver fails in solving the state value function
            `V(s)`, the last successful action is returned. When `False`, the action
            from the last MPC iteration is returned instead. By default, `False`.
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
            optimizer=optimizer,
            learnable_parameters=learnable_parameters,
            fixed_parameters=fixed_parameters,
            exploration=exploration,
            experience=experience,
            warmstart=warmstart,
            use_last_action_on_fail=use_last_action_on_fail,
            name=name,
        )
        self._sensitivity = self._init_sensitivity(linsolver)
        self._Phi = (
            monomials_basis_function(mpc.ns, 0, 2)
            if state_features is None
            else state_features
        )
        self.ridge_regression_regularization = ridge_regression_regularization
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
        sample = self.experience.sample()
        dJdtheta = _estimate_gradient_update(
            sample, self.discount_factor, self.ridge_regression_regularization
        )
        if self.policy_gradients is not None:
            self.policy_gradients.append(dJdtheta)
        return self.optimizer.update(dJdtheta)

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
                exploration = np.asarray((action - action_opt).elements())
                sol_vals = np.asarray(sol_opt.all_vals.elements())
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

    def _init_sensitivity(self, linsolver: str) -> Callable[[cs.DM, int], np.ndarray]:
        """Internal utility to compute the derivatives w.r.t. the learnable parameters
        and other functions in order to estimate the policy gradient."""
        assert self.optimizer._order == 1, "Expected 1st-order optimizer."
        nlp = self._V.nlp
        y = nlp.primal_dual
        theta = cs.vvcat(self._learnable_pars.sym.values())
        u0 = cs.vcat(self._V.first_actions.values())
        x_lam_p = cs.vertcat(nlp.primal_dual, nlp.p)

        # compute first bunch of derivatives
        nlp_ = NlpSensitivity(nlp, theta)
        Kt = nlp_.jacobians["K-p"]
        Ky = nlp_.jacobians["K-y"]
        dydu0 = cs.jacobian_sparsity(u0, y).T

        # instantiate linear solver (must be MX, so SX has to be converted)
        if nlp.sym_type is cs.SX:
            x_lam_p, x_lam_p_sx = cs.MX.sym("in", *x_lam_p.shape), x_lam_p
            Kt, Ky = cs.Function("sx2mx", (x_lam_p_sx,), (Kt, Ky))(x_lam_p)
        solver = (
            cs.solve  # cs.mldivide
            if linsolver == "mldivide"
            else cs.Linsol("linsolver", linsolver, Ky.sparsity()).solve
        )

        # compute sensitivity and convert to function (faster runtime)
        dpidtheta = -solver(Ky, Kt).T @ dydu0
        sensitivity = cs.Function(
            "dpidtheta",
            (x_lam_p,),
            (dpidtheta,),
            ("x_lam_p",),
            ("dpidtheta",),
            {"cse": True},
        )
        ntheta, na = dpidtheta.shape

        # wrap to conveniently return arrays. Casadi does not support tensors with
        # >2 dims, so dpidtheta gets squished in the 3rd dim and needs reshaping
        def func(sol_values: cs.DM, N: int) -> np.ndarray:
            return (
                np.ascontiguousarray(sensitivity(sol_values.T).elements())
                .reshape(ntheta, na, N, order="F")
                .transpose((2, 0, 1))
            )

        return func

    def _consolidate_rollout_into_memory(self) -> None:
        """Internal utility to compact current rollout into a single item in memory."""
        # convert rollout to arrays and clear it
        N, S, E, L, vals = _consolidate_rollout(self._rollout, self._V.ns, self._V.na)
        self._rollout.clear()

        # compute Phi, dpidtheta, Psi, and CAFA weight v
        Phi = np.ascontiguousarray(self._Phi(S.T).elements()).reshape(N + 1, -1)
        dpidtheta = self._sensitivity(vals, N)
        Psi = (dpidtheta @ E).reshape(N, dpidtheta.shape[1])
        R = self.ridge_regression_regularization
        v = _compute_cafa_weight_v(Phi, L, self.discount_factor, R)

        # save to experience
        self.store_experience((L, Phi, Psi, dpidtheta, v))
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
    S = np.empty((N + 1, ns))
    E = np.empty((N, na, 1))  # additional dim required for Psi
    L = np.empty(N)
    sol_vals = np.empty((N, rollout[0][-1].size))
    for i in nb.prange(N):
        s, e, cost, _, sol_val = rollout[i]
        S[i] = s.reshape(-1)
        E[i] = e
        L[i] = cost
        sol_vals[i] = sol_val
    S[-1] = rollout[-1][3].reshape(-1)
    return N, S, E, L, sol_vals


@nb.njit(cache=True, nogil=True)
def _compute_cafa_weight_v(
    Phi_all: np.ndarray, L: np.ndarray, discount_factor: float, regularization: float
) -> np.ndarray:
    """Compute the CAFA weight v via ridge regression."""
    # solve for v via ridge regression: -phi'(gamma phi+ - phi) v = phi'L
    Phi = Phi_all[:-1]
    Phi_next = Phi_all[1:]
    Phi_diff = discount_factor * Phi_next - Phi
    M = Phi_diff.T @ Phi @ Phi.T
    R = regularization * np.eye(M.shape[0])
    return np.linalg.solve(M @ Phi_diff + R, -M @ L)


@nb.njit(cache=True, nogil=True)
def _compute_cafa_weight_w(
    Phi_all: np.ndarray,
    Psi: np.ndarray,
    L: np.ndarray,
    v: np.ndarray,
    discount_factor: float,
    regularization: float,
) -> np.ndarray:
    """Compute the CAFA weight w via ridge regression."""
    # solve for w via ridge regression: psi' psi w = psi' (L + (gamma phi+ - phi) v)
    Phi = Phi_all[:-1]
    Phi_next = Phi_all[1:]
    Phi_diff = discount_factor * Phi_next - Phi
    A = Psi.T @ Psi
    b = Psi.T @ (L + Phi_diff @ v)
    R = regularization * np.eye(A.shape[0])
    return np.linalg.solve(A.T @ A + R, A.T @ b)


def _estimate_gradient_update(
    sample: Iterator[ExpType], discount_factor: float, regularization: float
) -> np.ndarray:
    """Internal utility to estimate the gradient of the policy."""
    # compute average v and w
    sample_ = list(sample)  # load whole iterator into a list
    v = np.mean([o[4] for o in sample_], 0)
    w_list = [
        _compute_cafa_weight_w(Phi, Psi, L, v, discount_factor, regularization)
        for L, Phi, Psi, _, _ in sample_
    ]
    w = np.mean(w_list, 0)

    # compute policy gradient estimate
    dJdtheta_list = [(o[3] @ o[3].transpose((0, 2, 1))).sum(0) @ w for o in sample_]
    return np.mean(dJdtheta_list, 0)
