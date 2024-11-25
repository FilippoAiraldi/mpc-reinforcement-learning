import sys
from collections.abc import Collection, Iterator
from typing import Callable, Generic, Literal, Optional, SupportsFloat, Union

import casadi as cs
import numba as nb
import numpy as np
import numpy.typing as npt
from csnlp.wrappers import Mpc, NlpSensitivity
from gymnasium import Env

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from ..core.experience import ExperienceReplay
from ..core.exploration import ExplorationStrategy, NoExploration
from ..core.parameters import LearnableParametersDict
from ..core.update import UpdateStrategy
from ..core.warmstart import WarmStartStrategy
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
    r"""Least-Squares Temporal Difference (LSTD) Deterministic Policy Gradient (DPG)
    agent, as introduced in :cite:`gros_reinforcement_2021` as its stochastic
    counterpart, and refined in :cite:`gros_towards_2019`. An application can be found
    in :cite:`cai_mpcbased_2021`.

    The DPG agent uses an MPC controller as policy provider and function approximation,
    and adjusts its parametrization according to the temporal-difference error, with the
    goal of improving the policy, in a direct fashion by estimating the gradient of the
    policy and descending in its direction.

    Parameters
    ----------
    mpc : :class:`csnlp.wrappers.Mpc`
        The MPC controller used as policy provider by this agent. The instance is
        modified in place to create the approximations of the state function
        :math:`V_\theta(s)` and action value function :math:`Q_\theta(s,a)`, so it is
        recommended not to modify it further after initialization of the agent.
        Moreover, some parameter and constraint names will need to be created, so an
        error is thrown if these names are already in use in the mpc.
    update_strategy : UpdateStrategy or int
        The strategy used to decide which frequency to update the mpc parameters with.
        If an ``int`` is passed, then the default strategy that updates every ``n``
        episodes is used (where ``n`` is the argument passed); otherwise, an instance
        of :class:`core.update.UpdateStrategy` can be passed to specify the desired
        strategy in more details.
    discount_factor : float
        In RL, the factor that discounts future rewards in favor of immediate rewards.
        Usually denoted as :math:`\gamma`. It should satisfy :math:`\gamma \in (0, 1]`.
    optimizer : GradientBasedOptimizer
        A gradient-based optimizer (e.g., :class:`optim.GradientDescent`) to
        compute the updates of the learnable parameters, based on the current
        gradient-based RL algorithm.
    learnable_parameters : :class:`core.parameters.LearnableParametersDict`
        A special dict containing the learnable parameters of the MPC (usually referred
        to as :math:`\theta`), together with their bounds and values. This dict is
        complementary to :attr:`fixed_parameters`, which contains the MPC parameters
        that are not learnt by the agent.
    exploration : :class:`core.exploration.ExplorationStrategy`
        Exploration strategy for inducing exploration in the online MPC policy. It is
        mandatory for DPG agents to have exploration.
    fixed_parameters : dict of (str, array_like) or collection of, optional
        A dict (or collection of dict, in case of the ``mpc`` wrapping an underlying
        :class:`csnlp.multistart.MultistartNlp` instance) whose keys are the names of
        the MPC parameters and the values are their corresponding values. Use this to
        specify fixed parameters, that is, non-learnable. If ``None``, then no fixed
        parameter is assumed.
    experience : int or ExperienceReplay, optional
        The container for experience replay memory. If ``None`` is passed, then a memory
        with unitary length is created, i.e., it keeps only the latest memory
        transition. If an integer ``n`` is passed, then a memory with the length ``n``
        is created and with sample size ``n``. Otherwise, pass an instance of
        :class:`core.experience.ExperienceReplay` to specify the requirements in more
        details.
    warmstart : "last" or "last-successful" or WarmStartStrategy, optional
        The warmstart strategy for the MPC's NLP. If ``"last-successful"``, the last
        successful solution is used to warmstart the solver for the next iteration.
        If ``"last"``, the last solution is used, regardless of success or failure.
        Furthermore, an instance of :class:`core.warmstart.WarmStartStrategy` can
        be passed to specify a strategy for generating multiple warmstart points for the
        MPC's NLP instance. This is useful to generate multiple initial conditions for
        highly non-convex, nonlinear problems. This feature can only be used with an
        MPC that has an underlying multistart NLP problem (see :mod:`csnlp.multistart`).
    hessian_type : {"none", "natural"}, optional
        The type of hessian to use in this (potentially) second-order algorithm.
        If ``"none"``, no second order information is used. If ``"natural"``, the Fisher
        information matrix is used to perform a natural policy gradient update. This
        option must be in accordance with the choice of ``optimizer``, that is, if the
        optimizer does not use second order information, then this option must be set to
        ``none``.
    rollout_length : int, optional
        Number of steps of each closed-loop simulation, which defines a complete
        trajectory of the states (i.e., a rollout), and is saved in the experience as a
        single item (since LSTD DPG needs to draw samples of trajectories). In case the
        env is episodic, it can be ``-1``, in which case the rollout length coincides
        with the episode's length. In case the env is not episodic, i.e., it never
        terminates, a length ``>0`` must be given in order to know when to save the
        current trajectory as an atomic item in memory.
    record_policy_performance: bool, optional
        If ``True``, the performance of each rollout is stored in the field
        :attr:`policy_performances`, which otherwise is ``None``. By default, does not
        record them.
    record_policy_gradient: bool, optional
        If ``True``, the (estimated) policy gradient of each update is stored in the
        field :attr:`policy_gradients`, which otherwise is `None`. By default, does not
        record them.
    state_features : casadi.Function, optional
        The state feature vector to be used in the linear approximation of the
        value function, which takes the form of

        .. math:: V_v(s) = \Phi(s)^\top v,

        where :math:`s` is the state, :math:`v` are the weights, and :math:`\Phi(s)` is
        the state feature vector. This function is assumed to have one input and one
        output. By default, if not provided, it is designed as all monomials of the
        state with degrees ``<= 2`` (see :func:`util.math.monomials_basis_function`).
    linsolver : "csparse" or "mldivide", optional
        The type of linear solver to be used for solving the linear system derived
        from the KKT conditions and used to estimate the gradient of the policy. By
        default, ``"csparse"`` is chosen as the KKT matrix is most often sparse.
    ridge_regression_regularization : float, optional
        Ridge regression regularization used during the computations of the LSTD
        weights via least-squares. By default, ``1e-6``.
    use_last_action_on_fail : bool, optional
        In case the MPC solver fails
         - if ``False``, the action from the last solver's iteration is returned anyway
           (though suboptimal)
         - if ``True``, the action from the last successful call to the MPC is returned
           instead (if the MPC has been solved at least once successfully).

        By default, ``False``.
    name : str, optional
        Name of the agent. If ``None``, one is automatically created from a counter of
        the class' instancies.

    Raises
    ------
    ValueError
        If the exploration strategy is ``None`` or an instance of ``NoExploration``, as
        DPG requires exploration.
    """

    def __init__(
        self,
        mpc: Mpc[SymType],
        update_strategy: Union[int, UpdateStrategy],
        discount_factor: float,
        optimizer: GradientBasedOptimizer,
        learnable_parameters: LearnableParametersDict[SymType],
        exploration: ExplorationStrategy,
        fixed_parameters: Union[
            None, dict[str, npt.ArrayLike], Collection[dict[str, npt.ArrayLike]]
        ] = None,
        experience: Union[None, int, ExperienceReplay[ExpType]] = None,
        warmstart: Union[
            Literal["last", "last-successful"], WarmStartStrategy
        ] = "last-successful",
        hessian_type: Literal["none", "natural"] = "none",
        rollout_length: int = -1,
        record_policy_performance: bool = False,
        record_policy_gradient: bool = False,
        state_features: Optional[cs.Function] = None,
        linsolver: Literal["csparse", "mldivide"] = "csparse",
        ridge_regression_regularization: float = 1e-6,
        use_last_action_on_fail: bool = False,
        name: Optional[str] = None,
    ) -> None:
        if exploration is None or isinstance(exploration, NoExploration):
            raise ValueError("DPG requires exploration, but none was provided.")
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
        self.hessian_type = hessian_type
        self._sensitivity = self._init_sensitivity(linsolver)
        self._Phi = (
            monomials_basis_function(mpc.ns, 0, 2)
            if state_features is None
            else state_features
        )
        self.regularization = ridge_regression_regularization
        self.rollout_length = rollout_length
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
        compute_fisher_mat = self.hessian_type == "natural"
        dJdtheta, fisher_hessian = _estimate_gradient_update(
            sample, self.discount_factor, self.regularization, compute_fisher_mat
        )
        if self.policy_gradients is not None:
            self.policy_gradients.append(dJdtheta)
        return self.optimizer.update(dJdtheta, fisher_hessian)

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
        rollout_length = self.rollout_length
        action_space = getattr(env, "action_space", None)
        gradient_based_exploration = self.exploration.mode == "gradient-based"
        action_keys = self.V.actions.keys()

        while not (truncated or terminated):
            # compute V(s)
            action, sol = self.state_value(state, False, action_space=action_space)
            if gradient_based_exploration:
                # NOTE: if the exploration affects the gradient, unfortunately we have
                # to solve again the NLP (but deterministically this time)
                action_opt, sol_opt = self.state_value(
                    state, True, action_space=action_space
                )
            else:
                # otherwise, just retrieve manually the unperturbed optimal action
                action_opt = cs.vertcat(*(sol.vals[u][:, 0] for u in action_keys))
                sol_opt = sol

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
                sol_vals = np.asarray(sol_opt.x_and_lam_and_p.elements())
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
            if rollout_length > 0 and len(self._rollout) >= rollout_length:
                self._consolidate_rollout_into_memory()
            self.on_timestep_end(env, episode, timestep)

        # consolidate rollout at the end of episode, if no length was specified
        if rollout_length <= 0:
            self._consolidate_rollout_into_memory()
        return rewards

    def _init_sensitivity(self, linsolver: str) -> Callable[[cs.DM, int], np.ndarray]:
        """Internal utility to compute the derivatives w.r.t. the learnable parameters
        and other functions in order to estimate the policy gradient."""
        assert (self.hessian_type == "none" and self.optimizer._order == 1) or (
            self.hessian_type == "natural" and self.optimizer._order == 2
        ), "expected 1st-order (2nd-order) optimizer with `none` (`natural`) hessian"
        nlp = self._V.nlp
        y = nlp.primal_dual
        theta = cs.vvcat(self._learnable_pars.sym.values())
        u0 = cs.vcat(self._V.first_actions.values())
        x_lam_p = cs.vertcat(nlp.primal_dual, nlp.p)

        # compute first bunch of derivatives
        snlp = NlpSensitivity(nlp, theta)
        Kt = snlp.jacobian("K-p")
        Ky = snlp.jacobian("K-y")
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
        v = _compute_cafa_weight_v(Phi, L, self.discount_factor, self.regularization)

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
        E[i, :, 0] = e
        L[i] = cost
        sol_vals[i] = sol_val
    S[-1] = rollout[-1][3].reshape(-1)
    return N, S, E, L, sol_vals


@nb.njit(cache=True, nogil=True)
def _compute_cafa_weight_v(
    Phi_all: np.ndarray, L: np.ndarray, discount_factor: float, regularization: float
) -> np.ndarray:
    """Compute the CAFA weight ``v`` via ridge regression."""
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
    """Compute the CAFA weight ``w`` via ridge regression."""
    # solve for w via ridge regression: psi' psi w = psi' (L + (gamma phi+ - phi) v)
    Phi = Phi_all[:-1]
    Phi_next = Phi_all[1:]
    Phi_diff = discount_factor * Phi_next - Phi
    A = Psi.T @ Psi
    b = Psi.T @ (L + Phi_diff @ v)
    R = regularization * np.eye(A.shape[0])
    return np.linalg.solve(A.T @ A + R, A.T @ b)


def _estimate_gradient_update(
    sample: Iterator[ExpType],
    discount_factor: float,
    regularization: float,
    return_fisher_hessian: bool,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Internal utility to estimate the gradient of the policy and possibly the Fisher
    information matrix as well."""
    # compute average v and w
    sample_ = list(sample)  # load whole iterator into a list
    v = np.mean([o[4] for o in sample_], 0)
    w_ = [
        _compute_cafa_weight_w(Phi, Psi, L, v, discount_factor, regularization)
        for L, Phi, Psi, _, _ in sample_
    ]
    w = np.mean(w_, 0)

    if return_fisher_hessian:
        # compute both policy gradient and Fisher information matrix
        fisher_hess_ = []
        dJdtheta_ = []
        for _, _, _, dpidtheta, _ in sample_:
            F = (dpidtheta @ dpidtheta.transpose((0, 2, 1))).sum(0)
            fisher_hess_.append(F)
            dJdtheta_.append(F @ w)
        return np.mean(dJdtheta_, 0), np.mean(fisher_hess_, 0)

    # compute only policy gradient estimate
    dJdtheta_ = [(o[3] @ o[3].transpose((0, 2, 1))).sum(0) @ w for o in sample_]
    return np.mean(dJdtheta_, 0), None
