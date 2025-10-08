import sys
from collections.abc import Collection
from typing import Callable, Generic, Literal, Optional, SupportsFloat, Union

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Solution
from csnlp.wrappers import Mpc, NlpSensitivity
from gymnasium import Env

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from ..core.experience import ExperienceReplay
from ..core.exploration import ExplorationStrategy
from ..core.parameters import LearnableParametersDict
from ..core.update import UpdateStrategy
from ..core.warmstart import WarmStartStrategy
from ..optim.gradient_based_optimizer import GradientBasedOptimizer
from .common.agent import ActType, ObsType, SymType
from .common.rl_learning_agent import LrType, RlLearningAgent

# the usual buffer of SARS tuples with terminated flags; see here why these are needed:
# https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
ExpType: TypeAlias = tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    SupportsFloat,
    npt.NDArray[np.floating],
    bool,
]


class LstdQLearningAgent(
    RlLearningAgent[SymType, ExpType, LrType], Generic[SymType, LrType]
):
    r"""Second-order Least-Squares Temporal Difference (LSTD) Q-learning agent, as first
    proposed in a simpler format in :cite:`gros_datadriven_2020`, and then in
    :cite:`esfahani_approximate_2021`.

    The Q-learning agent uses an MPC controller as policy provider and function
    approximation, and adjusts its parametrization according to the temporal-difference
    error, with the goal of improving the policy, in an indirect fashion by learning the
    action value function.

    Parameters
    ----------
    mpc : :class:`csnlp.wrappers.Mpc` or tuple of :class:`csnlp.wrappers.Mpc`
        The MPC controller used as policy provider by this agent. If a tuple, the first
        entry is used to create the approximation of the state function
        :math:`V_\theta(s)` and the second for that of  :math:`Q_\theta(s,a)`.
        Otherwise, the instance is modified in place to create both approximations,
        so it is recommended not to modify it further after initialization of the
        agent. Moreover, some parameter and constraint names will need to be created,
        so an error is thrown if these names are already in use in the mpc.
    update_strategy : UpdateStrategy or int
        The strategy used to decide which frequency to update the mpc parameters with.
        If an ``int`` is passed, then the default strategy that updates every ``n``
        env's steps is used (where ``n`` is the argument passed); otherwise, an instance
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
    fixed_parameters : dict of (str, array_like) or collection of, optional
        A dict (or collection of dict, in case of the ``mpc`` wrapping an underlying
        :class:`csnlp.multistart.MultistartNlp` instance) whose keys are the names of
        the MPC parameters and the values are their corresponding values. Use this to
        specify fixed parameters, that is, non-learnable. If ``None``, then no fixed
        parameter is assumed.
    exploration : :class:`core.exploration.ExplorationStrategy`, optional
        Exploration strategy for inducing exploration in the online MPC policy. By
        default ``None``, in which case :class:`core.exploration.NoExploration` is used.
    experience : int or ExperienceReplay, optional
        The container for experience replay memory. If ``None`` is passed, then a memory
        with unitary length is created, i.e., it keeps only the latest memory
        transition. If an integer ``n`` is passed, then a memory with the length ``n``
        is created and with sample size ``n``. Otherwise, pass an instance of
        :class:`core.experience.ExperienceReplay` to specify the requirements in more
        details.
    gradient_steps : int, optional
        When an update is due, how many gradient steps to perform. In each step, a new
        batch of transitions (whose sample size is controlled by the ``experience``
        argument) is sampled and a gradient step taken. Set to ``-1`` to take as
        many steps as the current length of experience replay buffer. By default, it is
        set to ``1``.
    warmstart : "last" or "last-successful" or WarmStartStrategy, optional
        The warmstart strategy for the MPC's NLP. If ``"last-successful"``, the last
        successful solution is used to warmstart the solver for the next iteration. If
        ``"last"``, the last solution is used, regardless of success or failure.
        Furthermore, an instance of :class:`core.warmstart.WarmStartStrategy` can
        be passed to specify a strategy for generating multiple warmstart points for the
        MPC's NLP instance. This is useful to generate multiple initial conditions for
        highly non-convex, nonlinear problems. This feature can only be used with an
        MPC that has an underlying multistart NLP problem (see :mod:`csnlp.multistart`).
    hessian_type : {"none", "approx", "full"}, optional
        The type of hessian to use in this (potentially) second-order algorithm.
        If ``"none"``, no second order information is used. If ``"approx"``, an easier
        approximation of it is used; otherwise, the full hessian is computed, but
        this is usually much more expensive. This option must be in accordance with the
        choice of ``optimizer``, that is, if the optimizer does not use second order
        information, then this option must be set to ``none``.
    fail_on_td_target : bool, optional
        If ``True``, failures in computing :math:`V_\theta(s_+)` for the TD target will
        raise an exception; otherwise, the TD target is still considered valid. By
        default, ``True``.
    record_td_errors: bool, optional
        If ``True``, the TD errors are recorded in the field :attr:`td_errors`, which
        otherwise is ``None``. By default, does not record them.
    use_last_action_on_fail : bool, optional
        In case the MPC solver fails
         - if ``False``, the action from the last solver's iteration is returned anyway
           (though suboptimal)
         - if ``True``, the action from the last successful call to the MPC is returned
           instead (if the MPC has been solved at least once successfully).

        By default, ``False``.
    remove_bounds_on_initial_action : bool, optional
        When ``True``, the upper and lower bounds on the initial action are removed in
        the action-value function approximator :math:`Q_\theta(s,a)` since the first
        action is constrained to be equal to the provided action :math:`a`. This is
        useful to avoid issues in the LICQ of the NLP. However, it can lead to numerical
        problems. By default, ``False``.
    name : str, optional
        Name of the agent. If ``None``, one is automatically created from a counter of
        the class' instancies.
    """

    def __init__(
        self,
        mpc: Mpc[SymType] | tuple[Mpc[SymType], Mpc[SymType]],
        update_strategy: Union[int, UpdateStrategy],
        discount_factor: float,
        optimizer: GradientBasedOptimizer,
        learnable_parameters: LearnableParametersDict,
        fixed_parameters: Union[
            None, dict[str, npt.ArrayLike], Collection[dict[str, npt.ArrayLike]]
        ] = None,
        exploration: Optional[ExplorationStrategy] = None,
        experience: Union[None, int, ExperienceReplay[ExpType]] = None,
        gradient_steps: int = 1,
        warmstart: Union[
            Literal["last", "last-successful"], WarmStartStrategy
        ] = "last-successful",
        hessian_type: Literal["none", "approx", "full"] = "approx",
        fail_on_td_target: bool = True,
        record_td_errors: bool = False,
        use_last_action_on_fail: bool = False,
        remove_bounds_on_initial_action: bool = False,
        name: Optional[str] = None,
    ) -> None:
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
        self.gradient_steps = gradient_steps
        self.hessian_type = hessian_type
        self._sensitivity = self._init_sensitivity(hessian_type)
        self._fail_on_td_target = fail_on_td_target
        self.td_errors: Optional[list[float]] = [] if record_td_errors else None

    def update(self) -> Optional[str]:
        raises = self._raises
        gradient_steps = (
            self.gradient_steps if self.gradient_steps > 0 else len(self.experience)
        )
        no_hessian = self.hessian_type == "none"
        statuses = ""
        ntheta = self._learnable_pars.size
        td_errors = self.td_errors

        for step in range(gradient_steps):
            mean_gradient = np.zeros(ntheta)
            mean_hessian = None if no_hessian else np.zeros((ntheta, ntheta))
            count_success = 0
            for i, (s, a, r, s_new, terminated) in enumerate(self.experience.sample()):
                # compute Q value estimate - if failure, terminate early
                solQ = self.action_value(s, a)
                if not solQ.success:
                    msg = f"Failure during update (trans. {i}; Q): {solQ.status}"
                    self.on_mpc_failure(-1, None, msg, raises)
                    if td_errors is not None:
                        td_errors.append(float("nan"))
                    continue

                # compute Q value target and TD error - if failure, terminate early only
                # if user asked for it via `fail_on_td_target`
                if terminated:
                    td_error = r - solQ.f
                else:
                    _, solV = self.state_value(s_new, True)
                    if not solV.success and self._fail_on_td_target:
                        msg = f"Failure during update (trans. {i}; V): {solV.status}"
                        self.on_mpc_failure(-1, None, msg, raises)
                        if td_errors is not None:
                            td_errors.append(float("nan"))
                        continue
                    td_error = r + self.discount_factor * solV.f - solQ.f
                if td_errors is not None:
                    td_errors.append(td_error)

                # accumulate sensitivities (gradient and hessian, if needed) of Q(s,a)
                count_success += 1
                if no_hessian:
                    dQ = self._sensitivity(solQ)
                else:
                    dQ, ddQ = self._sensitivity(solQ)
                    hessian = np.multiply.outer(dQ, dQ) - td_error * ddQ
                    mean_hessian += (hessian - mean_hessian) / count_success
                gradient = -td_error * dQ
                mean_gradient += (gradient - mean_gradient) / count_success

            # compute update with average gradient and hessian, if any
            if count_success > 0:
                status = self.optimizer.update(mean_gradient, mean_hessian)
                if status is not None:
                    statuses += f"{step}: {status}\n"
            else:
                statuses += f"{step}: no gradients computed, skipping update\n"

        return statuses if statuses else None

    def train_one_episode(
        self,
        env: Env[ObsType, ActType],
        episode: int,
        init_state: ObsType,
        raises: bool = True,
        behaviour_policy: Optional[Callable[[ObsType], ActType]] = None,
    ) -> float:
        truncated = terminated = False
        timestep = 0
        rewards = 0.0
        state = init_state
        action_space = getattr(env, "action_space", None)
        na = self.V.na
        ns = self.V.ns

        # NOTE: the point of this method is to rollout the behaviour policy and
        # populate the replay buffer with transitions. Updates are instead triggered via
        # callbacks on the events (e.g., timestep_end, env_step, etc.) and are not
        # part of this method but of `update()`

        while not (truncated or terminated):
            # compute the action to take
            if behaviour_policy is None:
                action, solV = self.state_value(state, False, action_space=action_space)
                if not solV.success:
                    self.on_mpc_failure(episode, None, solV.status, raises)
            else:
                action = behaviour_policy(state)

            # step the system with action
            new_state, cost, truncated, terminated, _ = env.step(action)
            self.on_env_step(env, episode, timestep)

            # store transition in the experience replay buffer, even if the MPC failed
            state_ = np.reshape(state, ns)
            action_ = np.reshape(action, na)
            new_state_ = np.reshape(new_state, ns)
            self.store_experience((state_, action_, cost, new_state_, terminated))

            # increase counters
            state = new_state
            rewards += float(cost)
            timestep += 1
            self.on_timestep_end(env, episode, timestep)
        return rewards

    def _init_sensitivity(
        self, hessian_type: Literal["none", "approx", "full"]
    ) -> Union[
        Callable[[Solution], np.ndarray],
        Callable[[Solution], tuple[np.ndarray, np.ndarray]],
    ]:
        """Internal utility to compute the derivative of ``Q(s,a)`` w.r.t. the learnable
        parameters, a.k.a., ``theta``."""
        ord = self.optimizer._order
        nlp = self._Q.nlp
        theta = cs.vvcat([nlp.parameters[p] for p in self._learnable_pars])
        x = nlp.x
        p = nlp.p
        lam_g_and_h = cs.vertcat(nlp.lam_g, nlp.lam_h)

        # compute first order sensitivity - necessary whatever the hessian type is
        snlp = NlpSensitivity(nlp, theta)
        gradient = snlp.jacobian("L-p")  # exact gradient, i.e., dQ/dtheta

        if hessian_type == "none":
            assert ord == 1, "Expected 1st-order optimizer with `hessian_type=none`."

            sensitivity = cs.Function(
                "lag_sens",
                [x, p, lam_g_and_h],
                [gradient],
                ["x", "p", "lam_g"],
                ["dQ"],
                {"cse": True},
            )

            def func(sol: Solution) -> np.ndarray:
                return np.asarray(sensitivity(sol.x, sol.p, sol.lam_g_and_h).elements())

        elif hessian_type == "approx":
            assert ord == 2, "Expected 2nd-order optimizer with `hessian_type=approx`."

            hessian = snlp.hessian("L-pp")  # approximate hessian

            # check if the hessian is not all zeros. If that's the case, we fall back to
            # computing just the gradient
            if hessian.nnz() > 0:
                sensitivity = cs.Function(
                    "lag_sens",
                    [x, p, lam_g_and_h],
                    [gradient, hessian],
                    ["x", "p", "lam_g"],
                    ["dQ", "ddQ"],
                    {"cse": True},
                )
                shape = sensitivity.size_out("ddQ")

                def func(sol: Solution) -> tuple[np.ndarray, np.ndarray]:
                    J, H = sensitivity(sol.x, sol.p, sol.lam_g_and_h)
                    return (
                        np.asarray(J.elements()),
                        np.reshape(H.elements(), shape, "F"),
                    )

            else:
                sensitivity = cs.Function(
                    "lag_sens",
                    [x, p, lam_g_and_h],
                    [gradient],
                    ["x", "p", "lam_g"],
                    ["dQ"],
                    {"cse": True},
                )

                def func(sol: Solution) -> tuple[np.ndarray, np.ndarray]:
                    J = sensitivity(sol.x, sol.p, sol.lam_g_and_h)
                    return np.asarray(J.elements()), 0.0

        else:
            assert ord == 2, "Expected 2nd-order optimizer with `hessian_type=full`."

            lam_lbx_and_ubx = cs.vertcat(nlp.lam_lbx, nlp.lam_ubx)
            Kp = snlp.jacobian("K-p")
            Ky = snlp.jacobian("K-y")
            dydtheta = -cs.solve(Ky, Kp)
            Lpy = cs.jacobian(gradient, nlp.primal_dual)
            hessian = snlp.hessian("L-pp") + Lpy @ dydtheta  # not sure if Lpy or Kp.T

            # check if the hessian is not all zeros. If that's the case, we fall back to
            # computing just the gradient
            if hessian.nnz() > 0:
                sensitivity = cs.Function(
                    "lag_sens",
                    [x, p, lam_g_and_h, lam_lbx_and_ubx],
                    [gradient, hessian],
                    ["x", "p", "lam_g", "lam_x"],
                    ["dQ", "ddQ"],
                    {"cse": True},
                )
                shape = sensitivity.size_out("ddQ")

                def func(sol: Solution) -> tuple[np.ndarray, np.ndarray]:
                    J, H = sensitivity(
                        sol.x, sol.p, sol.lam_g_and_h, sol.lam_lbx_and_ubx
                    )
                    return (
                        np.asarray(J.elements()),
                        np.reshape(H.elements(), shape, "F"),
                    )

            else:
                sensitivity = cs.Function(
                    "lag_sens",
                    [x, p, lam_g_and_h],  # should lam_lbx_and_ubx be included?
                    [gradient],
                    ["x", "p", "lam_g"],
                    ["dQ"],
                    {"cse": True},
                )

                def func(sol: Solution) -> tuple[np.ndarray, np.ndarray]:
                    J = sensitivity(sol.x, sol.p, sol.lam_g_and_h)
                    return np.asarray(J.elements()), 0.0

        return func
