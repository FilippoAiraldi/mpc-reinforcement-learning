import sys
from collections.abc import Collection, Iterable
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

# the experience buffer contains the gradient and, possibly, the hessian of the Bellman
# residuals w.r.t. the learnable parameters theta
ExpType: TypeAlias = Union[
    npt.NDArray[np.floating], tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
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
        warmstart: Union[
            Literal["last", "last-successful"], WarmStartStrategy
        ] = "last-successful",
        hessian_type: Literal["none", "approx", "full"] = "approx",
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
        self.hessian_type = hessian_type
        self._sensitivity = self._init_sensitivity(hessian_type)
        self.td_errors: Optional[list[float]] = [] if record_td_errors else None

    def update(self) -> Optional[str]:
        sample = self.experience.sample()
        if self.hessian_type == "none":
            gradient = np.mean(list(sample), 0)
            return self.optimizer.update(gradient)

        gradients, hessians = zip(*sample)
        gradient = np.mean(gradients, 0)
        hessian = np.mean(hessians, 0)
        return self.optimizer.update(gradient, hessian)

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
        action_space = getattr(env, "action_space", None)

        # solve for the first action
        action, solV = self.state_value(state, False, action_space=action_space)
        if not solV.success:
            self.on_mpc_failure(episode, None, solV.status, raises)

        while not (truncated or terminated):
            # compute Q(s,a)
            solQ = self.action_value(state, action)

            # step the system with action computed at the previous iteration
            new_state, cost, truncated, terminated, _ = env.step(action)
            self.on_env_step(env, episode, timestep)

            # compute V(s+) and store transition
            new_action, solV = self.state_value(
                new_state, False, action_space=action_space
            )
            if not self._try_store_experience(cost, solQ, solV):
                self.on_mpc_failure(
                    episode, timestep, f"{solQ.status} (Q); {solV.status} (V)", raises
                )

            # increase counters
            state = new_state
            action = new_action
            rewards += float(cost)
            timestep += 1
            self.on_timestep_end(env, episode, timestep)
        return rewards

    def train_one_rollout(
        self,
        rollout: Iterable[tuple[ObsType, ActType, float, ObsType]],
        episode: int,
        raises: bool = True,
    ) -> None:
        # in the case of off-policy q-learning, rollouts are made of
        # State-Action-Reward-next State (SARS) tuples
        for timestep, (state, action, cost, new_state) in enumerate(rollout, start=1):
            # compute Q(s,a)
            solQ = self.action_value(state, action)

            # compute V(s+) and store transition
            _, solV = self.state_value(new_state, False)
            if not self._try_store_experience(cost, solQ, solV):
                self.on_mpc_failure(
                    episode, timestep, f"{solQ.status} (Q); {solV.status} (V)", raises
                )
            self.on_timestep_end("off-policy", episode, timestep)

    def _init_sensitivity(
        self, hessian_type: Literal["none", "approx", "full"]
    ) -> Union[
        Callable[[cs.DM], np.ndarray], Callable[[cs.DM], tuple[np.ndarray, np.ndarray]]
    ]:
        """Internal utility to compute the derivative of ``Q(s,a)`` w.r.t. the learnable
        parameters, a.k.a., ``theta``."""
        ord = self.optimizer._order
        theta = cs.vvcat(self._learnable_pars.sym.values())
        nlp = self._Q.nlp
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

    def _try_store_experience(
        self, cost: SupportsFloat, solQ: Solution[SymType], solV: Solution[SymType]
    ) -> bool:
        """Internal utility that tries to store the gradient and hessian for the current
        transition in memory, if both ``V`` and ``Q`` were successful; otherwise, does
        not store it. Returns whether it was successful or not."""
        success = solQ.success and solV.success
        if success:
            td_error = cost + self.discount_factor * solV.f - solQ.f
            if self.hessian_type == "none":
                dQ = self._sensitivity(solQ)
                gradient = -td_error * dQ
                self.store_experience(gradient)
            else:
                dQ, ddQ = self._sensitivity(solQ)
                gradient = -td_error * dQ
                hessian = np.multiply.outer(dQ, dQ) - td_error * ddQ
                self.store_experience((gradient, hessian))
        else:
            td_error = np.nan

        if self.td_errors is not None:
            self.td_errors.append(td_error)
        return success
