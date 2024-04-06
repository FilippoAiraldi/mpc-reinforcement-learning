"""This example shows how to use BoTorch (a Bayesian Optimization library) to optimize
the parameters of a parametric MPC policy. The numerical example is inspired by [1].

References
----------
[1] Sorourifar, F., Makrygirgos, G., Mesbah, A. and Paulson, J.A., 2021. A data-driven
    automatic tuning method for MPC under uncertainty using constrained Bayesian
    optimization. IFAC-PapersOnLine, 54(3), pp.243-250.
"""

from logging import DEBUG
from operator import neg
from typing import Any, Optional

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from csnlp.multistart import (
    ParallelMultistartNlp,
    RandomStartPoint,
    RandomStartPoints,
    StructuredStartPoint,
    StructuredStartPoints,
)
from csnlp.wrappers import Mpc
from gpytorch.mlls import ExactMarginalLogLikelihood
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit, TransformReward
from scipy.stats.qmc import LatinHypercube

from mpcrl import (
    GlobOptLearningAgent,
    LearnableParameter,
    LearnableParametersDict,
    WarmStartStrategy,
)
from mpcrl.optim import GradientFreeOptimizer
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes


class CstrEnv(gym.Env[npt.NDArray[np.floating], float]):
    """
    ## Description

    Continuously stirred tank reactor environment. The ongoing reaction is
                    A -> B -> C, 2A -> D.

    ## Action Space

    The action is an array of shape `(1,)`, where the action is the normalized inflow
    rate of the tank. It is bounded in the range `[5, 35]`.

    ## Observation Space

    The state space is an array of shape `(4,)`, containing the concentrations of
    reagents A and B (mol/L), and the temperatures of the reactor and the coolant (°C).
    The first two states must be positive, while the latter two (the temperatures)
    should be bounded (but not forced) below `100` and `150`, respectively.

    The observation (a.k.a., measurable states) space is an array of shape `(2,)`,
    containing the concentration of reagent B and the temperature of the reactor. The
    former is unconstrained, while the latter should be bounded (but not forced) in the
    range `[`100`, 150]` (See Rewards section).

    The internal, non-observable states are the concentrations of reagent A and B, and
    the temperatures of the reactor and the coolant, for a total of 4 states.

    ## Rewards

    The reward here is intended as the number of moles of production of component B.
    However, penalties are incurred for violating bounds on the temperature of the
    reactor.

    ## Noises and Starting State

    The initial state is set to `[1, 1, 100, 100]`. It is possible to have zero-mean
    gaussian noise on the measurements of the states.

    ## Episode End

    The episode does not have an end, so wrapping it in, e.g., `TimeLimit`, is strongly
    suggested.
    """

    ns = 4  # number of states
    na = 1  # number of inputs
    reactor_temperature_bound = (100, 150)
    inflow_bound = (5, 35)
    x0 = np.asarray([1.0, 1.0, 100.0, 100.0])  # initial state

    def __init__(self, constraint_violation_penalty: float = 2e1) -> None:
        """Creates a CSTR environment.

        Parameters
        ----------
        constraint_violation_penalty : float, optional
            Reward penalty for violating soft constraints on the reactor temperature.
        """
        super().__init__()
        self.constraint_violation_penalty = constraint_violation_penalty
        self.observation_space = Box(
            np.array([0, 0, -273.15, -273.15]), np.inf, (self.ns,), np.float64
        )
        self.action_space = Box(*self.inflow_bound, (self.na,), np.float64)

        # build the nonlinear dynamics (see [1, Table 1] for these values)
        k01 = k02 = (1.287, 12)
        k03 = (9.043, 9)
        EA1R = EA2R = 9758.3
        EA3R = 7704.0
        DHAB = 4.2
        DHBC = 4.2
        DHAD = 4.2
        rho = 0.9342
        cP = 3.01
        cPK = 2.0
        A = 0.215
        VR = 10.01
        mK = 5.0
        Tin = 130.0
        kW = 4032
        QK = -4500
        x = cs.SX.sym("x", self.ns)
        F = cs.SX.sym("u", self.na)
        cA, cB, TR, TK = cs.vertsplit_n(x, self.ns)
        k1 = k01[0] * cs.exp(k01[1] * np.log(10) - EA1R / (TR + 273.15))
        k2 = k02[0] * cs.exp(k02[1] * np.log(10) - EA2R / (TR + 273.15))
        k3 = k03[0] * cs.exp(k03[1] * np.log(10) - EA3R / (TR + 273.15))
        cA_dot = F * (self.x0[0] - cA) - k1 * cA - k3 * cA**2
        cB_dot = -F * cB + k1 * cA - k2 * cB
        TR_dot = (
            F * (Tin - TR)
            + kW * A / (rho * cP * VR) * (TK - TR)
            - (k1 * cA * DHAB + k2 * cB * DHBC + k3 * cA**2 * DHAD) / (rho * cP)
        )
        TK_dot = (QK + kW * A * (TR - TK)) / (mK * cPK)
        x_dot = cs.vertcat(cA_dot, cB_dot, TR_dot, TK_dot)
        reward = VR * F * cB
        dae = {"x": x, "p": F, "ode": cs.cse(cs.simplify(x_dot)), "quad": reward}
        tf = 0.2 / 40  # 0.2 hours / 40 steps
        self.dynamics = cs.integrator("cstr_dynamics", "cvodes", dae, 0.0, tf)
        self.VR = VR
        self.tf = tf

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the state of the CSTR env."""
        super().reset(seed=seed, options=options)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        state = self.x0.copy()
        assert self.observation_space.contains(state), f"invalid reset state {state}"
        self._state = state.copy()
        return state, {}

    def step(
        self, action: cs.DM
    ) -> tuple[npt.NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Steps the CSTR env."""
        action = np.reshape(action, self.action_space.shape)
        integration = self.dynamics(x0=self._state, p=action)
        state = np.asarray(integration["xf"].elements())
        assert self.action_space.contains(action) and self.observation_space.contains(
            state
        ), f"invalid step action {action} or state {state}"

        reward = float(integration["qf"])
        reactor_temperature = self._state[2]
        reward -= self.constraint_violation_penalty * (
            max(0, self.reactor_temperature_bound[0] - reactor_temperature)
            + max(0, reactor_temperature - self.reactor_temperature_bound[1])
        )

        self._state = state.copy()
        return state, reward, False, False, {}


class NoisyFilterObservation(ObservationWrapper):
    """Wrapper for filtering the env's (internal) states to the subset of measurable
    ones. Moreover, it can corrupt the measurements with additive zero-mean gaussian
    noise."""

    def __init__(
        self,
        env: gym.Env,
        measurable_states: list[int],
        measurement_noise_std: Optional[list[float]] = None,
    ) -> None:
        """Instantiates the wrapper.

        Parameters
        ----------
        env : gymnasium Env
            The env to wrap.
        measurable_states : list of int
            The indices of the states that are measurables.
        measurement_noise_std : list of float, optional
            The standard deviation of the measurement noise to be applied to the
            measurements. If specified, must have the same length as the indices. If
            `None`, no noise is applied.
        """
        super().__init__(env)
        self.measurable_states = measurable_states
        self.measurement_noise_std = measurement_noise_std
        low = env.observation_space.low[measurable_states]
        high = env.observation_space.high[measurable_states]
        self.observation_space = Box(
            low, high, (len(measurable_states),), env.observation_space.dtype
        )

    def observation(
        self, observation: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        measurable = observation[self.measurable_states]
        if self.measurement_noise_std is not None:
            np.clip(
                measurable + self.np_random.normal(scale=self.measurement_noise_std),
                self.observation_space.low,
                self.observation_space.high,
                out=measurable,
            )
        return measurable


def get_cstr_mpc(env: CstrEnv, horizon: int = 10) -> Mpc[cs.SX]:
    """Returns an MPC controller for the given CSTR env."""
    nlp = ParallelMultistartNlp[cs.SX]("SX", starts=10, n_jobs=1)
    mpc = Mpc[cs.SX](nlp, horizon)

    # variables (state, action)
    y_space, u_space = env.observation_space, env.action_space
    ny, nu = y_space.shape[0], u_space.shape[0]
    y, _ = mpc.state("y", ny, bound_initial=False)
    u, _ = mpc.action("u", nu, u_space.low[:, None], u_space.high[:, None])

    # set the dynamics based on the NARX model - but first scale to [0, 1]
    lb = np.concatenate([[0.0, 100.0], u_space.low])
    ub = np.concatenate([[1.0, 150.0], u_space.high])
    n_weights = 1 + 2 * (ny + nu)
    narx_weights = (
        mpc.parameter("narx_weights", (n_weights * ny, 1)).reshape((-1, ny)).T
    )

    def narx_dynamics(y: cs.SX, u: cs.SX) -> cs.SX:
        yu = cs.vertcat(y, u)
        yu_scaled = (yu - lb) / (ub - lb)
        basis = cs.vertcat(cs.SX.ones(1), yu_scaled, yu_scaled**2)
        y_next_scaled = cs.mtimes(narx_weights, basis)
        y_next = y_next_scaled * (ub[:ny] - lb[:ny]) + lb[:ny]
        return cs.cse(cs.simplify(y_next))

    mpc.set_dynamics(narx_dynamics, n_in=2, n_out=1)

    # add constraints on the reactor temperature (soft and with backoff)
    b = mpc.parameter("backoff")
    _, _, slack_lb = mpc.constraint("TR_lb", y[1, :], ">=", 100.0 + b, soft=True)
    _, _, slack_ub = mpc.constraint("TR_ub", y[1, :], "<=", 150.0 - b, soft=True)

    # objective production of moles of B with penalties for violations
    mpc.minimize(
        env.get_wrapper_attr("VR") * env.get_wrapper_attr("tf") * cs.sum2(y[0, :-1] * u)
        + env.get_wrapper_attr("constraint_violation_penalty")
        * cs.sum2(slack_lb + slack_ub)
    )

    # solver
    opts = {
        "expand": True,
        "print_time": False,
        "bound_consistency": True,
        "calc_lam_x": False,
        "calc_lam_p": False,
        "calc_multipliers": False,
        "ipopt": {
            "max_iter": 500,
            "sb": "yes",
            "print_level": 0,
        },
    }
    mpc.init_solver(opts, solver="ipopt")
    return mpc


class BoTorchOptimizer(GradientFreeOptimizer):
    """Implements a Bayesian Optimization optimizer based on BoTorch."""

    prefers_dict = False  # ask and tell methods deal with arrays, not dicts

    def __init__(
        self, initial_random: int = 5, seed: Optional[int] = None, **kwargs: Any
    ) -> None:
        """Initializes the optimizer.

        Parameters
        ----------
        initial_random : int, optional
            Number of initial random guesses, by default 5. Must be positive.
        seed : int, optional
            Seed for the random number generator, by default `None`.
        """
        if initial_random <= 0:
            raise ValueError("`initial_random` must be positive.")
        super().__init__(**kwargs)
        self._initial_random = initial_random
        self._seed = seed

    def _init_update_solver(self) -> None:
        # compute the current bounds on the learnable parameters
        pars = self.learnable_parameters
        values = pars.value
        lb, ub = (values + bnd for bnd in self._get_update_bounds(values))

        # use latin hypercube sampling to generate the initial random guesses
        lhs = LatinHypercube(pars.size, seed=self._seed)
        self._train_inputs = lhs.random(self._initial_random) * (ub - lb) + lb
        self._train_targets = np.empty((0,))  # we dont know the targets yet

    def ask(self) -> tuple[npt.NDArray[np.floating], None]:
        # use targets to track the iteration
        iteration = self._train_targets.shape[0]

        # just return the next random guess, if still in the initial random phase
        if iteration < self._initial_random:
            return self._train_inputs[iteration], None

        # otherwise, use GP-BO to find the next guess
        # prepare data for fitting GP
        train_inputs = torch.from_numpy(self._train_inputs)
        train_targets = standardize(torch.from_numpy(self._train_targets).unsqueeze(-1))

        # fit the GP
        values = self.learnable_parameters.value
        bounds = torch.from_numpy(
            np.stack([values + bnd for bnd in self._get_update_bounds(values)])
        )
        normalize = Normalize(train_inputs.shape[-1], bounds=bounds)
        gp = SingleTaskGP(train_inputs, train_targets, input_transform=normalize)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))

        # maximize the acquisition function to get the next guess
        acqfun = ExpectedImprovement(gp, train_targets.amin(), maximize=False)
        acqfun_optimizer = optimize_acqf(
            acqfun, bounds, 1, 32, 128, {"seed": self._seed + iteration}
        )[0].numpy()
        self._train_inputs = np.append(self._train_inputs, acqfun_optimizer, axis=0)
        return acqfun_optimizer.reshape(-1), None

    def tell(self, values: npt.NDArray[np.floating], objective: float) -> None:
        # grab the current iteration and check that the tell method is called in the
        # correct order
        iteration = self._train_targets.size
        assert (values == self._train_inputs[iteration]).all()

        # append the new target to the training data
        self._train_targets = np.append(self._train_targets, objective)


if __name__ == "__main__":
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    # create the environment, and wrap it appropriately
    T_max = 40
    env = MonitorEpisodes(TimeLimit(CstrEnv(), max_episode_steps=T_max))
    env = TransformReward(env, neg)
    env = NoisyFilterObservation(env, [1, 2])

    # create the mpc and the dict of learnable parameters - the initial values we give
    # here do not really matter since BO will have a couple of initial random
    # evaluations anyway
    mpc = get_cstr_mpc(env)
    pars = mpc.parameters
    learnable_pars = LearnableParametersDict[cs.SX](
        (
            LearnableParameter(n, pars[n].shape, (ub + lb) / 2, lb, ub, pars[n])
            for n, lb, ub in [("narx_weights", -2, 2), ("backoff", 0, 5)]
        )
    )

    # since the MPC is highly nonlinear due to the NARX model, set up a warmstart
    # strategy in order to automatically try different initial conditions for the solver
    ns, na, N = mpc.ns, mpc.na, mpc.prediction_horizon
    warmstart = WarmStartStrategy(
        structured_points=StructuredStartPoints(
            {
                "y": StructuredStartPoint(
                    np.full((ns, N + 1), [[0.0], [50.0]]),
                    np.full((ns, N + 1), [[1.0], [150.0]]),
                ),
                "u": StructuredStartPoint(
                    np.full((na, N), env.action_space.low),
                    np.full((na, N), env.action_space.high),
                ),
            },
            multistarts=0,  # will be overwritten automatically
        ),
        random_points=RandomStartPoints(
            {
                "y": RandomStartPoint("normal", scale=[[0.2], [30]], size=(ns, N + 1)),
                "u": RandomStartPoint("normal", scale=5.0, size=(na, N)),
            },
            multistarts=0,  # will be overwritten automatically
            biases={
                "y": [[CstrEnv.x0[1]], [CstrEnv.x0[2]]],
                "u": sum(CstrEnv.inflow_bound) / 2,
            },
        ),
    )

    # create the agent, and wrap it appropriately
    agent = GlobOptLearningAgent(
        mpc=mpc,
        learnable_parameters=learnable_pars,
        warmstart=warmstart,
        optimizer=BoTorchOptimizer(seed=42),
    )
    agent = Log(
        RecordUpdates(agent), level=DEBUG, log_frequencies={"on_episode_end": 1}
    )

    # finally, launch the training
    episodes = 50
    agent.train(env=env, episodes=episodes, seed=69, raises=False)

    # plot the result
    import matplotlib.pyplot as plt

    X = np.asarray(env.observations)  # n_ep x T + 1 x ns
    U = np.squeeze(env.actions, (2, 3))  # n_ep x T
    R = np.asarray(env.rewards)  # n_ep x T

    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    time = np.linspace(0, 0.2, T_max + 1)
    linewidths = np.linspace(0.25, 2.5, episodes)

    axs[0].hlines(
        env.reactor_temperature_bound, time[0], time[-1], colors="k", linestyles="--"
    )
    axs[1].hlines(env.inflow_bound, time[0], time[-1], colors="k", linestyles="--")
    for i in range(episodes):
        axs[0].plot(time, X[i, :, 2], color="C0", lw=linewidths[i])
        axs[1].plot(time[:-1], U[i], color="C0", lw=linewidths[i])

    episodes = np.arange(1, episodes + 1)
    returns = R.sum(axis=1)
    idx_max = np.argmax(returns)
    axs[2].semilogy(episodes, np.maximum.accumulate(returns), color="C1")
    axs[2].semilogy(episodes, returns, "o", color="C0")
    axs[2].semilogy(episodes[idx_max], returns[idx_max], "*", markersize=10, color="C2")

    axs[0].set_xlabel(r"Time (h)")
    axs[0].set_ylabel(r"$T_R$ (°C)")
    axs[1].set_xlabel(r"Time (h)")
    axs[1].set_ylabel(r"F ($h^{-1}$)")
    axs[2].set_xlabel(r"Learning iteration")
    axs[2].set_ylabel(r"$n_B$ (mol)")

    # fig.savefig("examples/bayesopt.pdf")
    plt.show()
