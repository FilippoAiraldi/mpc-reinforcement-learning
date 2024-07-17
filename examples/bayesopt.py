"""This example shows how to use BoTorch (a Bayesian Optimization library) to optimize
the parameters of a parametric MPC policy. The numerical example is inspired by [1].

References
----------
[1] Sorourifar, F., Makrygirgos, G., Mesbah, A. and Paulson, J.A., 2021. A data-driven
    automatic tuning method for MPC under uncertainty using constrained Bayesian
    optimization. IFAC-PapersOnLine, 54(3), pp.243-250.
"""

from collections.abc import Iterable
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
from gymnasium import Env, ObservationWrapper
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
    range `[100, 150]` (See Rewards section).

    The internal, non-observable states are the concentrations of reagent A and B, and
    the temperatures of the reactor and the coolant, for a total of 4 states.

    ## Rewards

    The reward to be maximized here is to be intended as the number of moles of
    production of component B. However, penalties are incurred for violating bounds on
    the temperature of the reactor.

    ## Starting State

    The initial state is set to `[1, 1, 100, 100]`

    ## Episode End

    The episode does not have an end, so wrapping it in, e.g., `TimeLimit`, is strongly
    suggested.

    References
    ----------
    [1] Sorourifar, F., Makrygirgos, G., Mesbah, A. and Paulson, J.A., 2021. A
        data-driven automatic tuning method for MPC under uncertainty using constrained
        Bayesian optimization. IFAC-PapersOnLine, 54(3), pp.243-250.
    """

    ns = 4  # number of states
    na = 1  # number of inputs
    reactor_temperature_bound = (100, 150)
    inflow_bound = (5, 35)
    x0 = np.asarray([1.0, 1.0, 100.0, 100.0])  # initial state

    def __init__(self, constraint_violation_penalty: float) -> None:
        """Creates a CSTR environment.

        Parameters
        ----------
        constraint_violation_penalty : float, optional
            Reward penalty for violating soft constraints on the reactor temperature.
        """
        super().__init__()
        self.constraint_violation_penalty = constraint_violation_penalty
        self.observation_space = Box(
            np.array([0.0, 0.0, -273.15, -273.15]), np.inf, (self.ns,), np.float64
        )
        self.action_space = Box(*self.inflow_bound, (self.na,), np.float64)

        # set the nonlinear dynamics parameters (see [1, Table 1] for these values)
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
        self.VR = VR = 10.01
        mK = 5.0
        Tin = 130.0
        kW = 4032
        QK = -4500

        # instantiate states and control action
        x = cs.SX.sym("x", self.ns)
        cA, cB, TR, TK = cs.vertsplit_n(x, self.ns)
        F = cs.SX.sym("u", self.na)

        # define the states' PDEs
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

        # define the reward function, i.e., moles of B + constraint penalties
        lb_TR, ub_TR = self.reactor_temperature_bound
        reward = VR * F * cB - constraint_violation_penalty * (
            cs.fmax(0, lb_TR - TR) + cs.fmax(0, TR - ub_TR)
        )

        # build the casadi integrator
        dae = {"x": x, "p": F, "ode": cs.cse(cs.simplify(x_dot)), "quad": reward}
        self.tf = 0.2 / 40  # 0.2 hours / 40 steps
        self.dynamics = cs.integrator("cstr_dynamics", "cvodes", dae, 0.0, self.tf)

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
        self._state = self.x0
        assert self.observation_space.contains(
            self._state
        ), f"invalid reset state {self._state}"
        return self._state.copy(), {}

    def step(
        self, action: cs.DM
    ) -> tuple[npt.NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Steps the CSTR env."""
        action = np.reshape(action, self.action_space.shape)
        assert self.action_space.contains(action), f"invalid step action {action}"
        integration = self.dynamics(x0=self._state, p=action)
        self._state = np.asarray(integration["xf"].elements())
        assert self.observation_space.contains(
            self._state
        ), f"invalid step next state {self._state}"
        return self._state.copy(), float(integration["qf"]), False, False, {}


class NoisyFilterObservation(ObservationWrapper):
    """Wrapper for filtering the env's (internal) states to the subset of measurable
    ones. Moreover, it can corrupt the measurements with additive zero-mean gaussian
    noise."""

    def __init__(
        self,
        env: Env[npt.NDArray[np.floating], float],
        measurable_states: Iterable[int],
        measurement_noise_std: Optional[npt.ArrayLike] = None,
    ) -> None:
        """Instantiates the wrapper.

        Parameters
        ----------
        env : gymnasium Env
            The env to wrap.
        measurable_states : iterable of int
            The indices of the states that are measurables.
        measurement_noise_std : array-like, optional
            The standard deviation of the measurement noise to be applied to the
            measurements. If specified, must have the same length as the indices. If
            `None`, no noise is applied.
        """
        assert isinstance(env.observation_space, Box), "only Box spaces are supported."
        super().__init__(env)
        self.measurable_states = list(map(int, measurable_states))
        self.measurement_noise_std = measurement_noise_std
        low = env.observation_space.low[self.measurable_states]
        high = env.observation_space.high[self.measurable_states]
        self.observation_space = Box(low, high, low.shape, env.observation_space.dtype)

    def observation(
        self, observation: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        measurable = observation[self.measurable_states]
        if self.measurement_noise_std is not None:
            obs_space: Box = self.observation_space
            noise = self.np_random.normal(scale=self.measurement_noise_std)
            measurable = np.clip(measurable + noise, obs_space.low, obs_space.high)
        assert self.observation_space.contains(measurable), "Invalid measurable state."
        return measurable


def get_cstr_mpc(
    env: CstrEnv, horizon: int, multistarts: int, n_jobs: int
) -> Mpc[cs.SX]:
    """Returns an MPC controller for the given CSTR env."""
    nlp = ParallelMultistartNlp[cs.SX](
        "SX",
        starts=multistarts,
        parallel_kwargs={"n_jobs": n_jobs, "return_as": "generator"},
    )
    mpc = Mpc[cs.SX](nlp, horizon)

    # variables (state, action)
    y_space, u_space = env.observation_space, env.action_space
    ny, nu = y_space.shape[0], u_space.shape[0]
    y, _ = mpc.state(
        "y",
        ny,
        lb=y_space.low[:, None],
        ub=[[1e2], [1e3]],  # just some high numbers to bound the state domain
        bound_initial=False,
    )
    u, _ = mpc.action("u", nu, lb=u_space.low[:, None], ub=u_space.high[:, None])

    # set the dynamics based on the NARX model - but first scale approximately to [0, 1]
    lb = np.concatenate([[0.0, 100.0], u_space.low])
    ub = np.concatenate([[10.0, 150.0], u_space.high])
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

    # objective  is the production of moles of B with penalties for violations
    VR = env.get_wrapper_attr("VR")
    cv = env.get_wrapper_attr("constraint_violation_penalty")
    tf = env.get_wrapper_attr("tf")
    moles_B = VR * cs.sum2(u * y[0, :-1])
    constr_viol = cv * cs.sum2(slack_lb + slack_ub)
    mpc.minimize((constr_viol - moles_B) * tf)

    # solver
    opts = {
        "expand": True,
        "print_time": False,
        "bound_consistency": True,
        "calc_lam_x": False,
        "calc_lam_p": False,
        "calc_multipliers": False,
        "ipopt": {"max_iter": 1000, "sb": "yes", "print_level": 0},
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
        self._n_ask = -1  # to track the number of ask iterations

    def ask(self) -> tuple[npt.NDArray[np.floating], None]:
        self._n_ask += 1

        # if still in the initial random phase, just return the next random guess
        if self._n_ask < self._initial_random:
            return self._train_inputs[self._n_ask], None

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
        af = ExpectedImprovement(gp, train_targets.amin(), maximize=False)
        seed = self._seed + self._n_ask
        acqfun_optimizer = (
            optimize_acqf(af, bounds, 1, 16, 64, {"seed": seed})[0].numpy().reshape(-1)
        )
        return acqfun_optimizer, None

    def tell(self, values: npt.NDArray[np.floating], objective: float) -> None:
        iteration = self._n_ask
        if iteration < 0:
            raise RuntimeError("`ask` must be called before `tell`.")

        # append the new datum to the training data
        if iteration < self._initial_random:
            assert (
                values == self._train_inputs[iteration]
            ).all(), "`tell` called with a different value than the one given by `ask`."
            self._train_inputs[iteration] = values
        else:
            self._train_inputs = np.append(
                self._train_inputs, values.reshape(1, -1), axis=0
            )
        self._train_targets = np.append(self._train_targets, objective)


if __name__ == "__main__":
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    # create the environment, and wrap it appropriately
    constraint_violation_penalty = 4e3
    max_episode_steps = 40
    measurable_states = [1, 2]
    measurement_noise_std = (0.2, 10.0)
    env = NoisyFilterObservation(
        TransformReward(
            MonitorEpisodes(
                TimeLimit(
                    CstrEnv(constraint_violation_penalty=constraint_violation_penalty),
                    max_episode_steps=max_episode_steps,
                )
            ),
            f=neg,
        ),
        measurable_states=measurable_states,
        measurement_noise_std=measurement_noise_std,
    )

    # create the mpc and the dict of learnable parameters - the initial values we give
    # here do not really matter since BO will have a couple of initial random
    # evaluations anyway
    horizon = 10
    multistarts = 10
    mpc = get_cstr_mpc(env, horizon, multistarts, n_jobs=multistarts)
    pars = mpc.parameters
    learnable_pars = LearnableParametersDict[cs.SX](
        (
            LearnableParameter(n, pars[n].shape, (lb + ub) / 2, lb, ub, pars[n])
            for n, lb, ub in [("narx_weights", -2, 2), ("backoff", 0, 10)]
        )
    )

    # since the MPC is highly nonlinear due to the NARX model, set up a warmstart
    # strategy in order to automatically try different initial conditions for the solver
    Y = mpc.variables["y"].shape
    U = mpc.variables["u"].shape
    act_space = env.action_space
    multistarts_struct = (multistarts - 1) // 2
    multistarts_rand = multistarts - 1 - multistarts_struct
    ns, na, N = mpc.ns, mpc.na, mpc.prediction_horizon
    warmstart = WarmStartStrategy(
        structured_points=StructuredStartPoints(
            {
                "y": StructuredStartPoint(
                    np.full(Y, [[0.0], [50.0]]), np.full(Y, [[20.0], [150.0]])
                ),
                "u": StructuredStartPoint(
                    np.full(U, act_space.low), np.full(U, act_space.high)
                ),
            },
            multistarts=multistarts_struct,
        ),
        random_points=RandomStartPoints(
            {
                "y": RandomStartPoint("normal", scale=[[1.0], [20.0]], size=Y),
                "u": RandomStartPoint("normal", scale=5.0, size=U),
            },
            biases={
                "y": CstrEnv.x0[measurable_states].reshape(-1, 1),
                "u": sum(CstrEnv.inflow_bound) / 2,
            },
            multistarts=multistarts_rand,
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
    episodes = 40
    agent.train(env=env, episodes=episodes, seed=69, raises=False)

    # plot the result
    import matplotlib.pyplot as plt

    X = np.asarray(env.get_wrapper_attr("observations"))  # n_ep x T + 1 x ns
    U = np.squeeze(env.get_wrapper_attr("actions"), (2, 3))  # n_ep x T
    R = np.asarray(env.get_wrapper_attr("rewards"))  # n_ep x T
    reactor_temperature_bound = env.get_wrapper_attr("reactor_temperature_bound")
    inflow_bound = env.get_wrapper_attr("inflow_bound")
    reactor_temperature_bound = env.get_wrapper_attr("reactor_temperature_bound")

    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    time = np.linspace(0, 0.2, max_episode_steps + 1)
    linewidths = np.linspace(0.25, 2.5, episodes)
    axs[0].hlines(
        reactor_temperature_bound, time[0], time[-1], colors="k", linestyles="--"
    )
    axs[1].hlines(inflow_bound, time[0], time[-1], colors="k", linestyles="--")
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
