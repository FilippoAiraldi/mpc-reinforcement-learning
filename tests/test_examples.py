import logging
import pickle
import unittest
from operator import neg
from sys import platform
from typing import Any, Optional
from warnings import catch_warnings

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
from csnlp import Nlp
from csnlp.wrappers import Mpc
from gpytorch.mlls import ExactMarginalLogLikelihood
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit, TransformReward
from parameterized import parameterized
from scipy.io import loadmat
from scipy.stats.qmc import LatinHypercube

from mpcrl import (
    GlobOptLearningAgent,
    LearnableParameter,
    LearnableParametersDict,
    LstdDpgAgent,
    LstdQLearningAgent,
    UpdateStrategy,
)
from mpcrl import exploration as E
from mpcrl.optim import GradientDescent, GradientFreeOptimizer, NetwonMethod
from mpcrl.util.control import dlqr
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)


class LtiSystem(gym.Env[npt.NDArray[np.floating], float]):
    nx = 2  # number of states
    nu = 1  # number of inputs
    A = np.asarray([[0.9, 0.35], [0, 1.1]])  # state-space matrix A
    B = np.asarray([[0.0813], [0.2]])  # state-space matrix B
    x_bnd = (np.asarray([[0], [-1]]), np.asarray([[1], [1]]))  # bounds of state
    a_bnd = (-1, 1)  # bounds of control input
    w = np.asarray([[1e2], [1e2]])  # penalty weight for bound violations
    e_bnd = (-1e-1, 0)  # uniform noise bounds

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.x = np.asarray([0, 0.15]).reshape(self.nx, 1)
        return self.x, {}

    def get_stage_cost(self, state: npt.NDArray[np.floating], action: float) -> float:
        lb, ub = self.x_bnd
        return 0.5 * float(
            np.square(state).sum()
            + 0.5 * action**2
            + self.w.T @ np.maximum(0, lb - state)
            + self.w.T @ np.maximum(0, state - ub)
        )

    def step(
        self, action: cs.DM
    ) -> tuple[npt.NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        action = float(action)
        x_new = self.A @ self.x + self.B * action
        x_new[0] += self.np_random.uniform(*self.e_bnd)
        self.x = x_new
        r = self.get_stage_cost(self.x, action)
        return x_new, r, False, False, {}


class QLearningLinearMpc(Mpc[cs.SX]):
    horizon = 10
    discount_factor = 0.9
    learnable_pars_init = {
        "V0": np.asarray(0.0),
        "x_lb": np.asarray([0, 0]),
        "x_ub": np.asarray([1, 0]),
        "b": np.full(LtiSystem.nx, 0.0),
        "f": np.full(LtiSystem.nx + LtiSystem.nu, 0.0),
        "A": np.asarray([[1, 0.25], [0, 1]]),
        "B": np.asarray([[0.0312], [0.25]]),
    }

    def __init__(self) -> None:
        N = self.horizon
        gamma = self.discount_factor
        w = LtiSystem.w
        nx, nu = LtiSystem.nx, LtiSystem.nu
        x_bnd, a_bnd = LtiSystem.x_bnd, LtiSystem.a_bnd
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)
        V0 = self.parameter("V0")
        x_lb = self.parameter("x_lb", (nx,))
        x_ub = self.parameter("x_ub", (nx,))
        b = self.parameter("b", (nx, 1))
        f = self.parameter("f", (nx + nu, 1))
        A = self.parameter("A", (nx, nx))
        B = self.parameter("B", (nx, nu))
        x, _ = self.state("x", nx)
        u, _ = self.action("u", nu, lb=a_bnd[0], ub=a_bnd[1])
        s, _, _ = self.variable("s", (nx, N), lb=0)
        self.set_dynamics(lambda x, u: A @ x + B * u + b, n_in=2, n_out=1)
        self.constraint("x_lb", x_bnd[0] + x_lb - s, "<=", x[:, 1:])
        self.constraint("x_ub", x[:, 1:], "<=", x_bnd[1] + x_ub + s)
        A_init, B_init = self.learnable_pars_init["A"], self.learnable_pars_init["B"]
        S = cs.DM(dlqr(A_init, B_init, 0.5 * np.eye(nx), 0.25 * np.eye(nu))[1])
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize(
            V0
            + cs.bilin(S, x[:, -1])
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
            * cs.sum2(
                gammapowers
                * (cs.sum1(x[:, :-1] ** 2) + 0.5 * cs.sum1(u**2) + w.T @ s)
            )
        )
        opts = {
            "expand": True,
            "print_time": False,
            "ipopt": {"max_iter": 500, "sb": "yes", "print_level": 0},
        }
        self.init_solver(opts, solver="ipopt")


class DpgLinearMpc(Mpc[cs.SX]):
    horizon = 10
    discount_factor = 0.9
    learnable_pars_init = {  # test also with pars with additional dims
        "V0": np.asarray([[0.0], [0.0]]),
        "x_lb": np.asarray([[0], [0]]),
        "x_ub": np.asarray([[1], [0]]),
        "b": np.full((LtiSystem.nx, 1), 0.0),
        "f": np.full((LtiSystem.nx + LtiSystem.nu, 1), 0.0),
        "A": np.asarray([[1, 0.25], [0, 1]]),
        "B": np.asarray([[0.0312], [0.25]]),
    }

    def __init__(self) -> None:
        N = self.horizon
        gamma = self.discount_factor
        w = LtiSystem.w
        nx, nu = LtiSystem.nx, LtiSystem.nu
        x_bnd, a_bnd = LtiSystem.x_bnd, LtiSystem.a_bnd
        nlp = Nlp[cs.SX]()
        super().__init__(nlp, N)
        V0 = self.parameter("V0", (nx,))
        x_lb = self.parameter("x_lb", (nx,))
        x_ub = self.parameter("x_ub", (nx,))
        b = self.parameter("b", (nx, 1))
        f = self.parameter("f", (nx + nu, 1))
        A = self.parameter("A", (nx * nx, 1)).reshape((nx, nx))
        B = self.parameter("B", (nx * nu, 1)).reshape((nx, nu))
        x, _ = self.state("x", nx)
        u, _ = self.action("u", nu, lb=a_bnd[0], ub=a_bnd[1])
        s, _, _ = self.variable("s", (nx, N), lb=0)
        self.set_dynamics(lambda x, u: A @ x + B * u + b, n_in=2, n_out=1)
        self.constraint("x_lb", x_bnd[0] + x_lb - s, "<=", x[:, 1:])
        self.constraint("x_ub", x[:, 1:], "<=", x_bnd[1] + x_ub + s)
        A_init, B_init = self.learnable_pars_init["A"], self.learnable_pars_init["B"]
        S = cs.DM(dlqr(A_init, B_init, 0.5 * np.eye(nx), 0.25 * np.eye(nu))[1])
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize(
            V0.T @ x[:, 0]  # to have a derivative, V0 must be a function of the state
            + cs.bilin(S, x[:, -1])
            + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
            + 0.5
            * cs.sum2(
                gammapowers
                * (cs.sum1(x[:, :-1] ** 2) + 0.5 * cs.sum1(u**2) + w.T @ s)
            )
        )
        opts = {
            "expand": True,
            "print_time": False,
            "ipopt": {"max_iter": 500, "sb": "yes", "print_level": 0},
        }
        self.init_solver(opts, solver="ipopt")


class CstrEnv(gym.Env[npt.NDArray[np.floating], float]):
    ns = 4  # number of states
    na = 1  # number of inputs
    reactor_temperature_bound = (100, 150)
    inflow_bound = (5, 35)
    x0 = np.asarray([1.0, 1.0, 100.0, 100.0])  # initial state

    def __init__(self, constraint_violation_penalty: float = 2e1) -> None:
        super().__init__()
        self.constraint_violation_penalty = constraint_violation_penalty
        self.observation_space = Box(
            np.array([0, 0, -np.inf, -np.inf]), np.inf, (self.ns,), np.float64
        )
        self.action_space = Box(*self.inflow_bound, (self.na,), np.float64)
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
    mpc = Mpc[cs.SX](Nlp[cs.SX]("SX"), horizon)
    y_space, u_space = env.observation_space, env.action_space
    ny, nu = y_space.shape[0], u_space.shape[0]
    y, _ = mpc.state("y", ny, bound_initial=False)
    u, _ = mpc.action("u", nu, u_space.low[:, None], u_space.high[:, None])
    lb = np.concatenate([[0.0, 100.0], env.action_space.low])
    ub = np.concatenate([[1.0, 150.0], env.action_space.high])
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
    b = mpc.parameter("backoff")
    _, _, slack_lb = mpc.constraint("TR_lb", y[1, :], ">=", 100.0 + b, soft=True)
    _, _, slack_ub = mpc.constraint("TR_ub", y[1, :], "<=", 150.0 - b, soft=True)
    mpc.minimize(
        env.VR * env.tf * cs.sum2(y[0, :-1] * u)
        + env.constraint_violation_penalty * cs.sum2(slack_lb + slack_ub)
    )
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
    prefers_dict = False  # ask and tell methods deal with arrays, not dicts

    def __init__(
        self, initial_random: int = 5, seed: Optional[int] = None, **kwargs: Any
    ) -> None:
        if initial_random <= 0:
            raise ValueError("`initial_random` must be positive.")
        super().__init__(**kwargs)
        self._initial_random = initial_random
        self._seed = seed

    def _init_update_solver(self) -> None:
        pars = self.learnable_parameters
        values = pars.value
        lb, ub = (values + bnd for bnd in self._get_update_bounds(values))
        lhs = LatinHypercube(pars.size, seed=self._seed)
        self._train_inputs = lhs.random(self._initial_random) * (ub - lb) + lb
        self._train_targets = np.empty((0,))  # we dont know the targets yet

    def ask(self) -> tuple[npt.NDArray[np.floating], None]:
        iteration = self._train_targets.shape[0]
        if iteration < self._initial_random:
            return self._train_inputs[iteration], None
        train_inputs = torch.from_numpy(self._train_inputs)
        train_targets = standardize(torch.from_numpy(self._train_targets).unsqueeze(-1))
        values = self.learnable_parameters.value
        bounds = torch.from_numpy(
            np.stack([values + bnd for bnd in self._get_update_bounds(values)])
        )
        normalize = Normalize(train_inputs.shape[-1], bounds=bounds)
        gp = SingleTaskGP(train_inputs, train_targets, input_transform=normalize)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        acqfun = ExpectedImprovement(gp, train_targets.amin(), maximize=False)
        acqfun_optimizer = optimize_acqf(
            acqfun, bounds, 1, 32, 128, {"seed": self._seed + iteration}
        )[0].numpy()
        self._train_inputs = np.append(self._train_inputs, acqfun_optimizer, axis=0)
        return acqfun_optimizer.reshape(-1), None

    def tell(self, values: npt.NDArray[np.floating], objective: float) -> None:
        iteration = self._train_targets.size
        assert (values == self._train_inputs[iteration]).all()
        self._train_targets = np.append(self._train_targets, objective)


DATA = loadmat(f"tests/data_test_examples_{platform}.mat", squeeze_me=True)


class TestExamples(unittest.TestCase):
    @parameterized.expand([(False,), (True,)])
    def test_q_learning__with_copy_and_pickle(self, use_copy: bool):
        mpc = QLearningLinearMpc()
        learnable_pars = LearnableParametersDict[cs.SX](
            (
                LearnableParameter(
                    name=name, shape=val.shape, value=val, sym=mpc.parameters[name]
                )
                for name, val in mpc.learnable_pars_init.items()
            )
        )
        env = MonitorEpisodes(TimeLimit(LtiSystem(), max_episode_steps=100))
        agent = Log(
            RecordUpdates(
                LstdQLearningAgent(
                    update_strategy=1,
                    mpc=mpc,
                    learnable_parameters=learnable_pars,
                    discount_factor=mpc.discount_factor,
                    optimizer=NetwonMethod(
                        learning_rate=5e-2,
                        max_percentage_update=1e3,  # does nothing; allows to test qp
                    ),
                    hessian_type="approx",
                    record_td_errors=True,
                )
            ),
            level=logging.DEBUG,
            log_frequencies={"on_timestep_end": 100},
        )

        agent_copy = agent.copy()
        if use_copy:
            agent = agent_copy
        J = agent.train(env=env, episodes=1, seed=69).item()
        agent = pickle.loads(pickle.dumps(agent))

        X = env.observations[0].squeeze().T
        U = env.actions[0].squeeze()
        R = env.rewards[0]
        TD = np.squeeze(agent.td_errors)
        parnames = ["V0", "x_lb", "x_ub", "b", "f", "A", "B"]
        PARS = np.concatenate(
            [np.reshape(agent.updates_history[n], -1) for n in parnames]
        )

        # from scipy.io import savemat
        # DATA.update({
        #     "ql_J": J,
        #     "ql_X": X,
        #     "ql_U": U,
        #     "ql_R": R,
        #     "ql_TD": TD,
        #     "ql_pars": PARS,
        # })
        # savemat(f"tests/data_test_examples_{platform}.mat", DATA)

        np.testing.assert_allclose(J, DATA["ql_J"], rtol=1e0, atol=1e0)
        np.testing.assert_allclose(X, DATA["ql_X"], rtol=1e0, atol=1e0)
        np.testing.assert_allclose(U, DATA["ql_U"], rtol=1e0, atol=1e0)
        np.testing.assert_allclose(R, DATA["ql_R"], rtol=1e1, atol=1e1)
        np.testing.assert_allclose(TD, DATA["ql_TD"], rtol=1e1, atol=1e1)
        np.testing.assert_allclose(PARS, DATA["ql_pars"], rtol=1e0, atol=1e0)

    @parameterized.expand([(False,), (True,)])
    def test_dpg__with_copy_and_pickle(self, use_copy: bool):
        mpc = DpgLinearMpc()
        learnable_pars = LearnableParametersDict[cs.SX](
            (
                LearnableParameter(
                    name=name,
                    shape=np.prod(val.shape),
                    value=val.flatten(order="F"),
                    sym=cs.vec(mpc.parameters[name]),
                )
                for name, val in mpc.learnable_pars_init.items()
            )
        )
        env = MonitorEpisodes(TimeLimit(LtiSystem(), max_episode_steps=200))
        rollout_length = 50
        agent = Log(
            RecordUpdates(
                LstdDpgAgent(
                    mpc=mpc,
                    learnable_parameters=learnable_pars,
                    discount_factor=mpc.discount_factor,
                    optimizer=GradientDescent(
                        learning_rate=1e-7,
                        max_percentage_update=1e3,  # does nothing; allows to test qp
                    ),
                    update_strategy=UpdateStrategy(rollout_length, "on_timestep_end"),
                    rollout_length=rollout_length,
                    exploration=E.GreedyExploration(0.05),
                    record_policy_performance=True,
                    record_policy_gradient=True,
                )
            ),
            level=logging.DEBUG,
            log_frequencies={"on_timestep_end": 200},
        )

        agent_copy = agent.copy()
        if use_copy:
            agent = agent_copy
        J = agent.train(env=env, episodes=1, seed=69).item()
        agent = pickle.loads(pickle.dumps(agent))

        X = env.observations[0].squeeze().T
        U = env.actions[0].squeeze()
        R = env.rewards[0]
        Jest = np.asarray(agent.policy_performances)
        Gest = np.asarray(agent.policy_gradients)
        parnames = ["V0", "x_lb", "x_ub", "b", "f", "A", "B"]
        PARS = np.concatenate(
            [np.reshape(agent.updates_history[n], -1) for n in parnames]
        )

        # from scipy.io import savemat
        # DATA.update({
        #     "dpg_J": J,
        #     "dpg_X": X,
        #     "dpg_U": U,
        #     "dpg_R": R,
        #     "dpg_Jest": Jest,
        #     "dpg_Gest": Gest,
        #     "dpg_pars": PARS,
        # })
        # savemat(f"tests/data_test_examples_{platform}.mat", DATA)

        np.testing.assert_allclose(J, DATA["dpg_J"], rtol=1e0, atol=1e0)
        np.testing.assert_allclose(X, DATA["dpg_X"], rtol=1e0, atol=1e0)
        np.testing.assert_allclose(U, DATA["dpg_U"], rtol=1e1, atol=1e1)
        np.testing.assert_allclose(R, DATA["dpg_R"], rtol=1e1, atol=1e1)
        np.testing.assert_allclose(Jest, DATA["dpg_Jest"], rtol=1e0, atol=1e0)
        np.testing.assert_allclose(Gest, DATA["dpg_Gest"], rtol=1e3, atol=1e3)
        np.testing.assert_allclose(PARS, DATA["dpg_pars"], rtol=1e0, atol=1e0)

    @parameterized.expand([(False,), (True,)])
    def test_bayesopt__with_copy_and_pickle(self, use_copy: bool):
        torch.manual_seed(0)
        env = MonitorEpisodes(TimeLimit(CstrEnv(), max_episode_steps=40))
        env = TransformReward(env, neg)
        env = NoisyFilterObservation(env, [1, 2])
        mpc = get_cstr_mpc(env)
        pars = mpc.parameters
        learnable_pars = LearnableParametersDict[cs.SX](
            (
                LearnableParameter(n, pars[n].shape, (ub + lb) / 2, lb, ub, pars[n])
                for n, lb, ub in [("narx_weights", -2, 2), ("backoff", 0, 5)]
            )
        )
        agent = GlobOptLearningAgent(
            mpc=mpc,
            learnable_parameters=learnable_pars,
            optimizer=BoTorchOptimizer(initial_random=2, seed=42),
        )
        agent = RecordUpdates(agent)
        agent_copy = agent.copy()
        if use_copy:
            agent = agent_copy

        with catch_warnings():
            J = agent.train(env=env, episodes=5, seed=69, raises=False)
        agent = pickle.loads(pickle.dumps(agent))

        X = np.squeeze(env.observations)
        U = np.squeeze(env.actions, (2, 3))
        R = np.squeeze(env.rewards)

        # from scipy.io import savemat
        # DATA.update({"bo_J": J, "bo_X": X, "bo_U": U, "bo_R": R})
        # savemat(f"tests/data_test_examples_{platform}.mat", DATA)

        np.testing.assert_allclose(J, DATA["bo_J"], rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(X, DATA["bo_X"], rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(U, DATA["bo_U"], rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(R, DATA["bo_R"], rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
