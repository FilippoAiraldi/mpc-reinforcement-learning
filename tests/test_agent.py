import pickle
import unittest
from functools import lru_cache
from itertools import product
from unittest.mock import MagicMock, Mock, call
from warnings import catch_warnings

import casadi as cs
import numpy as np
from csnlp import MultistartNlp, Nlp, Solution, scaling
from csnlp.wrappers import Mpc, NlpScaling
from parameterized import parameterized, parameterized_class
from scipy import io as matio

from mpcrl import (
    Agent,
    ExperienceReplay,
    LearnableParametersDict,
    LearningAgent,
    MpcSolverWarning,
)
from mpcrl import exploration as E
from mpcrl import schedulers as S

OPTS = {
    "expand": True,
    "print_time": False,
    "ipopt": {
        "max_iter": 500,
        "sb": "yes",
        # for debugging
        "print_level": 0,
        "print_user_options": "no",
        "print_options_documentation": "no",
    },
}
RESULTS = matio.loadmat(r"tests/tests_data.mat")


@lru_cache
def get_dynamics(g: float, alpha: float, dt: float) -> cs.Function:
    x, u, d = cs.MX.sym("x", 3), cs.MX.sym("u", 2), cs.MX.sym("d", 3)
    x_next = x + cs.vertcat(x[1], u[0] / x[2] - g, -alpha * u[0]) * dt + d * 0
    return cs.Function("F", [x, u, d], [x_next], ["x", "u", "d"], ["x+"])


def get_mpc(horizon: int, multistart: bool):
    N = horizon
    T = 100
    K = 3
    dt = T / N
    yT = 100000
    g = 9.81
    alpha = 1 / (300 * g)
    y_nom = 1e5
    v_nom = 2e3
    m_nom = 3e5
    u_nom = 1e8
    scaler = scaling.Scaler()
    scaler.register("y", scale=y_nom)
    scaler.register("y_0", scale=y_nom)
    scaler.register("v", scale=v_nom)
    scaler.register("v_0", scale=v_nom)
    scaler.register("m", scale=m_nom)
    scaler.register("m_0", scale=m_nom)
    scaler.register("u1", scale=u_nom)
    scaler.register("u2", scale=u_nom)
    nlp = (
        MultistartNlp[cs.MX](sym_type="MX", starts=K)
        if multistart
        else Nlp[cs.MX](sym_type="MX")
    )
    nlp = NlpScaling[cs.MX](nlp, scaler=scaler, warns=False)
    mpc = Mpc[cs.MX](nlp, prediction_horizon=N)
    y, _ = mpc.state("y")
    _, _ = mpc.state("v")
    m, _ = mpc.state("m", lb=0)
    mpc.action("u1", lb=0, ub=5e7)
    u2, _ = mpc.action("u2", lb=0, ub=5e7)
    mpc.disturbance("d", 3)
    mpc.set_dynamics(get_dynamics(g, alpha, dt))
    mpc.constraint("yT", y[-1], "==", yT)
    mpc.minimize(m[0] - m[-1] + cs.sum2(u2))
    mpc.init_solver(OPTS)
    return mpc


class DummyLearningAgent(LearningAgent):
    def update(self, *args, **kwargs):
        return

    def train(self, *args, **kwargs):
        return


@parameterized_class("multistart_nlp", [(True,), (False,)])
class TestAgent(unittest.TestCase):
    def test_init__raises__mpc_with_no_actions(self):
        with self.assertRaisesRegex(
            ValueError, "Expected Mpc with na>0; got na=0 instead."
        ):
            Agent(mpc=Mpc(Nlp(), 4))

    def test_init__instantiates_V_and_Q_as_two_different_mpcs(self):
        agent = Agent(mpc=get_mpc(3, self.multistart_nlp))
        self.assertIsInstance(agent.Q, Mpc)
        self.assertIsInstance(agent.V, Mpc)
        self.assertIsNot(agent.Q, agent.V)

    def test_init__instantiates_V_and_Q_correctly(self):
        agent = Agent(mpc=get_mpc(3, self.multistart_nlp))
        self.assertIn(agent.cost_perturbation_parameter, agent.V.parameters.keys())
        self.assertNotIn(agent.cost_perturbation_parameter, agent.Q.parameters.keys())
        self.assertIn(agent.init_action_parameter, agent.Q.parameters.keys())
        self.assertIn(agent.init_action_constraint, agent.Q.constraints.keys())
        self.assertNotIn(agent.init_action_parameter, agent.V.parameters.keys())
        self.assertNotIn(agent.init_action_constraint, agent.V.constraints.keys())

    def test_unwrapped(self):
        agent = Agent(mpc=get_mpc(3, self.multistart_nlp))
        agent2 = agent.unwrapped
        self.assertIs(agent, agent2)

    @parameterized.expand([(True,), (False,)])
    def test__is_deepcopyable_and_pickleable(self, copy: bool):
        epsilon, epsilon_decay_rate = 0.7, 0.75
        strength, strength_decay_rate = 0.5, 0.75
        epsilon_scheduler = S.ExponentialScheduler(epsilon, epsilon_decay_rate)
        strength_scheduler = S.ExponentialScheduler(strength, strength_decay_rate)
        exploration = E.EpsilonGreedyExploration(
            epsilon=epsilon_scheduler, strength=strength_scheduler, seed=42
        )
        agent1 = Agent(mpc=get_mpc(3, self.multistart_nlp), exploration=exploration)

        if copy:
            agent2 = agent1.copy()
        else:
            with agent1.pickleable():
                agent2: Agent = pickle.loads(pickle.dumps(agent1))

        self.assertIsNot(agent1, agent2)
        self.assertIsNot(agent1.Q, agent2.Q)
        self.assertIsNot(agent1.V, agent2.V)
        self.assertIs(agent1.exploration, exploration)
        exp1, exp2 = agent1.exploration, agent2.exploration
        self.assertIsNot(exp1, exp2)
        self.assertIsNot(exp1, exp2)
        for scheduler in ["epsilon_scheduler", "strength_scheduler"]:
            sc1, sc2 = getattr(exp1, scheduler), getattr(exp2, scheduler)
            self.assertIsNot(sc1, sc2)
            self.assertEqual(sc1.value, sc2.value)
            self.assertEqual(sc1.factor, sc2.factor)

    @parameterized.expand(product(["V", "Q"], [False, True], [False, True]))
    def test_solve_mpc__calls_mpc_with_correct_args(
        self, mpctype: str, vector: bool, multiple_pars: bool
    ):
        # sourcery skip: low-code-quality
        starts = 3
        horizon = 3
        if not self.multistart_nlp:
            multiple_pars = False
        fixed_pars = {
            Agent.cost_perturbation_parameter: [42, 69],
            "d": cs.DM([5, 6, 7]),
        }
        fixed_pars_ = (
            (fixed_pars.copy() for _ in range(starts))
            if multiple_pars
            else fixed_pars.copy()
        )
        mpc = get_mpc(horizon, self.multistart_nlp)
        agent = Agent(mpc=mpc, fixed_parameters=fixed_pars_)
        mpc: Mpc[cs.SX] = getattr(agent, mpctype)

        s = {"y": 0, "v": 10, "m": 5e5}
        a = {"u1": 1, "u2": 2}
        if vector:
            s = cs.DM(s.values())
            a = cs.DM(a.values())
        if mpctype == "V":
            a = None

        pert = cs.DM([65, 79])
        vals0 = object()
        sol = Solution(
            f=5, vars=None, vals=vals0, stats={"success": True}, _get_value=None
        )
        agent._last_solution = sol
        if self.multistart_nlp:
            mpc.nlp.solve_multi = MagicMock(return_value=sol)
        else:
            mpc.nlp.solve = MagicMock(return_value=sol)

        agent.solve_mpc(mpc, state=s, action=a, perturbation=pert)

        call_pars = {
            **fixed_pars,
            "y_0": s[0] if vector else s["y"],
            "v_0": s[1] if vector else s["v"],
            "m_0": s[2] if vector else s["m"],
        }
        if mpctype == "V":
            call_pars[Agent.cost_perturbation_parameter] = pert
        else:
            call_pars[Agent.init_action_parameter] = a if vector else cs.DM(a.values())
        if self.multistart_nlp:
            mpc.nlp.solve_multi.assert_called_once()
            kwargs = mpc.nlp.solve_multi.call_args.kwargs
        else:
            mpc.nlp.solve.assert_called_once()
            kwargs = mpc.nlp.solve.call_args.kwargs
        self.assertIs(kwargs["vals0"], vals0)
        if multiple_pars:
            for pars_i in kwargs["pars"]:
                self.assertEqual(len(mpc.unwrapped._pars.keys() - pars_i.keys()), 0)
                for key in call_pars:
                    np.testing.assert_allclose(pars_i[key], call_pars[key], rtol=0)
        else:
            pars = kwargs["pars"]
            self.assertEqual(len(mpc.unwrapped._pars.keys() - pars.keys()), 0)
            for key in call_pars:
                np.testing.assert_allclose(pars[key], call_pars[key], rtol=0)

    @parameterized.expand(product((False, True), (False, True)))
    def test_state_value__computes_right_solution(
        self, vector: bool, deterministic: bool
    ):  # sourcery skip: move-assign
        starts = 3
        horizon = 100
        fixed_pars = {"d": cs.DM([5, 6, 7])}
        if self.multistart_nlp:
            fixed_pars = [fixed_pars.copy() for _ in range(starts)]

        exploration = E.GreedyExploration(0)
        mpc = get_mpc(horizon, self.multistart_nlp)
        agent = Agent(mpc=mpc, fixed_parameters=fixed_pars, exploration=exploration)

        state = {"y": 0, "v": 0, "m": 5e5}
        vals0 = {"y": 0, "v": 0, "m": 5e5, "u1": 1e8, "u2": 0}
        agent._last_solution = Solution(f=0, vars=0, vals=vals0, stats=0, _get_value=0)
        if vector:
            state = cs.DM(state.values())

        sol = agent.state_value(state=state, vals0=None, deterministic=deterministic)
        self.assertTrue(sol.success)
        np.testing.assert_allclose(sol.f, RESULTS["state_value_f"].item(), rtol=1e-3)
        np.testing.assert_allclose(
            sol.vals["u1"],
            RESULTS["state_value_us"],
            rtol=1e-7,
            atol=1e-7,
        )

    @parameterized.expand(product((False, True), (False, True)))
    def test_action_value__computes_right_solution(self, vector: bool, a_optimal: bool):
        # sourcery skip: move-assign
        starts = 3
        horizon = 100
        fixed_pars = {"d": cs.DM([5, 6, 7])}
        if self.multistart_nlp:
            fixed_pars = [fixed_pars.copy() for _ in range(starts)]

        exploration = E.GreedyExploration(0)
        mpc = get_mpc(horizon, self.multistart_nlp)
        agent = Agent(mpc=mpc, fixed_parameters=fixed_pars, exploration=exploration)

        a_opt, a_subopt = 5e7, 1.42e7
        state = {"y": 0, "v": 0, "m": 5e5}
        action = {"u1": a_opt if a_optimal else a_subopt, "u2": 0}
        vals0 = {**state, **action}
        agent._last_solution = Solution(f=0, vars=0, vals=vals0, stats=0, _get_value=0)
        if vector:
            state = cs.DM(state.values())
            action = cs.DM(action.values())

        sol = agent.action_value(state=state, action=action, vals0=None)
        self.assertTrue(sol.success)
        u1_0_0 = sol.value(mpc.unscaled_variables["u1"][0, 0])
        if a_optimal:
            np.testing.assert_allclose(u1_0_0, a_opt)
            np.testing.assert_allclose(
                sol.vals["u1"],
                RESULTS["state_value_us"],
                rtol=1e-7,
                atol=1e-7,
            )
        else:
            np.testing.assert_allclose(u1_0_0, a_subopt)
            self.assertTrue(sol.f >= RESULTS["state_value_f"].item())

    def test_evaluate__performs_correct_calls(self):
        seed = 69
        episodes = 3
        episode_length = 10
        Ttot = episodes * episode_length
        reset_states = [object() for _ in range(episodes)]
        reset_options = object()
        step_states = [object() for _ in range(Ttot)]
        rewards = np.random.randn(Ttot)
        truncateds = [False] * Ttot
        terminateds = np.full(Ttot, 0, dtype=bool)
        terminateds[::-episode_length] = True
        infos = [{}] * Ttot
        env = MagicMock()
        env.reset = Mock(side_effect=zip(reset_states, infos))
        env.step = Mock(
            side_effect=zip(step_states, rewards, truncateds, terminateds, infos)
        )
        actions1 = np.random.randn(Ttot, 2, 1)
        actions2 = np.random.randn(Ttot, 3, 1)
        successes = [True] * (Ttot - 1) + [False]
        sols = map(
            lambda u1, u2, success: Solution(
                0,
                {},
                {"u1": u1, "u2": u2},
                {"success": success, "return_status": "bad"},
                None,
            ),
            actions1,
            actions2,
            successes,
        )
        agent = Agent(mpc=get_mpc(3, False))
        agent.state_value = Mock(side_effect=sols)
        deterministic = object()

        with catch_warnings(record=True) as cw:
            returns = agent.evaluate(
                env,
                episodes,
                deterministic=deterministic,
                seed=seed,
                raises=False,
                env_reset_options=reset_options,
            )

        np.testing.assert_allclose(returns, rewards.reshape(-1, episode_length).sum(1))
        env.reset.assert_has_calls(
            [call(seed=seed + i, options=reset_options) for i in range(episodes)]
        )
        for mcall, u1, u2 in zip(env.step.mock_calls, actions1, actions2):
            self.assertEqual(len(mcall.args), 1)
            np.testing.assert_array_equal(mcall.args[0], cs.vertcat(u1, u2))
        states = np.column_stack(
            (reset_states, np.asarray(step_states).reshape(-1, episode_length))
        )[:, :-1].flatten()
        agent.state_value.assert_has_calls(
            [call(state, deterministic) for state in states]
        )
        self.assertEqual(len(cw), 1)
        self.assertIs(cw[0].category, MpcSolverWarning)


class TestLearningAgent(unittest.TestCase):
    def test__init__creates_default_experience_and_exploration(self):
        learnable_parameters = LearnableParametersDict()
        agent = DummyLearningAgent(
            mpc=get_mpc(3, False), learnable_parameters=learnable_parameters
        )
        self.assertIsInstance(agent.exploration, E.ExplorationStrategy)
        self.assertIsInstance(agent.experience, ExperienceReplay)
        self.assertIs(agent.learnable_parameters, learnable_parameters)

    def test__store_experience__appends_experience(self):
        experience = MagicMock()
        experience.append = Mock()
        item = object()
        agent = DummyLearningAgent(
            mpc=get_mpc(3, False), learnable_parameters={}, experience=experience
        )
        self.assertIs(agent.experience, experience)
        agent.store_experience(item)
        experience.append.assert_called_once_with(item)


if __name__ == "__main__":
    unittest.main()
