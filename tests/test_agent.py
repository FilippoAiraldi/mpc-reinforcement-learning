import pickle
import unittest
from functools import lru_cache
from itertools import product
from unittest.mock import MagicMock, Mock, call
from warnings import catch_warnings

import casadi as cs
import numpy as np
from csnlp import Nlp, scaling
from csnlp.core.solutions import EagerSolution
from csnlp.multistart import StackedMultistartNlp
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
from mpcrl.util.seeding import mk_seed

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
RESULTS = matio.loadmat(r"tests/data_test_agent.mat")


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
        StackedMultistartNlp[cs.MX](sym_type="MX", starts=K)
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
    mpc.set_nonlinear_dynamics(get_dynamics(g, alpha, dt))
    mpc.constraint("yT", y[-1], "==", yT)
    mpc.minimize(m[0] - m[-1] + cs.sum2(u2))
    mpc.init_solver(OPTS)
    return mpc


class DummyLearningAgent(LearningAgent):
    def update(self, *args, **kwargs):
        return

    def train_one_episode(self, *args, **kwargs):
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

    @parameterized.expand([(None,), ("gradient-based",), ("additive",)])
    def test_init__instantiates_V_and_Q_correctly(self, exploration_mode: str):
        if exploration_mode is None:
            exploration = E.NoExploration()
        else:
            exploration = E.GreedyExploration(0.5, mode=exploration_mode)
        agent = Agent(mpc=get_mpc(3, self.multistart_nlp), exploration=exploration)

        V_pars_keys = agent.V.parameters.keys()
        Q_pars_keys = agent.Q.parameters.keys()
        if exploration_mode is None or exploration_mode == "additive":
            self.assertNotIn(agent.cost_perturbation_parameter, V_pars_keys)
        else:
            self.assertIn(agent.cost_perturbation_parameter, V_pars_keys)
        self.assertNotIn(agent.cost_perturbation_parameter, Q_pars_keys)
        self.assertIn(agent.init_action_parameter, Q_pars_keys)
        self.assertIn(agent.init_action_constraint, agent.Q.constraints.keys())
        self.assertNotIn(agent.init_action_parameter, V_pars_keys)
        self.assertNotIn(agent.init_action_constraint, agent.V.constraints.keys())

    def test_init__removes_bounds_on_initial_action_in_Q_correctly(self):
        mpc = get_mpc(3, self.multistart_nlp)
        mpc.unwrapped.remove_variable_bounds = Mock()
        agent = Agent(mpc=mpc, remove_bounds_on_initial_action=True)
        self.assertFalse(agent.V.remove_variable_bounds.called)
        calls = agent.Q.remove_variable_bounds.call_args_list
        self.assertEqual(len(calls), mpc.na)
        for i, mcall in enumerate(calls):
            expected_name = f"u{i + 1}"
            na_ = mpc.actions[expected_name].size1()
            name, direction, idx = mcall.args
            self.assertEqual(name, expected_name)
            self.assertEqual(direction, "both")
            self.assertEqual(list(idx), [(r, 0) for r in range(na_)])

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
        agent1 = Agent(mpc=get_mpc(3, self.multistart_nlp))
        agent1._exploration = exploration  # a bit of cheating

        if copy:
            agent2 = agent1.copy()
        else:
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
        starts = 3
        horizon = 3
        multiple_pars &= self.multistart_nlp
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
        exploration = E.GreedyExploration(0.5, mode="gradient-based")
        agent = Agent(mpc=mpc, fixed_parameters=fixed_pars_, exploration=exploration)
        mpc: Mpc[cs.SX] = getattr(agent, mpctype)

        s = {"y": 0, "v": 10, "m": 5e5}
        if vector:
            s = cs.DM(s.values())
        if mpctype == "V":
            a = None
        else:
            a = {"u1": 1, "u2": 2}
            if vector:
                a = cs.DM(a.values())

        pert = cs.DM([65, 79])
        vals0 = object()
        sol = EagerSolution(
            f=None,
            p_sym=None,
            p=None,
            x_sym=None,
            x=None,
            lam_g_and_h_sym=None,
            lam_g_and_h=None,
            lam_lbx_and_ubx_sym=None,
            lam_lbx_and_ubx=None,
            vars=None,
            vals=vals0,
            dual_vars=None,
            dual_vals=None,
            stats={"success": True},
            solver_plugin=None,
        )
        agent._last_solution = sol
        method = "solve_multi" if self.multistart_nlp else "solve"
        setattr(mpc.nlp, method, MagicMock(return_value=sol))

        agent._solve_mpc(mpc, state=s, action=a, perturbation=pert)

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

        overwritten_method = getattr(mpc.nlp, method)
        overwritten_method.assert_called_once()
        call_args = overwritten_method.call_args.args
        actual_call_pars, actual_call_vals0 = call_args
        if multiple_pars:
            for pars_i in actual_call_pars:
                self.assertEqual(len(mpc.unwrapped._pars.keys() - pars_i.keys()), 0)
                for key in call_pars:
                    np.testing.assert_allclose(pars_i[key], call_pars[key], rtol=0)
        else:
            pars = actual_call_pars
            self.assertEqual(len(mpc.unwrapped._pars.keys() - pars.keys()), 0)
            for key in call_pars:
                np.testing.assert_allclose(pars[key], call_pars[key], rtol=0)
        self.assertIs(actual_call_vals0, vals0)

    @parameterized.expand(product((False, True), (False, True)))
    def test_state_value__computes_right_solution(
        self, vector: bool, deterministic: bool
    ):
        starts = 3
        horizon = 100
        fixed_pars = {"d": cs.DM([5, 6, 7])}
        if self.multistart_nlp:
            fixed_pars = [fixed_pars.copy() for _ in range(starts)]

        mpc = get_mpc(horizon, self.multistart_nlp)
        agent = Agent(mpc=mpc, fixed_parameters=fixed_pars)

        state = {"y": 0, "v": 0, "m": 5e5}
        vals0 = {"y": 0, "v": 0, "m": 5e5, "u1": 1e8, "u2": 0}
        agent._last_solution = EagerSolution(
            f=0.0,
            p_sym=None,
            p=None,
            x_sym=None,
            x=None,
            lam_g_and_h_sym=None,
            lam_g_and_h=None,
            lam_lbx_and_ubx_sym=None,
            lam_lbx_and_ubx=None,
            vars=None,
            vals=vals0,
            dual_vars=None,
            dual_vals=None,
            stats=None,
            solver_plugin=None,
        )
        if vector:
            state = cs.DM(state.values())

        action, sol = agent.state_value(
            state=state, vals0=None, deterministic=deterministic
        )
        self.assertTrue(sol.success)
        np.testing.assert_array_equal(
            action, cs.vertcat(sol.vals["u1"][:, 0], sol.vals["u2"][:, 0])
        )
        np.testing.assert_allclose(sol.f, RESULTS["state_value_f"].item(), rtol=1e-3)
        np.testing.assert_allclose(
            sol.vals["u1"],
            RESULTS["state_value_us"],
            rtol=1e-7,
            atol=1e-7,
        )

    @parameterized.expand(product((False, True), (False, True)))
    def test_action_value__computes_right_solution(self, vector: bool, a_optimal: bool):
        starts = 3
        horizon = 100
        fixed_pars = {"d": cs.DM([5, 6, 7])}
        if self.multistart_nlp:
            fixed_pars = [fixed_pars.copy() for _ in range(starts)]

        mpc = get_mpc(horizon, self.multistart_nlp)
        agent = Agent(mpc=mpc, fixed_parameters=fixed_pars)

        a_opt, a_subopt = 5e7, 1.42e7
        state = {"y": 0, "v": 0, "m": 5e5}
        action = {"u1": a_opt if a_optimal else a_subopt, "u2": 0}
        vals0 = {**state, **action}
        agent._last_solution = EagerSolution(
            f=None,
            p_sym=None,
            p=None,
            x_sym=None,
            x=None,
            lam_g_and_h_sym=None,
            lam_g_and_h=None,
            lam_lbx_and_ubx_sym=None,
            lam_lbx_and_ubx=None,
            vars=None,
            vals=vals0,
            dual_vars=None,
            dual_vals=None,
            stats=None,
            solver_plugin=None,
        )
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
        actions = [cs.vertcat(u1, u2) for u1, u2 in zip(actions1, actions2)]
        successes = [True] * (Ttot - 1) + [False]
        sols = map(
            lambda u1, u2, success: EagerSolution(
                f=None,
                p_sym=None,
                p=None,
                x_sym=None,
                x=None,
                lam_g_and_h_sym=None,
                lam_g_and_h=None,
                lam_lbx_and_ubx_sym=None,
                lam_lbx_and_ubx=None,
                vars=None,
                vals={"u1": u1, "u2": u2},
                dual_vars=None,
                dual_vals=None,
                stats={"success": success, "return_status": "bad"},
                solver_plugin=None,
            ),
            actions1,
            actions2,
            successes,
        )
        agent = Agent(mpc=get_mpc(3, False))
        agent.state_value = Mock(side_effect=zip(actions, sols))
        deterministic = object()

        with catch_warnings(record=True) as cw:
            returns = Agent.evaluate(
                agent,
                env,
                episodes,
                deterministic=deterministic,
                seed=seed,
                raises=False,
                env_reset_options=reset_options,
            )

        np.testing.assert_allclose(returns, rewards.reshape(-1, episode_length).sum(1))
        rng = np.random.default_rng(seed)
        env.reset.assert_has_calls(
            [call(seed=mk_seed(rng), options=reset_options) for i in range(episodes)]
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
    def test_init__creates_default_experience_and_exploration(self):
        learnable_parameters = LearnableParametersDict()
        agent = DummyLearningAgent(
            mpc=get_mpc(3, False),
            learnable_parameters=learnable_parameters,
            update_strategy=1,
        )
        self.assertIsInstance(agent.exploration, E.ExplorationStrategy)
        self.assertIsInstance(agent.experience, ExperienceReplay)
        self.assertIs(agent.learnable_parameters, learnable_parameters)

    def test_store_experience__appends_experience(self):
        experience = MagicMock()
        experience.append = Mock()
        item = object()
        agent = DummyLearningAgent(
            mpc=get_mpc(3, False),
            learnable_parameters={},
            experience=experience,
            update_strategy=1,
        )
        self.assertIs(agent.experience, experience)
        agent.store_experience(item)
        experience.append.assert_called_once_with(item)


if __name__ == "__main__":
    unittest.main()
