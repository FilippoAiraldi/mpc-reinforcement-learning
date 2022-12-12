import pickle
import unittest
from functools import lru_cache
from itertools import product
from unittest.mock import MagicMock, Mock, call

import casadi as cs
import numpy as np
from csnlp import MultistartNlp, Nlp, Solution, scaling
from csnlp.wrappers import Mpc, NlpScaling
from parameterized import parameterized, parameterized_class
from scipy import io as matio

from mpcrl import Agent
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
    x, u, d = cs.SX.sym("x", 3), cs.SX.sym("u", 2), cs.SX.sym("d", 3)
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
    nlp = NlpScaling[cs.MX](nlp, scaler=scaler)
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

    @parameterized.expand([(False,), (True,)])
    def test_evaluate(self, action_expr_: bool = True):
        rewards = np.random.randn(2)
        states = [object(), object(), object()]
        env = MagicMock()
        env.reset = Mock(return_value=(states[0], {}))
        env.step = Mock(
            side_effect=[
                (states[1], rewards[0], False, False, {}),
                (states[2], rewards[1], True, False, {}),
            ]
        )
        if action_expr_:
            actions = [object(), object()]
            get_value = Mock(side_effect=actions)
            sol = Solution(0, {}, {}, {}, get_value)
            action_expr = object()
        else:
            action = np.full((1, 1), object(), dtype=object)
            sol = Solution(0, {}, {"u": action}, {}, None)
            action_expr = None
        deterministic = object()
        agent = Agent(mpc=get_mpc(3, False))
        agent.state_value = Mock(return_value=sol)

        returns = agent.evaluate(
            env, 1, deterministic=deterministic, seed=69, action_expr=action_expr
        )

        np.testing.assert_array_equal(returns, np.asarray([rewards.sum()]))
        env.reset.assert_called_once()
        if action_expr_:
            get_value.assert_has_calls([call(action_expr, eval=True) for _ in range(2)])
            env.step.assert_has_calls([call(action) for action in actions])
        else:
            env.step.assert_has_calls([call(action) for _ in range(2)])



if __name__ == "__main__":
    unittest.main()
