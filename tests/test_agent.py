import os
import tempfile
import unittest
from unittest.mock import MagicMock
from itertools import product

import casadi as cs
import numpy as np
from csnlp import MultistartNlp, Nlp, Solution
from csnlp.util import io
from csnlp.util.scaling import Scaler
from csnlp.wrappers import Mpc, NlpScaling
from parameterized import parameterized, parameterized_class

from mpcrl.agents.agent import Agent

TMPFILENAME: str = ""
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


def get_dynamics(g: float, alpha: float, dt: float) -> cs.Function:
    x, u = cs.SX.sym("x", 3), cs.SX.sym("u", 2)
    x_next = x + cs.vertcat(x[1], u[0] / x[2] - g, -alpha * u[0]) * dt
    return cs.Function("F", [x, u], [x_next], ["x", "u"], ["x+"])


def get_mpc(multistart: bool):
    N = 10
    T = 100
    K = 3
    dt = T / N
    yT = 100000
    g = 9.81
    alpha = 1 / (300 * g)
    seed = 69
    nlp = (
        MultistartNlp(sym_type="MX", starts=K, seed=seed)
        if multistart
        else Nlp(sym_type="MX", seed=seed)
    )
    y_nom = 1e5
    v_nom = 2e3
    m_nom = 3e5
    u_nom = 1e8
    scaler = Scaler()
    scaler.register("y", scale=y_nom)
    scaler.register("y_0", scale=y_nom)
    scaler.register("v", scale=v_nom)
    scaler.register("v_0", scale=v_nom)
    scaler.register("m", scale=m_nom)
    scaler.register("m_0", scale=m_nom)
    scaler.register("u1", scale=u_nom)
    scaler.register("u2", scale=u_nom)
    nlp = NlpScaling(nlp, scaler=scaler, warns=False)
    mpc = Mpc(nlp, prediction_horizon=N)
    y, _ = mpc.state("y")
    _, _ = mpc.state("v")
    m, _ = mpc.state("m", lb=0)
    mpc.action("u1", lb=0, ub=5e7)
    mpc.action("u2", lb=0, ub=5e7)
    F = get_dynamics(g, alpha, dt)
    mpc.dynamics = F
    mpc.constraint("yT", y[-1], "==", yT)
    mpc.minimize(m[0] - m[-1])
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
        agent = Agent(mpc=get_mpc(self.multistart_nlp))
        self.assertIsInstance(agent.Q, Mpc)
        self.assertIsInstance(agent.V, Mpc)
        self.assertIsNot(agent.Q, agent.V)

    def test_init__instantiates_V_and_Q_correctly(self):
        agent = Agent(mpc=get_mpc(self.multistart_nlp))
        self.assertIn(Agent.cost_perturbation_par, agent.V.parameters.keys())
        self.assertNotIn(Agent.cost_perturbation_par, agent.Q.parameters.keys())
        self.assertIn(Agent.init_action_par, agent.Q.parameters.keys())
        self.assertIn(Agent.init_action_con, agent.Q.constraints.keys())
        self.assertNotIn(Agent.init_action_par, agent.V.parameters.keys())
        self.assertNotIn(Agent.init_action_con, agent.V.constraints.keys())

    def test_unwrapped(self):
        agent = Agent(mpc=get_mpc(self.multistart_nlp))
        agent2 = agent.unwrapped
        self.assertIs(agent, agent2)

    def test_np_random(self):
        agent = Agent(mpc=get_mpc(self.multistart_nlp))
        self.assertIsInstance(agent.np_random, np.random.Generator)
        self.assertIs(agent.np_random, agent.V.np_random)
        self.assertIsNot(agent.np_random, agent.Q.np_random)

    def test_copy(self):
        agent1 = Agent(mpc=get_mpc(self.multistart_nlp))
        agent2 = agent1.copy()
        self.assertIsNot(agent1, agent2)
        self.assertIsNot(agent1.Q, agent2.Q)
        self.assertIsNot(agent1.V, agent2.V)

    def test__is_pickleable(self):
        agent1 = Agent(mpc=get_mpc(self.multistart_nlp))

        global TMPFILENAME
        TMPFILENAME = next(tempfile._get_candidate_names())
        with agent1.pickleable():
            io.save(TMPFILENAME, agent=agent1, check=42)

        loadeddata = io.load(TMPFILENAME)
        self.assertEqual(loadeddata["check"], 42)
        agent2: Agent = loadeddata["agent"]
        self.assertIsInstance(agent2, Agent)
        self.assertIsInstance(agent2.Q, Mpc)
        self.assertIsInstance(agent2.V, Mpc)
        self.assertIsNot(agent1, agent2)
        self.assertIsNot(agent1.Q, agent2.Q)
        self.assertIsNot(agent1.V, agent2.V)

    def tearDown(self) -> None:
        try:
            os.remove(f"{TMPFILENAME}.pkl")
        finally:
            return super().tearDown()


if __name__ == "__main__":
    unittest.main()
