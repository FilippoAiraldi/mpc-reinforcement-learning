import unittest
from functools import lru_cache
from itertools import count
from unittest.mock import Mock, call

import casadi as cs
import numpy as np
from csnlp import Nlp, scaling
from csnlp.multistart import StackedMultistartNlp
from csnlp.wrappers import Mpc, NlpScaling

from mpcrl import (
    LearnableParameter,
    LearnableParametersDict,
    LearningAgent,
    wrappers_agents,
)
from mpcrl.wrappers.envs.monitor_infos import compact_dicts_into_one


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
        StackedMultistartNlp[cs.MX](sym_type="MX", starts=K)
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
    mpc.init_solver()
    return mpc


class DummyLearningAgent(LearningAgent):
    def update(self, *args, **kwargs):
        return

    def train_one_episode(self, *args, **kwargs):
        return


AGENT = DummyLearningAgent(
    mpc=get_mpc(3, False),
    learnable_parameters=LearnableParametersDict(),
    update_strategy=1,
)


class TestWrapperAndLearningWrapper(unittest.TestCase):
    def test_attr__raises__when_accessing_private_attrs(self):
        wrapped = wrappers_agents.LearningWrapper(AGENT)
        with self.assertRaisesRegex(
            AttributeError, "Accessing private attribute '_x' is prohibited."
        ):
            wrapped._x

    def test_unwrapped__unwraps_nlp_correctly(self):
        wrapped = wrappers_agents.LearningWrapper(AGENT)
        self.assertIs(AGENT, wrapped.unwrapped)

    def test_str_and_repr(self):
        wrapped = wrappers_agents.LearningWrapper(AGENT)
        S = wrapped.__str__()
        self.assertIn(wrappers_agents.LearningWrapper.__name__, S)
        self.assertIn(AGENT.__str__(), S)
        S = wrapped.__repr__()
        self.assertIn(wrappers_agents.LearningWrapper.__name__, S)
        self.assertIn(AGENT.__repr__(), S)

    def test_is_wrapped(self):
        self.assertFalse(AGENT.is_wrapped())

        wrapped = wrappers_agents.LearningWrapper(AGENT)
        self.assertTrue(wrapped.is_wrapped(wrappers_agents.LearningWrapper))
        self.assertFalse(wrapped.is_wrapped(cs.SX))


class TestRecordUpdates(unittest.TestCase):
    def test__update__records_learnable_parameters_correctly(self):
        P, K = 3, 10
        pars = np.random.rand(P, K + 1)
        parsdict = LearnableParametersDict(
            [LearnableParameter(f"p{i}", 1, pars[i, 0]) for i in range(P)]
        )
        iter = count(1)
        agent = AGENT.copy()
        agent._learnable_pars = parsdict
        agent.on_update = Mock(
            return_value=None,
            side_effect=lambda: agent.learnable_parameters.update_values(
                pars[:, next(iter)]
            ),
        )
        wrapped = wrappers_agents.RecordUpdates(agent)

        for _ in range(K):
            wrapped.on_update()

        agent.on_update.__wrapped__.assert_has_calls([call()] * K)
        self.assertListEqual(
            [f"p{i}" for i in range(P)],
            list(wrapped.updates_history.keys()),
        )
        np.testing.assert_equal(
            np.stack(list(wrapped.updates_history.values())).squeeze(),
            pars,
        )


class TestMonitorEpisode(unittest.TestCase):
    def test__compact_dicts_into_one(self):
        act = compact_dicts_into_one(
            [
                {"a": 3, "b": 2},
                {"b": 3, "c": 2},
                {"a": 3, "c": 2},
                {"c": 3, "d": 2},
                {"a": 3, "d": 2},
            ],
            fill_value=np.nan,
        )
        exp = {
            "a": [3, np.nan, 3, np.nan, 3],
            "b": [2, 3, np.nan, np.nan, np.nan],
            "c": [np.nan, 2, 2, 3, np.nan],
            "d": [np.nan, np.nan, np.nan, 2, 2],
        }
        self.assertDictEqual(act, exp)


if __name__ == "__main__":
    unittest.main()
