import logging
import os
import unittest
from functools import lru_cache
from itertools import combinations, product
from random import random
from typing import Any
from unittest.mock import Mock, call

import casadi as cs
import gymnasium as gym
import numpy as np
from csnlp import Nlp, scaling
from csnlp.multistart import StackedMultistartNlp
from csnlp.wrappers import Mpc, NlpScaling
from parameterized import parameterized

from mpcrl import (
    Agent,
    LearnableParameter,
    LearnableParametersDict,
    LearningAgent,
    wrappers_agents,
    wrappers_envs,
)
from mpcrl.wrappers.envs.monitor_infos import _compact_dicts


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
    mpc.set_nonlinear_dynamics(get_dynamics(g, alpha, dt))
    mpc.constraint("yT", y[-1], "==", yT)
    mpc.minimize(m[0] - m[-1] + cs.sum2(u2))
    mpc.init_solver()
    return mpc


class DummyAgent(Agent): ...


class DummyLearningAgent(LearningAgent):
    def update(self, *args, **kwargs):
        return

    def train_one_episode(self, *args, **kwargs):
        return


def mk_agent() -> DummyLearningAgent:
    return DummyLearningAgent(
        mpc=get_mpc(3, False),
        learnable_parameters=LearnableParametersDict(),
        update_strategy=1,
    )


class SimpleEnv(gym.Env[object, object]):
    T_MAX = 10

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.INTERNAL_RESET_INFOS = []
        self.INTERNAL_FINALIZED_RESET_INFOS = {"reset_info": []}
        self.INTERNAL_STEP_INFOS = []
        self.INTERNAL_FINALIZED_STEP_INFOS = {"step_info": []}
        self.INTERNAL_OBSERVATIONS = []
        self.INTERNAL_ACTIONS = []
        self.INTERNAL_REWARDS = []

    def reset(self, *args: Any, **kwargs: Any) -> tuple[object, dict[str, Any]]:
        super().reset(*args, **kwargs)
        self.t = 0
        obs = object()
        info = {"reset_info": object()}
        self.INTERNAL_RESET_INFOS.append(info)
        self.INTERNAL_FINALIZED_RESET_INFOS["reset_info"].append(info["reset_info"])
        self.INTERNAL_STEP_INFOS.append([])
        self.INTERNAL_FINALIZED_STEP_INFOS["step_info"].append([])
        self.INTERNAL_OBSERVATIONS.append([obs])
        self.INTERNAL_ACTIONS.append([])
        self.INTERNAL_REWARDS.append([])
        return obs, info

    def step(self, action: object) -> tuple[object, float, bool, bool, dict[str, Any]]:
        self.t += 1
        obs = object()
        reward = np.random.rand()
        terminated = self.t >= self.T_MAX
        info = {"step_info": object()}
        self.INTERNAL_STEP_INFOS[-1].append(info)
        self.INTERNAL_FINALIZED_STEP_INFOS["step_info"][-1].append(info["step_info"])
        self.INTERNAL_OBSERVATIONS[-1].append(obs)
        self.INTERNAL_ACTIONS[-1].append(action)
        self.INTERNAL_REWARDS[-1].append(reward)
        return obs, reward, terminated, False, info


class TestWrapperAndLearningWrapper(unittest.TestCase):
    def test_attr__raises__when_accessing_private_attrs(self):
        wrapped = wrappers_agents.LearningWrapper(mk_agent())
        with self.assertRaisesRegex(
            AttributeError, "Accessing private attribute '_x' is prohibited."
        ):
            wrapped._x

    def test_unwrapped__unwraps_nlp_correctly(self):
        agent = mk_agent()
        wrapped = wrappers_agents.LearningWrapper(agent)
        self.assertIs(agent, wrapped.unwrapped)

    def test_str_and_repr(self):
        agent = mk_agent()
        wrapped = wrappers_agents.LearningWrapper(agent)
        S = wrapped.__str__()
        self.assertIn(wrappers_agents.LearningWrapper.__name__, S)
        self.assertIn(agent.__str__(), S)
        S = wrapped.__repr__()
        self.assertIn(wrappers_agents.LearningWrapper.__name__, S)
        self.assertIn(agent.__repr__(), S)

    def test_is_wrapped(self):
        agent = mk_agent()
        self.assertFalse(agent.is_wrapped())

        wrapped = wrappers_agents.LearningWrapper(agent)
        self.assertTrue(wrapped.is_wrapped(wrappers_agents.LearningWrapper))
        self.assertFalse(wrapped.is_wrapped(cs.SX))

    def test_detach_wrapper(self):
        cp = lambda d: {k: {k_: id(v_) for k_, v_ in v.items()} for k, v in d.items()}
        agent_0 = mk_agent()
        agent_0_hooks = cp(agent_0._hooks)
        wrapped_1 = wrappers_agents.RecordUpdates(agent_0)
        wrapped_1_hooks = cp(agent_0._hooks)
        wrapped_2 = wrappers_agents.Log(
            wrapped_1, level=logging.DEBUG, log_frequencies={"on_timestep_end": 1000}
        )
        wrapped_2_hooks = cp(agent_0._hooks)
        for d1, d2 in combinations(
            (agent_0_hooks, wrapped_1_hooks, wrapped_2_hooks), 2
        ):
            with self.assertRaises(AssertionError):
                self.assertDictEqual(d1, d2)

        detached_1 = wrapped_2.detach_wrapper()
        self.assertIs(detached_1, wrapped_1)
        self.assertDictEqual(cp(agent_0._hooks), wrapped_1_hooks)

        detached_0 = detached_1.detach_wrapper()
        self.assertIs(detached_0, agent_0)
        self.assertDictEqual(cp(agent_0._hooks), agent_0_hooks)

        detached_recursive = wrapped_2.detach_wrapper(recursive=True)
        self.assertIs(detached_recursive, agent_0)
        self.assertDictEqual(cp(agent_0._hooks), agent_0_hooks)


class TestLog(unittest.TestCase):
    def setUp(self):
        self.agent = mk_agent()
        self.env = SimpleEnv()
        self.name = "test_logger"
        self.log = None

    def tearDown(self) -> None:
        # remove logging files with name "DummyLearningAgent*.txt"
        for handler in self.log.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                if os.path.isfile(handler.baseFilename):
                    os.remove(handler.baseFilename)

    def test_init__to_file__creates_log_file(self):
        self.log = wrappers_agents.Log(self.agent, to_file=True)
        found = False
        for handler in self.log.logger.handlers:
            if isinstance(
                handler, logging.FileHandler
            ) and handler.baseFilename.endswith(f"{self.agent.name}.txt"):
                found = True
                break
        self.assertTrue(found)

    def test_init__establishes_correct_hooks(self):
        agent = DummyAgent(get_mpc(3, False))
        log_frequencies = {
            "on_episode_start": 2,
            "on_episode_end": 3,
            "on_env_step": 4,
            "on_timestep_end": 5,
            "on_update": 6,
        }
        exclude_mandatory = ["on_mpc_failure", "on_validation_start"]
        self.log = wrappers_agents.Log(
            agent, log_frequencies=log_frequencies, exclude_mandatory=exclude_mandatory
        )

        expected_hooks = {
            "on_episode_start",
            "on_episode_end",
            "on_env_step",
            "on_timestep_end",
            "on_validation_end",
        }
        actual_hooks = set(self.log.unwrapped._hooks.keys())
        self.assertEqual(len(set.difference(expected_hooks, actual_hooks)), 0)

    def test_on_mpc_failure(self):
        episode = 1
        timestep = 10
        status = "failed"
        raises = False
        self.log = wrappers_agents.Log(self.agent)
        with self.assertLogs(self.log.logger, logging.WARNING):
            self.log.on_mpc_failure(episode, timestep, status, raises)

    def test_on_validation_start(self):
        self.log = wrappers_agents.Log(self.agent)
        with self.assertLogs(self.log.logger, logging.DEBUG):
            self.log.on_validation_start(self.env)

    def test_on_validation_end(self):
        returns = np.array([1.0, 2.0, 3.0])
        self.log = wrappers_agents.Log(self.agent)
        with self.assertLogs(self.log.logger, logging.INFO):
            self.log.on_validation_end(self.env, returns)

    def test_on_episode_start(self):
        episode = 1
        state = np.array([1, 2, 3])
        self.log = wrappers_agents.Log(
            self.agent, log_frequencies={"on_episode_start": 1}
        )
        with self.assertLogs(self.log.logger, logging.DEBUG):
            self.log.on_episode_start(self.env, episode, state)

    def test_on_episode_end(self):
        episode = 1
        rewards = 10.0
        self.log = wrappers_agents.Log(
            self.agent, log_frequencies={"on_episode_end": 1}
        )
        with self.assertLogs(self.log.logger, logging.INFO):
            self.log.on_episode_end(self.env, episode, rewards)

    def test_on_env_step(self):
        episode = 1
        timestep = 10
        self.log = wrappers_agents.Log(self.agent, log_frequencies={"on_env_step": 1})
        with self.assertLogs(self.log.logger, logging.DEBUG):
            self.log.on_env_step(self.env, episode, timestep)

    def test_on_timestep_end(self):
        episode = 1
        timestep = 10
        self.log = wrappers_agents.Log(
            self.agent, log_frequencies={"on_timestep_end": 1}
        )
        with self.assertLogs(self.log.logger, logging.DEBUG):
            self.log.on_timestep_end(self.env, episode, timestep)

    def test_on_update_failure(self):
        episode = 1
        timestep = 10
        errormsg = "update failed"
        raises = False
        self.log = wrappers_agents.Log(self.agent)
        with self.assertLogs(self.log.logger, logging.WARNING):
            self.log.on_update_failure(episode, timestep, errormsg, raises)

    def test_on_training_start(self):
        self.log = wrappers_agents.Log(self.agent)
        with self.assertLogs(self.log.logger, logging.DEBUG):
            self.log.on_training_start(self.env)

    def test_on_training_end(self):
        returns = np.array([1.0, 2.0, 3.0])
        self.log = wrappers_agents.Log(self.agent)
        with self.assertLogs(self.log.logger, logging.INFO):
            self.log.on_training_end(self.env, returns)

    def test_on_update(self):
        self.log = wrappers_agents.Log(self.agent, log_frequencies={"on_update": 1})
        with self.assertLogs(self.log.logger, logging.INFO):
            self.log.on_update()


class TestRecordUpdates(unittest.TestCase):
    @parameterized.expand([(1,), (3,), (20,)])
    def test__update__records_learnable_parameters_correctly(self, frequency: int):
        n_params, n_updates = 3, 10
        pars = np.random.randn(n_updates + 1, n_params)
        parsnames = [f"p{i}" for i in range(n_params)]
        parsdict = LearnableParametersDict(
            (LearnableParameter(n, 1, pars[0, i]) for i, n in enumerate(parsnames))
        )
        agent = mk_agent()
        agent._learnable_pars = parsdict

        on_update_original = agent.on_update
        pars_iter = iter(pars[1:])

        def side_effect():
            agent.learnable_parameters.update_values(next(pars_iter))
            on_update_original()

        agent.on_update = Mock(return_value=None, side_effect=side_effect)
        wrapped = wrappers_agents.RecordUpdates(agent, frequency)

        for _ in range(n_updates):
            wrapped.on_update()

        agent.on_update.assert_has_calls([call()] * n_updates)
        self.assertListEqual(parsnames, list(wrapped.updates_history.keys()))
        pars_recorded = np.squeeze(list(wrapped.updates_history.values()), 2).T
        np.testing.assert_equal(pars_recorded, pars[::frequency])


class TestEvaluate(unittest.TestCase):
    @parameterized.expand([(False,), (True,)])
    def test_evaluate__does_not_evaluate_if_not_training(self, is_training):
        frequency = 10
        repeats = 2
        agent = mk_agent()
        agent._is_training = is_training
        agent.evaluate = Mock()
        env = SimpleEnv()
        _ = wrappers_agents.Evaluate(agent, env, "on_episode_end", frequency=frequency)

        n_calls = frequency * repeats
        _ = [agent.on_episode_end(env, i, random()) for i in range(n_calls)]

        if is_training:
            self.assertGreater(agent.evaluate.call_count, 0)
        else:
            self.assertEqual(agent.evaluate.call_count, 0)

    @parameterized.expand(product((False, True), (False, True)))
    def test_evaluate__evaluates_with_correct_frequency(
        self, eval_immediately: bool, fix_seed: bool
    ):
        frequency = 10
        repeats = 2
        returns = [object() for _ in range(repeats + eval_immediately)]
        returns_iter = iter(returns)
        agent = mk_agent()
        agent.evaluate = Mock(side_effect=lambda *_, **__: next(returns_iter))
        env = SimpleEnv()
        wrapped = wrappers_agents.Evaluate(
            agent,
            env,
            "on_episode_end",
            frequency=frequency,
            eval_immediately=eval_immediately,
            fix_seed=fix_seed,
        )
        agent._is_training = True  # otherwise, Evaluate will not be invoked

        n_calls = frequency * repeats
        n_calls += int(frequency / 2)  # adds some spurious calls
        _ = [agent.on_episode_end(env, i, random()) for i in range(n_calls)]

        self.assertEqual(agent.evaluate.call_count, repeats + eval_immediately)
        self.assertListEqual(wrapped.eval_returns, returns)
        if fix_seed:
            seeds = (call.args[3] for call in agent.evaluate.call_args_list)
            self.assertEqual(len(set(seeds)), 1)


class TestMonitorEpisodesAndInfos(unittest.TestCase):
    def test__compact_dicts(self):
        act = _compact_dicts(
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

    def test_monitor_infos__records_infos_correctly(self):
        env = wrappers_envs.MonitorInfos(SimpleEnv())
        n_episodes = 3
        for _ in range(n_episodes):
            env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(object())
        self.assertListEqual(
            list(env.reset_infos), env.get_wrapper_attr("INTERNAL_RESET_INFOS")
        )
        self.assertDictEqual(
            env.finalized_reset_infos(),
            env.get_wrapper_attr("INTERNAL_FINALIZED_RESET_INFOS"),
        )
        self.assertListEqual(
            list(env.step_infos), env.get_wrapper_attr("INTERNAL_STEP_INFOS")
        )
        self.assertDictEqual(
            env.finalized_step_infos(),
            env.get_wrapper_attr("INTERNAL_FINALIZED_STEP_INFOS"),
        )

    def test_monitor_episodes__records_episodes_correctly(self):
        env = wrappers_envs.MonitorEpisodes(SimpleEnv())
        n_episodes = 3
        for _ in range(n_episodes):
            env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(object())
        np.testing.assert_array_equal(
            env.observations, env.get_wrapper_attr("INTERNAL_OBSERVATIONS")
        )
        np.testing.assert_array_equal(
            env.actions, env.get_wrapper_attr("INTERNAL_ACTIONS")
        )
        np.testing.assert_array_equal(
            env.rewards, env.get_wrapper_attr("INTERNAL_REWARDS")
        )
        self.assertListEqual(
            list(env.episode_lengths), [env.get_wrapper_attr("T_MAX")] * n_episodes
        )


if __name__ == "__main__":
    unittest.main()
