import logging
import os
import pickle
import sys
import unittest
from operator import neg
from sys import platform
from warnings import catch_warnings

sys.path.append(os.path.join(os.getcwd(), "examples"))

import casadi as cs
import numpy as np
import torch
from bayesopt import BoTorchOptimizer, CstrEnv, NoisyFilterObservation, get_cstr_mpc
from dpg import LinearMpc as DpgLinearMpc
from dpg import LtiSystem as DpgLtiSystem
from gymnasium.wrappers import TimeLimit, TransformReward
from parameterized import parameterized
from q_learning import LinearMpc as QLearningLinearMpc
from q_learning import LtiSystem as QLearningLtiSystem
from q_learning_offpolicy import LinearMpc as QLearningOffPolicyLinearMpc
from q_learning_offpolicy import LtiSystem as QLearningOffPolicyLtiSystem
from q_learning_offpolicy import get_rollout_generator
from scipy.io import loadmat

from mpcrl import (
    GlobOptLearningAgent,
    LearnableParameter,
    LearnableParametersDict,
    LstdDpgAgent,
    LstdQLearningAgent,
    UpdateStrategy,
)
from mpcrl import exploration as E
from mpcrl.optim import GradientDescent, NetwonMethod
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

torch.set_default_device("cpu")
torch.set_default_dtype(torch.float64)


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
        env = MonitorEpisodes(TimeLimit(QLearningLtiSystem(), max_episode_steps=100))
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
                    name=name, shape=val.shape, value=val, sym=mpc.parameters[name]
                )
                for name, val in mpc.learnable_pars_init.items()
            )
        )
        env = MonitorEpisodes(TimeLimit(DpgLtiSystem(), max_episode_steps=200))
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

    @parameterized.expand([(False,), (True,)])
    def test_q_learning_offpolicy__with_copy_and_pickle(self, use_copy: bool):
        mpc = QLearningOffPolicyLinearMpc()
        learnable_pars = LearnableParametersDict[cs.SX](
            (
                LearnableParameter(
                    name=name, shape=val.shape, value=val, sym=mpc.parameters[name]
                )
                for name, val in mpc.learnable_pars_init.items()
            )
        )
        agent = Log(
            RecordUpdates(
                LstdQLearningAgent(
                    mpc=mpc,
                    learnable_parameters=learnable_pars,
                    discount_factor=mpc.discount_factor,
                    update_strategy=1,
                    optimizer=NetwonMethod(learning_rate=5e-2),
                    hessian_type="approx",
                    record_td_errors=True,
                    remove_bounds_on_initial_action=True,
                )
            ),
            level=logging.DEBUG,
            log_frequencies={"on_episode_end": 1},
        )

        seed = 69
        env_factory = lambda: MonitorEpisodes(
            TimeLimit(QLearningOffPolicyLtiSystem(), 100)
        )
        generate_rollout = get_rollout_generator(env_factory, seed)

        agent_copy = agent.copy()
        if use_copy:
            agent = agent_copy
        J = agent.train_offpolicy(
            episode_rollouts=(generate_rollout(n) for n in range(6)),
            seed=seed,
            eval_frequency=2,
            eval_env_factory=env_factory,
            eval_kwargs={
                "episodes": 2,
            },
            eval_at_start=False,
        )
        agent = pickle.loads(pickle.dumps(agent))

        parnames = ["V0", "x_lb", "x_ub", "b", "f", "A", "B"]
        PARS = np.concatenate(
            [np.reshape(agent.updates_history[n], -1) for n in parnames]
        )
        TD = np.squeeze(agent.td_errors)

        # from scipy.io import savemat
        # DATA.update({"ql_offpol_J": J, "ql_offpol_TD": TD, "ql_offpol_pars": PARS})
        # savemat(f"tests/data_test_examples_{platform}.mat", DATA)

        np.testing.assert_allclose(J, DATA["ql_offpol_J"], rtol=1e0, atol=1e0)
        np.testing.assert_allclose(TD, DATA["ql_offpol_TD"], rtol=1e1, atol=1e1)
        np.testing.assert_allclose(PARS, DATA["ql_offpol_pars"], rtol=1e0, atol=1e0)


if __name__ == "__main__":
    unittest.main()
