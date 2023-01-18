import logging
import pickle
import unittest
from typing import Any, Dict, Optional, Tuple

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp.util.math import quad_form
from csnlp.wrappers import Mpc
from gymnasium.wrappers import TimeLimit
from parameterized import parameterized
from scipy.io import loadmat

from mpcrl import LearnableParameter, LearnableParametersDict, LstdQLearningAgent
from mpcrl.util.math import dlqr
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes


class TestExamples(unittest.TestCase):
    @parameterized.expand([(True,), (False,)])
    def test_q_learning__with_copy_and_pickle(self, use_copy: bool):
        class LtiSystem(gym.Env[npt.NDArray[np.floating], float]):
            """A simple discrete-time LTI system affected by noise.ad"""

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
                options: Optional[Dict[str, Any]] = None,
            ) -> Tuple[npt.NDArray[np.floating], Dict[str, Any]]:
                """Resets the state of the LTI system."""
                super().reset(seed=seed, options=options)
                self.x = np.asarray([0, 0.15]).reshape(self.nx, 1)
                return self.x, {}

            def get_stage_cost(
                self, state: npt.NDArray[np.floating], action: float
            ) -> float:
                """Computes the stage cost `L(s,a)`."""
                lb, ub = self.x_bnd
                return 0.5 * float(
                    np.square(state).sum()
                    + 0.5 * action**2
                    + self.w.T @ np.maximum(0, lb - state)
                    + self.w.T @ np.maximum(0, state - ub)
                )

            def step(
                self, action: cs.DM
            ) -> Tuple[npt.NDArray[np.floating], float, bool, bool, Dict[str, Any]]:
                """Steps the LTI system."""
                action = float(action)
                x_new = self.A @ self.x + self.B * action
                x_new[0] += self.np_random.uniform(*self.e_bnd)
                self.x = x_new
                r = self.get_stage_cost(self.x, action)
                return x_new, r, False, False, {}

        class LinearMpc(Mpc[cs.SX]):
            """A simple linear MPC controller."""

            horizon = 10
            discount_factor = 0.9
            learnable_pars_init = {
                "V0": np.asarray([[0.0]]),
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

                # parameters (need to be flat)
                V0 = self.parameter("V0")
                x_lb = self.parameter("x_lb", (nx,))
                x_ub = self.parameter("x_ub", (nx,))
                b = self.parameter("b", (nx, 1))
                f = self.parameter("f", (nx + nu, 1))
                A = self.parameter("A", (nx * nx, 1)).reshape((nx, nx))
                B = self.parameter("B", (nx * nu, 1)).reshape((nx, nu))

                # variables (state, action, slack)
                x, _ = self.state("x", nx)
                u, _ = self.action("u", nu, lb=a_bnd[0], ub=a_bnd[1])
                s, _, _ = self.variable("s", (nx, N), lb=0)

                # dynamics
                self.set_dynamics(lambda x, u: A @ x + B * u + b, n_in=2, n_out=1)

                # other constraints
                self.constraint("x_lb", x_bnd[0] + x_lb - s, "<=", x[:, 1:])
                self.constraint("x_ub", x[:, 1:], "<=", x_bnd[1] + x_ub + s)

                # objective
                A_init, B_init = (
                    self.learnable_pars_init["A"],
                    self.learnable_pars_init["B"],
                )
                S = cs.DM(dlqr(A_init, B_init, 0.5 * np.eye(nx), 0.25 * np.eye(nu))[1])
                gammapowers = cs.DM(gamma ** np.arange(N)).T
                self.minimize(
                    V0
                    + quad_form(S, x[:, -1])
                    + cs.sum2(f.T @ cs.vertcat(x[:, :-1], u))
                    + 0.5
                    * cs.sum2(
                        gammapowers
                        * (cs.sum1(x[:, :-1] ** 2) + 0.5 * cs.sum1(u**2) + w.T @ s)
                    )
                )

                # solver
                opts = {
                    "expand": True,
                    "print_time": False,
                    "ipopt": {
                        "max_iter": 500,
                        "sb": "yes",
                        "print_level": 0,
                    },
                }
                self.init_solver(opts, solver="ipopt")

        mpc = LinearMpc()
        learnable_pars = LearnableParametersDict[cs.SX](
            (
                LearnableParameter(
                    name=name,
                    size=np.prod(val.shape),
                    value=val.flatten(order="F"),
                    sym=cs.vec(mpc.parameters[name]),
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
                    learning_rate=5e-2,
                    hessian_type="approx",
                    record_td_errors=True,
                    max_percentage_update=1e3,  # does nothing, just allows to test qp
                )
            ),
            level=logging.DEBUG,
            log_frequencies={"on_env_step": 100},
        )

        agent_copy = agent.copy()
        if use_copy:
            agent = agent_copy
        J = LstdQLearningAgent.train(agent, env=env, episodes=1, seed=69).item()
        agent = pickle.loads(pickle.dumps(agent))

        X = env.observations[0].squeeze().T
        U = env.actions[0].squeeze()
        R = env.rewards[0]
        TD = np.squeeze(agent.td_errors)
        parnames = ["V0", "x_lb", "x_ub", "b", "f", "A", "B"]
        pars = {n: np.squeeze(agent.updates_history[n]) for n in parnames}

        d = loadmat("tests/examples_data.mat", squeeze_me=True)
        np.testing.assert_allclose(J, d["q_learning_J"], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(X, d["q_learning_X"], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(U, d["q_learning_U"], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(R, d["q_learning_R"], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(TD, d["q_learning_TD"], rtol=1e-4, atol=1e-4)
        pars_expected = d["q_learning_pars"].item()
        for i, par in enumerate(pars.values()):
            np.testing.assert_allclose(par, pars_expected[i], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
