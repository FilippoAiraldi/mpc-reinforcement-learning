"""Reproduces the first numerical example in [1], but in an off-policy setting.

References
----------
[1] Gros, S. and Zanon, M., 2019. Data-driven economic NMPC using reinforcement
    learning. IEEE Transactions on Automatic Control, 65(2), pp. 636-648.
"""

import logging
from collections.abc import Callable, Iterable
from typing import Any, Optional

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp.wrappers import Mpc
from gymnasium.wrappers import TimeLimit
from more_itertools import pairwise

from mpcrl import Agent, LearnableParameter, LearnableParametersDict, LstdQLearningAgent
from mpcrl.optim import NetwonMethod
from mpcrl.util.control import dlqr
from mpcrl.wrappers.agents import Evaluate, Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

# first, create classes for environment and mpc controller


class LtiSystem(gym.Env[npt.NDArray[np.floating], float]):
    """A simple discrete-time LTI system affected by noise."""

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
        """Resets the state of the LTI system."""
        super().reset(seed=seed, options=options)
        self.x = np.asarray([0, 0.15]).reshape(self.nx, 1)
        return self.x, {}

    def get_stage_cost(self, state: npt.NDArray[np.floating], action: float) -> float:
        """Computes the stage cost `L(s,a)`."""
        lb, ub = self.x_bnd
        return (
            0.5
            * (
                np.square(state).sum()
                + 0.5 * action**2
                + self.w.T @ np.maximum(0, lb - state)
                + self.w.T @ np.maximum(0, state - ub)
            ).item()
        )

    def step(
        self, action: cs.DM
    ) -> tuple[npt.NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Steps the LTI system."""
        action = float(action)
        x_new = self.A @ self.x + self.B * action
        x_new[0] += self.np_random.uniform(*self.e_bnd)
        r = self.get_stage_cost(self.x, action)
        self.x = x_new
        return x_new, r, False, False, {}


class LinearMpc(Mpc[cs.SX]):
    """A simple linear MPC controller."""

    horizon = 10
    discount_factor = 0.9
    learnable_pars_init = {
        "V0": np.asarray(0.0),
        "x_lb": np.asarray([0, 0]),
        "x_ub": np.asarray([1, 0]),
        "b": np.zeros(LtiSystem.nx),
        "f": np.zeros(LtiSystem.nx + LtiSystem.nu),
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

        # parameters
        V0 = self.parameter("V0")
        x_lb = self.parameter("x_lb", (nx,))
        x_ub = self.parameter("x_ub", (nx,))
        b = self.parameter("b", (nx, 1))
        f = self.parameter("f", (nx + nu, 1))
        A = self.parameter("A", (nx, nx))
        B = self.parameter("B", (nx, nu))

        # variables (state, action, slack)
        x, _ = self.state("x", nx, bound_initial=False)
        u, _ = self.action("u", nu, lb=a_bnd[0], ub=a_bnd[1])
        s, _, _ = self.variable("s", (nx, N), lb=0)

        # dynamics
        self.set_dynamics(lambda x, u: A @ x + B * u + b, n_in=2, n_out=1)

        # other constraints
        self.constraint("x_lb", x_bnd[0] + x_lb - s, "<=", x[:, 1:])
        self.constraint("x_ub", x[:, 1:], "<=", x_bnd[1] + x_ub + s)

        # objective
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

        # solver
        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            # "jit": True,
            # "jit_cleanup": True,
            "ipopt": {
                # "linear_solver": "ma97",
                # "linear_system_scaling": "mc19",
                # "nlp_scaling_method": "equilibration-based",
                "max_iter": 500,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")


def get_rollout_generator(
    rollout_seed: int,
) -> Callable[[int], Iterable[tuple[np.ndarray, float, float, np.ndarray]]]:
    """Returns a function used to generates rollouts from a nominal agent."""
    nominal_agent = Agent(LinearMpc(), LinearMpc.learnable_pars_init.copy())

    def _generate_rollout(n):
        # run the nominal agent on the environment once
        env = MonitorEpisodes(TimeLimit(LtiSystem(), 100))
        nominal_agent.evaluate(env, episodes=1, seed=rollout_seed + n)

        # transform the collected env data into a SARS sequence
        S, A, R = (
            env.observations[0].squeeze(),
            env.actions[0].squeeze(),
            env.rewards[0],
        )
        return ((s, a, r, s_next) for (s, s_next), a, r in zip(pairwise(S), A, R))

    return _generate_rollout


if __name__ == "__main__":
    # now, let's create the instances of such classes
    seed = 69
    mpc = LinearMpc()
    learnable_pars = LearnableParametersDict[cs.SX](
        (
            LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
            for name, val in mpc.learnable_pars_init.items()
        )
    )
    agent = Evaluate(
        Log(
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
        ),
        eval_env=TimeLimit(LtiSystem(), 100),
        hook="on_episode_end",
        frequency=10,
        n_eval_episodes=5,
        eval_immediately=True,
        seed=seed,
    )

    # before training, let's create a nominal non-learning agent which will be used to
    # generate expert rollout data. This data will then be used to train the off-policy
    # q-learning agent.
    generate_rollout = get_rollout_generator(rollout_seed=69)

    # finally, we can launch the training
    n_rollouts = 100
    agent.train_offpolicy(
        episode_rollouts=(generate_rollout(n) for n in range(n_rollouts)), seed=seed
    )
    eval_returns = np.asarray(agent.eval_returns)

    # plot the results
    import matplotlib.pyplot as plt

    _, axs = plt.subplots(2, 1, constrained_layout=True)
    eval_returns_avg = eval_returns.mean(1)
    eval_returns_std = eval_returns.std(1)
    evals = np.arange(1, eval_returns.shape[0] + 1)
    axs[0].plot(agent.td_errors, "o", markersize=1)
    axs[0].set_ylabel("Time steps")
    axs[0].set_ylabel(r"$\tau$")
    patch = axs[1].fill_between(
        evals,
        eval_returns_avg - eval_returns_std,
        eval_returns_avg + eval_returns_std,
        alpha=0.3,
    )
    axs[1].plot(evals, eval_returns_avg, color=patch.get_facecolor())
    axs[1].set_ylabel("Evaluations")
    axs[1].set_ylabel(r"$\sum L$")

    _, axs = plt.subplots(3, 2, constrained_layout=True, sharex=True)
    updates_history = {k: np.asarray(v) for k, v in agent.updates_history.items()}
    axs[0, 0].plot(updates_history["b"])
    axs[0, 1].plot(np.stack([updates_history[n][:, 0] for n in ("x_lb", "x_ub")], -1))
    axs[1, 0].plot(updates_history["f"])
    axs[1, 1].plot(updates_history["V0"])
    axs[2, 0].plot(updates_history["A"].reshape(-1, 4))
    axs[2, 1].plot(updates_history["B"].squeeze())
    axs[0, 0].set_ylabel("$b$")
    axs[0, 1].set_ylabel("$x_1$")
    axs[1, 0].set_ylabel("$f$")
    axs[1, 1].set_ylabel("$V_0$")
    axs[2, 0].set_ylabel("$A$")
    axs[2, 1].set_ylabel("$B$")

    plt.show()
