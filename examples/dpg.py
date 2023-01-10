"""Reproduces the first numerical example in [1] but using LSTD DPG

References
----------
[1] Gros, S. and Zanon, M., 2019. Data-driven economic NMPC using reinforcement
    learning. IEEE Transactions on Automatic Control, 65(2), pp. 636-648.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import casadi as cs
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp.util.math import quad_form
from csnlp.wrappers import Mpc
from gymnasium.wrappers import TimeLimit

from mpcrl import (
    LearnableParameter,
    LearnableParametersDict,
    LstdDpgAgent,
    UpdateStrategy,
)
from mpcrl import exploration as E
from mpcrl.util.math import dlqr
from mpcrl.wrappers import Log, RecordUpdates

# first, create classes for environment and mpc controller


class LtiSystem(gym.Env[npt.NDArray[np.double], float]):
    """A simple discrete-time LTI system affected by noise.ad"""

    nx = 2  # number of states
    nu = 1  # number of inputs
    A = np.asarray([[0.9, 0.35], [0, 1.1]])  # state-space matrix A
    B = np.asarray([[0.0813], [0.2]])  # state-space matrix B
    x_bnd = (np.asarray([[0], [-1]]), np.asarray([[1], [1]]))  # bounds of state
    a_bnd = (-1, 1)  # bounds of control input
    w = np.asarray([[1e2], [1e2]])  # penalty weight for bound violations
    e_bnd = (-1e-1, 0)  # uniform noise bounds
    X: List[npt.NDArray[np.double]] = []
    U: List[float] = []
    R: List[float] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.double], Dict[str, Any]]:
        """Resets the state of the LTI system."""
        super().reset(seed=seed, options=options)
        self.x = np.asarray([0, 0.15]).reshape(self.nx, 1)
        self.X, self.U, self.R = [self.x], [], []
        return self.x, {}

    def get_stage_cost(self, state: npt.NDArray[np.double], action: float) -> float:
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
    ) -> Tuple[npt.NDArray[np.double], float, bool, bool, Dict[str, Any]]:
        """Steps the LTI system."""
        action = float(action)
        x_new = self.A @ self.x + self.B * action
        x_new[0] += self.np_random.uniform(*self.e_bnd)
        self.x = x_new
        r = self.get_stage_cost(self.x, action)
        self.U.append(action)
        self.X.append(self.x)
        self.R.append(r)
        return x_new, r, False, False, {}


class LinearMpc(Mpc[cs.SX]):
    """A simple linear MPC controller."""

    horizon = 10
    discount_factor = 0.9
    learnable_pars_init = {
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

        # parameters (need to be flat)
        V0 = self.parameter("V0", (nx,))
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
        A_init, B_init = self.learnable_pars_init["A"], self.learnable_pars_init["B"]
        S = cs.DM(
            dlqr(A_init, B_init, 0.5 * np.eye(nx), 0.25)[1]  # type: ignore[arg-type]
        )
        gammapowers = cs.DM(gamma ** np.arange(N)).T
        self.minimize(
            V0.T @ x[:, 0]  # to have a derivative, V0 must be a function of the state
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


# now, let's create the instances of such classes and start the training
mpc = LinearMpc()
learnable_pars = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(
            name=name,
            size=int(np.prod(val.shape)),
            value=val.flatten(order="F"),
            sym=cs.vec(mpc.parameters[name]),
        )
        for name, val in mpc.learnable_pars_init.items()
    )
)
env = TimeLimit(LtiSystem(), max_episode_steps=int(1e4))
rollout_length = 200
agent = Log(  # type: ignore[var-annotated]
    RecordUpdates(
        LstdDpgAgent(
            mpc=mpc,
            learnable_parameters=learnable_pars,
            discount_factor=mpc.discount_factor,
            learning_rate=1e-6,
            update_strategy=UpdateStrategy(rollout_length, "on_env_step"),
            rollout_length=rollout_length,
            exploration=E.GreedyExploration(0.5),  # type: ignore[arg-type]
            record_policy_performance=True,
            record_policy_gradient=True,
        )
    ),
    level=logging.DEBUG,
    log_frequencies={"on_env_step": 1000},
)
agent.train(env=env, episodes=1, seed=69)


# plot the results
X = np.concatenate(env.X, axis=-1)
U = np.asarray(env.U)
R = np.asarray(env.R)
_, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
axs[0].plot(X[0])
axs[1].plot(X[1])
axs[2].plot(U)
for i in range(2):
    axs[0].axhline(env.x_bnd[i][0], color="r")
    axs[1].axhline(env.x_bnd[i][1], color="r")
    axs[2].axhline(env.a_bnd[i], color="r")
axs[0].set_ylabel("$s_1$")
axs[1].set_ylabel("$s_2$")
axs[2].set_ylabel("$a$")

_, axs = plt.subplots(3, 1, constrained_layout=True)
axs[0].plot(agent.policy_performances)
axs[1].plot(np.linalg.norm(agent.policy_gradients, axis=1))
axs[2].semilogy(R, "o", markersize=1)
axs[0].set_ylabel(r"$J(\pi_\theta)$")
axs[1].set_ylabel(r"$\nabla_\theta J(\pi_\theta)$")
axs[2].set_ylabel("$L$")

_, axs = plt.subplots(3, 2, constrained_layout=True, sharex=True)
axs[0, 0].plot(np.asarray(agent.updates_history["b"]))
axs[0, 1].plot(
    np.stack(
        [np.asarray(agent.updates_history[n])[:, 0] for n in ("x_lb", "x_ub")], axis=-1
    ),
)
axs[1, 0].plot(np.asarray(agent.updates_history["f"]))
axs[1, 1].plot(np.asarray(agent.updates_history["V0"]))
axs[2, 0].plot(np.asarray(agent.updates_history["A"]))
axs[2, 1].plot(np.asarray(agent.updates_history["B"]))
axs[0, 0].set_ylabel("$b$")
axs[0, 1].set_ylabel("$x_1$")
axs[1, 0].set_ylabel("$f$")
axs[1, 1].set_ylabel("$V_0$")
axs[2, 0].set_ylabel("$A$")
axs[2, 1].set_ylabel("$B$")

plt.show()
