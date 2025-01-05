r"""
.. _examples_qlearning:

On-policy Q-learning
====================

This example tries to reproduce the results from the linear MPC numerical experiment in
:cite:`gros_datadriven_2020`. We are given an RL environment whose cost function is

.. math::
    L(s,a) = \frac{1}{2} \left(
        s^\top s + \frac{1}{2} a^2 + w^\top \max\{0, \underline{s} - s\}
        + w^\top \max\{0, s - \overline{s}\}
    \right)

where :math:`s` is the state, :math:`a` is the action, :math:`w` is a weight vector, and
:math:`\underline{s}` and :math:`\overline{s}` are the lower and upper bounds of the
state, respectively. The dynamics of the real environment are

.. math::
    s_+ = \begin{bmatrix} 0.9 & 0.35 \\ 0 & 1.1 \end{bmatrix} s
        + \begin{bmatrix} 0.0813 \\ 0.2 \end{bmatrix} a
        + \begin{bmatrix} e \\ 0 \end{bmatrix}

where :math:`e \sim \mathcal{U}(-0.1, 0)`. Given the state :math:`s_k`, the following
MPC scheme is used to control the system

.. math::
   \begin{aligned}
      \min_{x_{0:N}, u_{0:N-1}, \sigma_{1:N}} \quad &
        V_0 + x_N^\top S x_N + \sum_{i=1}^{N}{ w^\top \sigma_i } \\
        & + \sum_{i=0}^{N-1}{ \gamma^i
            \left(
                x_i^\top x_i + 0.5 u_i^2 +
                f^\top \begin{bmatrix} x_i \\ u_i \end{bmatrix}
            \right)
        } \\
      \textrm{s.t.} \quad & x_0 = s_k \\
                          & x_{i+1} = A x_i + B u_i + b & i=0,\dots,N-1 \\
                          & \underline{s} + \underline{x} - \sigma_i \leq x_i
                            \leq \overline{s} + \overline{x} + \sigma_i
                            \quad & i=1,\dots,N
   \end{aligned}

with :math:`\gamma = 0.9`, and the learnable parameters are

.. math:: \theta = \left(
        V_0, \underline{x}, \overline{x}, b, f, A, B
    \right)

The parameters are initialized differently, and in particular, the prediction model of
the MPC is initialized wrongly as

.. math::
    A = \begin{bmatrix} 1 & 0.25 \\ 0 & 1 \end{bmatrix}, \quad
    B = \begin{bmatrix} 0.0312 \\ 0.25 \end{bmatrix},

and :math:`S` is the solution to the corresponding discrete-time algebraic Riccati
equation, i.e., computed with the wrong dynamics matrices. The task is simple: find a
parametrization :math:`\theta` such that the cost function is minimized.  To solve it,
we will employ a second-order LSTD Q-learning algorithm.
"""

import logging
from typing import Any, Optional

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp.wrappers import Mpc
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit

from mpcrl import LearnableParameter, LearnableParametersDict, LstdQLearningAgent
from mpcrl.optim import NetwonMethod
from mpcrl.util.control import dlqr
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

# %%
# Defining the environment
# ------------------------
# First things first, we need to build the environment. We will use the :mod:`gymnasium`
# library to do so. The most important methods are :func:`gymnasium.Env.reset` and
# :func:`gymnasium.Env.step`, which will be called to reset the environment to its
# initial state and to step the dynamics and receive a realization of the reward signal,
# respectively. The environment is defined as a the following class.


class LtiSystem(gym.Env[npt.NDArray[np.floating], float]):
    """A simple discrete-time LTI system affected by uniform noise."""

    nx = 2  # number of states
    nu = 1  # number of inputs
    A = np.asarray([[0.9, 0.35], [0, 1.1]])  # state-space matrix A
    B = np.asarray([[0.0813], [0.2]])  # state-space matrix B
    x_bnd = (np.asarray([[0], [-1]]), np.asarray([[1], [1]]))  # bounds of state
    a_bnd = (-1, 1)  # bounds of control input
    w = np.asarray([[1e2], [1e2]])  # penalty weight for bound violations
    e_bnd = (-1e-1, 0)  # uniform noise bounds
    action_space = Box(*a_bnd, (nu,), np.float64)

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
        """Computes the stage cost :math:`L(s,a)`."""
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


# %%
# Defining the MPC controller
# ---------------------------
# The second component is the MPC controller. We'll create a custom that, of course,
# inherits from :class:`csnlp.wrappers.Mpc`. The implementation is as follows, and it is
# in line with the theory presented above.


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
        self.set_affine_dynamics(A, B, c=b)

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
                gammapowers * (cs.sum1(x[:, :-1] ** 2) + 0.5 * cs.sum1(u**2) + w.T @ s)
            )
        )

        # solver
        opts = {
            "expand": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            "fatrop": {"max_iter": 500, "print_level": 0},
        }
        self.init_solver(opts, solver="fatrop", type="nlp")


# %%
# Simulation
# ----------
# So far, we have only defined the classes for the environment and the MPC controller.
# Now, it is time to instantiate these and run the simulation. This is comprised of
# multiple steps, which are detailed below.
#
# 1. We instantiate the environment. Note how it is wrapped in two different wrappers:
#    :class:`gymnasium.wrappers.TimeLimit` is used to impose a maximum amount of steps
#    to be simulated, whereas :class:`mpcrl.wrappers.envs.MonitorEpisodes` is used to
#    record the state, action and reward signals at each time step for plotting
#    purposes.
# 2. We instantiate the MPC controller and define its learnable parameters.
# 3. We instantiate the Q-learning agent. We pass different options to it, such as
#    the update strategy, the optimizer, the Hessian type, etc. For plotting purposes,
#    it is also wrapped such that the updated parameters are recorded. And we also log
#    the progress of the simulation.
# 4. We run the simulation. Under the hood, the agent will interact with the
#    environment, collect data, and update the parameters of the MPC controller.
# 5. Finally, we plot the results. The first plot shows the evolution of the states and
#    the control action, and the corresponding bounds. The second plot shows the
#    TD error and the time-wise stage cost realizations. The last plot shows how each
#    learnable parameter evolves over time.

if __name__ == "__main__":
    # instantiate the env and wrap it
    env = MonitorEpisodes(TimeLimit(LtiSystem(), max_episode_steps=5_000))

    # now build the MPC and the dict of learnable parameters
    mpc = LinearMpc()
    learnable_pars = LearnableParametersDict[cs.SX](
        (
            LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
            for name, val in mpc.learnable_pars_init.items()
        )
    )

    # build and wrap appropriately the agent
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
        log_frequencies={"on_timestep_end": 1000},
    )

    # launch the training simulation
    agent.train(env=env, episodes=1, seed=69)

    # plot the results
    import matplotlib.pyplot as plt

    X = env.get_wrapper_attr("observations")[0].squeeze().T
    U = env.get_wrapper_attr("actions")[0].squeeze()
    R = env.get_wrapper_attr("rewards")[0]
    _, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
    axs[0].plot(X[0])
    axs[1].plot(X[1])
    axs[2].plot(U)
    for i in range(2):
        axs[0].axhline(env.get_wrapper_attr("x_bnd")[i][0], color="r")
        axs[1].axhline(env.get_wrapper_attr("x_bnd")[i][1], color="r")
        axs[2].axhline(env.get_wrapper_attr("a_bnd")[i], color="r")
    axs[0].set_ylabel("$s_1$")
    axs[1].set_ylabel("$s_2$")
    axs[2].set_ylabel("$a$")

    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(agent.td_errors, "o", markersize=1)
    axs[1].semilogy(R, "o", markersize=1)
    axs[0].set_ylabel(r"$\tau$")
    axs[1].set_ylabel("$L$")

    _, axs = plt.subplots(3, 2, constrained_layout=True, sharex=True)
    axs[0, 0].plot(np.asarray(agent.updates_history["b"]))
    axs[0, 1].plot(
        np.stack(
            [np.asarray(agent.updates_history[n])[:, 0] for n in ("x_lb", "x_ub")], -1
        ),
    )
    axs[1, 0].plot(np.asarray(agent.updates_history["f"]))
    axs[1, 1].plot(np.asarray(agent.updates_history["V0"]))
    axs[2, 0].plot(np.asarray(agent.updates_history["A"]).reshape(-1, 4))
    axs[2, 1].plot(np.asarray(agent.updates_history["B"]).squeeze())
    axs[0, 0].set_ylabel("$b$")
    axs[0, 1].set_ylabel("$x_1$")
    axs[1, 0].set_ylabel("$f$")
    axs[1, 1].set_ylabel("$V_0$")
    axs[2, 0].set_ylabel("$A$")
    axs[2, 1].set_ylabel("$B$")
    plt.show()
