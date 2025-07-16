# Reinforcement Learning with Model Predictive Control

**M**odel **P**redictive **C**ontrol-based **R**einforcement **L**earning (**mpcrl**,
for short) is a library for training model-based Reinforcement Learning (RL) [[1]](#1)
agents with Model Predictive Control (MPC) [[2]](#2) as function approximation.

> |   |   |
> |---|---|
> | **Documentation** | <https://mpc-reinforcement-learning.readthedocs.io/en/stable/>         |
> | **Download**      | <https://pypi.python.org/pypi/mpcrl/>                                  |
> | **Source code**   | <https://github.com/FilippoAiraldi/mpc-reinforcement-learning/>        |
> | **Report issues** | <https://github.com/FilippoAiraldi/mpc-reinforcement-learning/issues/> |

[![PyPI version](https://badge.fury.io/py/mpcrl.svg)](https://badge.fury.io/py/mpcrl)
[![Source Code License](https://img.shields.io/badge/license-MIT-blueviolet)](https://github.com/FilippoAiraldi/mpc-reinforcement-learning/blob/main/LICENSE)
![Python 3.9](https://img.shields.io/badge/python->=3.9-green.svg)

[![Tests](https://github.com/FilippoAiraldi/mpc-reinforcement-learning/actions/workflows/tests.yml/badge.svg)](https://github.com/FilippoAiraldi/mpc-reinforcement-learning/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/mpc-reinforcement-learning/badge/?version=stable)](https://mpc-reinforcement-learning.readthedocs.io/en/stable/?badge=stable)
[![Downloads](https://static.pepy.tech/badge/mpcrl)](https://www.pepy.tech/projects/mpcrl)
[![Maintainability](https://qlty.sh/gh/FilippoAiraldi/projects/mpc-reinforcement-learning/maintainability.svg)](https://qlty.sh/gh/FilippoAiraldi/projects/mpc-reinforcement-learning)
[![Code Coverage](https://qlty.sh/gh/FilippoAiraldi/projects/mpc-reinforcement-learning/coverage.svg)](https://qlty.sh/gh/FilippoAiraldi/projects/mpc-reinforcement-learning)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Introduction

This framework, also referred to as _RL with/using MPC_, was first proposed in [[3]](#3)
and has so far been shown effective in various
applications, with different learning algorithms and more sound theory, e.g., [[4](#4),
[5](#5), [7](#7), [8](#8)]. It merges two powerful control techinques into a single
data-driven one

- MPC, a well-known control methodology that exploits a prediction model to predict the
  future behaviour of the environment and compute the optimal action

- and RL, a Machine Learning paradigm that showed many successes in recent years (with
  games such as chess, Go, etc.) and is highly adaptable to unknown and complex-to-model
  environments.

The figure below shows the main idea behind this learning-based control approach. The
MPC controller, parametrized in its objective, predictive model and constraints (or a
subset of these), acts both as policy provider (i.e., providing an action to the
environment, given the current state) and as function approximation for the state and
action value functions (i.e., predicting the expected return following the current
control policy from the given state and state-action pair). Concurrently, an RL
algorithm is employed to tune this parametrization of the MPC in such a way to increase
the controller's performance and achieve an (sub)optimal policy. For this purpose,
different algorithms can be employed, two of the most successful being Q-learning
[[4]](#4) and Deterministic Policy Gradient (DPG) [[5]](#5).

<div align="center">
  <img src="https://raw.githubusercontent.com/FilippoAiraldi/mpc-reinforcement-learning/main/docs/_static/mpcrl.diagram.light.png" alt="mpcrl-diagram" height="300">
</div>

---

## Installation

### Using `pip`

You can use `pip` to install **mpcrl** with the command

```bash
pip install mpcrl
```

**mpcrl** has the following dependencies

- Python 3.9 or higher (though support and testing for 3.9 are deprecated)
- [csnlp](https://casadi-nlp.readthedocs.io/en/stable/)
- [SciPy](https://scipy.org/)
- [Gymnasium](https://gymnasium.farama.org/)
- [Numba](https://numba.pydata.org/)
- [typing_extensions](https://pypi.org/project/typing-extensions/) (only for Python 3.9)

If you'd like to play around with the source code instead, run

```bash
git clone https://github.com/FilippoAiraldi/mpc-reinforcement-learning.git
```

The `main` branch contains the main releases of the packages (and the occasional post
release). The `experimental` branch is reserved for the implementation and test of new
features and hosts the release candidates. You can then install the package to edit it
as you wish as

```bash
pip install -e /path/to/mpc-reinforcement-learning
```

---

## Getting started

Here we provide the skeleton of a simple application of the library. The aim of the code
below is to let an MPC control strategy learn how to optimally control a simple Linear
Time Invariant (LTI) system. The cost (i.e., the opposite of the reward) of controlling
this system in state $s \in \mathbb{R}^{n_s}$ with action
$a \in \mathbb{R}^{n_a}$ is given by

$$
L(s,a) = s^\top Q s + a^\top R a,
$$

where $Q \in \mathbb{R}^{n_s \times n_s}$ and $R \in \mathbb{R}^{n_a \times n_a}$ are
suitable positive definite matrices. This is a very well-known problem in optimal
control theory. However, here, in the context of RL, these matrices are not known, and
we can only observe realizations of the cost for each state-action pair our controller
visits. The underlying system dynamics are described by the usual state-space model

$$
s_{k+1} = A s_k + B a_k,
$$

whose matrices $A \in \mathbb{R}^{n_s \times n_s}$ and
$B \in \mathbb{R}^{n_s \times n_a}$ could again in general be unknown. The control
action $a_k$ is assumed bounded in the interval $[-1,1]$. In what follows we will go
through the usual steps in setting up and solving such a task.

### Environment

The first ingredient to implement is the LTI system in the form of a `gymnasium.Env`
class. Fill free to fill in the missing parts based on your needs. The
`gymnasium.Env.reset` method should initialize the state of the system, while the
`gymnasium.Env.step` method should update the state of the system based on the action
provided and mainly return the new state and the cost.

```python
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
import numpy as np


class LtiSystem(Env):
    ns = ...  # number of states (must be continuous)
    na = ...  # number of actions (must be continuous)
    A = ...  # state-space matrix A
    B = ...  # state-space matrix B
    Q = ...  # state-cost matrix Q
    R = ...  # action-cost matrix R
    action_space = Box(-1.0, 1.0, (na,), np.float64)  # action space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.s = ...  # set initial state
        return self.s, {}

    def step(self, action):
        a = np.reshape(action, self.action_space.shape)
        assert self.action_space.contains(a)
        c = self.s.T @ self.Q @ self.s + a.T @ self.R @ a
        self.s = self.A @ self.s + self.B @ a
        return self.s, c, False, False, {}


# lastly, instantiate the environment with a wrapper to ensure the simulation finishes
env = TimeLimit(LtiSystem(), max_steps=5000)
```

### Controller

As aforementioned, we'd like to control this system via an MPC controller. Therefore,
the next step is to craft one. To do so, we leverage the `csnlp` package, in particular
its `csnlp.wrappers.Mpc` class (on top of that, under the hood, we exploit this package
also to compute the sensitivities of the MPC controller w.r.t. its parametrization,
which are crucial in calculating the RL updates). In mathematical terms, the MPC looks
like this:

$$
\begin{aligned}
  \min_{x_{0:N}, u_{0:N-1}} \quad & \sum_{i=0}^{N-1}{ x_i^\top \tilde{Q} x_i + u_i^\top \tilde{R} u_i } & \\
  \textrm{s.t.} \quad & x_0 = s_k \\
                      & x_{i+1} = \tilde{A} x_i + \tilde{B} u_i, \quad & i=0,\dots,N-1 \\
                      & -1 \le u_k \le 1, \quad & i=0,\dots,N-1
\end{aligned}
$$

where $\tilde{Q}, \tilde{R}, \tilde{A}, \tilde{B}$ do not necessarily have to match
the environment's $Q, R, A, B$ as they represent a possibly approximated a priori
knowledge on the sytem. In code, we can implement this as follows.

```python
import casadi as cs
from csnlp import Nlp
from csnlp.wrappers import Mpc

N = ...  # prediction horizon
mpc = Mpc[cs.SX](Nlp(), N)

# create the parametrization of the controller
nx, nu = LtiSystem.ns, LtiSystem.na
Atilde = mpc.parameter("Atilde", (nx, nx))
Btilde = mpc.parameter("Btilde", (nx, nu))
Qtilde = mpc.parameter("Qtilde", (nx, nx))
Rtilde = mpc.parameter("Rtilde", (nu, nu))

# create the variables of the controller
x, _ = mpc.state("x", nx)
u, _ = mpc.action("u", nu, lb=-1.0, ub=1.0)

# set the dynamics
mpc.set_linear_dynamics(Atilde, Btilde)

# set the objective
mpc.minimize(
    sum(cs.bilin(Qtilde, x[:, i]) + cs.bilin(Rtilde, u[:, i]) for i in range(N))
)

# initiliaze the solver with some options
opts = {
    "print_time": False,
    "bound_consistency": True,
    "calc_lam_x": True,
    "calc_lam_p": False,
    "ipopt": {"max_iter": 500, "sb": "yes", "print_level": 0},
}
mpc.init_solver(opts, solver="ipopt")
```

### Learning

The last step is to train the controller using an RL algorithm. For instance, here we
use Q-Learning. The idea is to let the controller interact with the environment, observe
the cost, and update the MPC parameters accordingly. This can be achieved by computing
the temporal difference error

$$
\delta_k = L(s_k, a_k) + \gamma V_\theta(s_{k+1}) - Q_\theta(s_k, a_k),
$$

where $\gamma$ is the discount factor, and $V_\theta$ and $Q_\theta$ are the state and
state-action value functions, both provided by the parametrized MPC controller with
$\theta = \{\tilde{A}, \tilde{B}, \tilde{Q}, \tilde{R}\}$. The update rule for the
parameters is then given by

$$
\theta \gets \theta + \alpha \delta_k \nabla_\theta Q_\theta(s_k, a_k),
$$

where $\alpha$ is the learning rate, and $\nabla_\theta Q_\theta(s_k, a_k)$ is the
sensitivity of the state-action value function w.r.t. the parameters. All of this can be
implemented as follows.

```python
from mpcrl import LearnableParameter, LearnableParametersDict, LstdQLearningAgent
from mpcrl.optim import GradientDescent

# give some initial values to the learnable parameters (shapes must match!)
learnable_pars_init = {"Atilde": ..., "Btilde": ..., "Qtilde": ..., "Rtilde": ...}

# create the set of parameters that should be learnt
learnable_pars = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
        for name, val in learnable_pars_init.items()
    )
)

# instantiate the learning agent
agent = LstdQLearningAgent(
    mpc=mpc,
    learnable_parameters=learnable_pars,
    discount_factor=...,  # a number in (0,1], e.g.,  1.0
    update_strategy=...,  # an integer, e.g., 1
    optimizer=GradientDescent(learning_rate=...),
    record_td_errors=True,
)

# finally, launch the training for 5000 timesteps. The method will return an array of
# (hopefully) decreasing costs
costs = agent.train(env=env, episodes=1, seed=69)
```

---

## Examples

Our
[examples](https://mpc-reinforcement-learning.readthedocs.io/en/stable/auto_examples/index.html)
subdirectory contains examples on how to use the library on some academic, small-scale
application (a small linear time-invariant (LTI) system), tackled both with
[on-policy Q-learning](https://mpc-reinforcement-learning.readthedocs.io/en/stable/auto_examples/gradient-based-onpolicy/q_learning.html#sphx-glr-auto-examples-gradient-based-onpolicy-q-learning-py),
[off-policy Q-learning](https://mpc-reinforcement-learning.readthedocs.io/en/stable/auto_examples/gradient-based-offpolicy/q_learning_offpolicy.html#sphx-glr-auto-examples-gradient-based-offpolicy-q-learning-offpolicy-py)
and
[DPG](https://mpc-reinforcement-learning.readthedocs.io/en/stable/auto_examples/gradient-based-onpolicy/dpg.html#sphx-glr-auto-examples-gradient-based-onpolicy-dpg-py).
While the aforementioned algorithms are all gradient-based, we also provide an
[example on how to use Bayesian Optimization (BO)](https://mpc-reinforcement-learning.readthedocs.io/en/stable/auto_examples/gradient-free/bayesopt.html#sphx-glr-auto-examples-gradient-free-bayesopt-py)
[[6]](#6) to tune the MPC parameters in a gradient-free way.

---

## License

The repository is provided under the MIT License. See the LICENSE file included with
this repository.

---

## Author

[Filippo Airaldi](https://www.tudelft.nl/staff/f.airaldi/), PhD Candidate
[f.airaldi@tudelft.nl | filippoairaldi@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/me/about/departments/delft-center-for-systems-and-control/)
in [Delft University of Technology](https://www.tudelft.nl/en/)

Copyright (c) 2024 Filippo Airaldi.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest
in the program “mpcrl” (Reinforcement Learning with Model Predictive Control) written by
the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of ME.

---

## References

<a id="1">[1]</a>
Sutton, R.S. and Barto, A.G. (2018).
[Reinforcement learning: An introduction](https://mitpress-mit-edu.tudelft.idm.oclc.org/9780262039246/reinforcement-learning/).
Cambridge, MIT press.

<a id="2">[2]</a>
Rawlings, J.B., Mayne, D.Q. and Diehl, M. (2017).
[Model Predictive Control: theory, computation, and design (Vol. 2)](https://sites.engineering.ucsb.edu/~jbraw/mpc/).
Madison, WI: Nob Hill Publishing.

<a id="3">[3]</a>
Gros, S. and Zanon, M. (2020).
[Data-Driven Economic NMPC Using Reinforcement Learning](https://ieeexplore-ieee-org.tudelft.idm.oclc.org/document/8701462).
IEEE Transactions on Automatic Control, 65(2), 636-648.

<a id="4">[4]</a>
Esfahani, H. N. and Kordabad,  A. B. and Gros, S. (2021).
[Approximate Robust NMPC using Reinforcement Learning](https://ieeexplore-ieee-org.tudelft.idm.oclc.org/document/9655129).
European Control Conference (ECC), 132-137.

<a id="5">[5]</a>
Cai, W. and Kordabad, A. B. and Esfahani, H. N. and Lekkas, A. M. and Gros, S. (2021).
[MPC-based Reinforcement Learning for a Simplified Freight Mission of Autonomous Surface Vehicles](https://ieeexplore-ieee-org.tudelft.idm.oclc.org/document/9683750).
60th IEEE Conference on Decision and Control (CDC), 2990-2995.

<a id="6">[6]</a>
Garnett, R., 2023. [Bayesian Optimization](https://bayesoptbook.com/).
Cambridge University Press.

<a id="7">[7]</a>
Gros, S. and Zanon, M. (2022).
[Learning for MPC with stability & safety guarantees](https://www.sciencedirect.com/science/article/pii/S0005109822004605).
Automatica, 164, 110598.

<a id="8">[8]</a>
Zanon, M. and Gros, S. (2021).
[Safe Reinforcement Learning Using Robust MPC](https://ieeexplore.ieee.org/abstract/document/9198135/).
IEEE Transactions on Automatic Control, 66(8), 3638-3652.
