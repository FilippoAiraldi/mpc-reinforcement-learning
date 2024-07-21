------------------------------------------------------
A simple Reinforcement Learning task for an LTI system
------------------------------------------------------

Here we provide the skeleton of a simple application of the library. The aim of the code
below is to let an MPC control strategy learn how to optimally control a simple Linear
Time Invariant (LTI) system. The cost (i.e., the opposite of the reward) of controlling
this system in state :math:`s \in \mathbb{R}^{n_s}` with action
:math:`a \in \mathbb{R}^{n_a}` is given by

.. math:: L(s,a) = s^\top Q s + a^\top R a,

where :math:`Q \in \mathbb{R}^{n_s \times n_s}` and
:math:`R \in \mathbb{R}^{n_a \times n_a}` are suitable positive definite matrices. This
is a very well-known problem in optimal control theory. However, here, in the context of
RL, these matrices are not known, and we can only observe realizations of the cost for
each state-action pair our controller visits. The underlying system dynamics are
described by the usual state-space model

.. math:: s_{k+1} = A s_k + B a_k,

whose matrices :math:`A \in \mathbb{R}^{n_s \times n_s}` and
:math:`B \in \mathbb{R}^{n_s \times n_a}` could again in general be unknown. The control
action :math:`a_k` is assumed bounded in the interval :math:`[-1,1]`. In what follows we
will go through the usual steps in setting up and solving such a task.

Environment
===========

The first ingredient to implement is the LTI system in the form of a
:class:`gymnasium.Env` class. Fill free to fill in the missing parts based on your
needs. The :func:`gymnasium.Env.reset` method should initialize the state of the system,
while the :func:`gymnasium.Env.step` method should update the state of the system based
on the action provided and mainly return the new state and the cost.

.. code:: python

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


Controller
==========

As aforementioned, we'd like to control this system via an MPC controller. Therefore,
the next step is to craft one. To do so, we leverage the :mod:`csnlp` package, in
particular its :class:`csnlp.wrappers.Mpc` class (on top of that, under the hood, we
exploit this package also to compute the sensitivities of the MPC controller w.r.t. its
parametrization, which are crucial in calculating the RL updates). In mathematical
terms, the MPC looks like this:

.. math::
   \begin{aligned}
   \min_{x_{0:N}, u_{0:N-1}} \quad & \sum_{i=0}^{N-1}{ x_i^\top \tilde{Q} x_i + u_i^\top \tilde{R} u_i } & \\
   \textrm{s.t.} \quad & x_0 = s_k \\
                        & x_{i+1} = \tilde{A} x_i + \tilde{B} u_i, \quad & i=0,\dots,N-1 \\
                        & -1 \le u_k \le 1, \quad & i=0,\dots,N-1
   \end{aligned}

where :math:`\tilde{Q}, \tilde{R}, \tilde{A}, \tilde{B}` do not necessarily have to
match the environment's :math:`Q, R, A, B` as they represent a possibly approximated a
priori knowledge on the sytem. In code, we can implement this as follows.

.. code:: python

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
   mpc.set_dynamics(lambda x, u: Atilde @ x + Btilde @ u, n_in=2, n_out=1)

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

Learning
========

The last step is to train the controller using an RL algorithm. For instance, here we
use Q-Learning. The idea is to let the controller interact with the environment, observe
the cost, and update the MPC parameters accordingly. This can be achieved by computing
the temporal difference error

.. math:: \delta_k = L(s_k, a_k) + \gamma V_\theta(s_{k+1}) - Q_\theta(s_k, a_k),

where :math:`\gamma` is the discount factor, and :math:`V_\theta` and :math:`Q_\theta`
are the state and state-action value functions, both provided by the parametrized MPC
controller with :math:`\theta = \{\tilde{A}, \tilde{B}, \tilde{Q}, \tilde{R}\}`. The
update rule for the parameters is then given by

.. math:: \theta \gets \theta + \alpha \delta_k \nabla_\theta Q_\theta(s_k, a_k),

where :math:`\alpha` is the learning rate, and :math:`\nabla_\theta Q_\theta(s_k, a_k)`
is the sensitivity of the state-action value function w.r.t. the parameters. All of this
can be implemented as follows.

.. code:: python

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
