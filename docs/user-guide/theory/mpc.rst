------------------------
Model Predictive Control
------------------------

Model Predictive Control (MPC) :cite:`rawlings_model_2017` is a well-known predictive
methodology in the context of optimal control. Why is it predictive? Because at its core
lies a model of the system (i.e., a mathematical description of the dynamics of the
system  the MPC controller is supposed to control) that is used to predict how said
system will evolve when different control actions are applied to it. Then, the MPC
scheme selects the optimal actions to apply to the system based on a user-defined cost
criterion and constraints. Most usually, MPC controllers are used in a closed-loop
setup and in a receding horizon fashion: at each time step, the controller receives a
measurement of the current state of the system, solves the aforementioned optimization
problem to find the sequence of optimal actions to apply to the system, applies only the
first optimal action from this sequence to the system, and then waits for the next
measurement to repeat the process.

Mathematically speaking, consider a discrete-time system described at time step
:math:`k` by

.. math:: s_{k+1} = f(s_k,a_k),

where :math:`f` represent the (possibly nonlinear and/or stochastic) dynamics of the
system/environment, :math:`s_k` is its state, and :math:`a_k` the action applied to it.
A very generic MPC controller looks like

.. math::
   \begin{aligned}
      \min_{x_{0:N}, u_{0:N-1}} \quad &
         \lambda(x_0) + \sum_{i=0}^{N-1}{ \gamma^i \ell(x_i,u_i) }
         + \gamma^N T(x_N) & \\
      \textrm{s.t.} \quad & x_0 = s_k \\
                          & x_{i+1} = f(x_i, u_i) \quad & i=0,\dots,N-1 \\
                          & h(x_i,u_i) \leq 0 \quad & i=0,\dots,N-1 \\
                          & h_f(x_N) \leq 0,
   \end{aligned}

where :math:`N` is the so-called prediction horizon, :math:`x_i` and :math:`u_i` are the
states and actions at time step :math:`i` over the horizon, :math:`\lambda` is the
initial cost function, :math:`\ell` the stage cost, :math:`T` the terminal cost, and
:math:`h` and :math:`h_f` are inequality constraints. The solution to this optimization
problem, :math:`u_0^\star`, is the action that is then applied to the system, i.e.,
:math:`a_k = u_0^\star`.

Parametric MPC
==============

More often than not, the dynamics of the system are not known exactly, but is known to
belong to (or at least, is well-approximated by) a parametric family of models
:math:`f_\theta`, where :math:`\theta` usually refers to such parameters. Likewise, the
cost terms and constraints are usually also functions of some parameters that the
designer can tune. In this case, the MPC problem, parametrized by :math:`\theta`, is
generically described by

.. math::
   \begin{aligned}
      \min_{x_{0:N}, u_{0:N-1}} \quad &
         \lambda_\theta(x_0) + \sum_{i=0}^{N-1}{ \gamma^i \ell_\theta(x_i,u_i) }
         + \gamma^N T_\theta(x_N) & \\
      \textrm{s.t.} \quad & x_0 = s_k \\
                          & x_{i+1} = f_\theta(x_i, u_i) \quad & i=0,\dots,N-1 \\
                          & h_\theta(x_i,u_i) \leq 0 \quad & i=0,\dots,N-1 \\
                          & h_{f,\theta}(x_N) \leq 0,
   \end{aligned}

The challenge in designing a good MPC controller is then to find the optimal parameters
(where we still have to define what "optimal" means) that will make the controller
perform well in practice when deployed onto the system. This is usually done by manual
tuning; however, in the context of high nonlinearities, stochasticity, or high dimension
of the system, this can be a daunting task. This is where reinforcement learning will
come in handy.
