-----------------------------------------------------
Reinforcement Learning using Model Predictive Control
-----------------------------------------------------

If you have followed along, it is not difficult to see that the parametric MPC scheme
that was discussed above is more than suitable for being used as a policy provider in
the context of RL. This concept was first introduced and properly formulated in
:cite:`gros_datadriven_2020`.


MPC as function approximation
=============================

In fact, the MPC scheme naturally acts as a policy provider, so the definition of its
policy in the context of RL follows naturally as

.. math:: a_k = u_0^\star = \pi_\theta(s_k).

What's more, as shown in :cite:`gros_datadriven_2020`, the MPC controller can also be
employed to approximate the value functions as

.. math::
   \begin{aligned}
      V_\theta(s_k) = \min_{x_{0:N}, u_{0:N-1}} \quad &
         \lambda_\theta(x_0) + \sum_{i=0}^{N-1}{ \gamma^i \ell_\theta(x_i,u_i) }
         + \gamma^N T_\theta(x_N) & \\
      \textrm{s.t.} \quad & x_0 = s_k \\
                          & x_{i+1} = f_\theta(x_i, u_i) \quad & i=0,\dots,N-1 \\
                          & h_\theta(x_i,u_i) \leq 0 \quad & i=0,\dots,N-1 \\
                          & h_{f,\theta}(x_N) \leq 0,
   \end{aligned}

and

.. math::
   \begin{aligned}
      Q_\theta(s_k,a_k) = \min_{x_{0:N}, u_{0:N-1}} \quad &
         \lambda_\theta(x_0) + \sum_{i=0}^{N-1}{ \gamma^i \ell_\theta(x_i,u_i) }
         + \gamma^N T_\theta(x_N) & \\
      \textrm{s.t.} \quad & x_0 = s_k \\
                          & a_0 = a_k \\
                          & x_{i+1} = f_\theta(x_i, u_i) \quad & i=0,\dots,N-1 \\
                          & h_\theta(x_i,u_i) \leq 0 \quad & i=0,\dots,N-1 \\
                          & h_{f,\theta}(x_N) \leq 0.
   \end{aligned}

The Bellman relationships also hold, with

.. math::
   \pi_\theta(s) = \arg\min_{a \in \mathbb{A}} Q_\theta(s,a), \quad
   V_\theta(s) = \arg\min_{a \in \mathbb{A}} Q_\theta(s,a).

However, approximating the policy and the value functions with some function
approximation scheme is only half of the story. The other half is to understand how to
adjust the parameters :math:`\theta` of such paramatric approximation in order to
improve the RL performance and minimize the incurred costs. This is where famous
gradient-based RL algorithms come into play. Nevertheless, to apply these algorithms,
the gradient of the MPC quantities w.r.t. the parameters :math:`\theta` must be
evaluated. To do so, :cite:`gros_datadriven_2020` proposed to leverage nonlinear
sensitivity analysis techniques :cite:`buskens_sensitivity_2001` that exploit the KKT
conditions to compute such sensitivities.

Q-learning with MPC
===================

Unsurprisingly, the Q-learning algorithm can be employed to tune the parameters of the
MPC controller to improve its performance. The underlying idea of Q-learning is to
approximate as best as possible the unknown optimal Q-function :math:`Q^\star` by
minimizing the Bellman residual, i.e.,

.. math::
   \min_{\theta \in \Theta} \mathbb{E} \left[
      \left\lVert Q^\star(s,a) - Q_\theta(s,a) \right\rVert^2
   \right].

This can be (approximately) achieved with the famous update rule

.. math:: \theta \leftarrow \theta + \alpha \delta_k \nabla_\theta Q_\theta(s_k,a_k),

where :math:`\alpha` is the learning rate and :math:`\delta_k` is the Temporal
Difference (TD) error at time step :math:`k`. :cite:`esfahani_approximate_2021` improves
upon the update above and embeds it with second order information, i.e., it includes not
only the gradient of the approximation, but also an estimate of its hessian.

So far, these concepts are pretty standard to Q-learning. The real question is, how can
we compute :math:`\nabla_\theta Q_\theta(s_k,a_k)` when the action value function is
provided by an MPC optimization scheme? It turns out that the answer is not very
complex, and according to :cite:`buskens_sensitivity_2001` we have that

.. math:: \nabla_\theta Q_\theta(s_k,a_k) =
      \nabla_\theta \mathcal{L}(y^\star, \theta),

where :math:`\mathcal{L}_\theta` is the Lagrangian of the MPC optimization problem
evaluated at the optimal primal-dual solution :math:`y^\star` of the NLP problem.


Deterministic Policy Gradient with MPC
======================================

What if, instead of learning the optimal Q-function from data with the hope to
inderectly recover the optimal policy from it, we directly learn the policy that
minimizes the returns directly? This is the idea behind policy gradient methods, which
attempt to estimate :math:`\nabla_\theta J(\pi_\theta)` and use it to update the
parametrization. In other words, the update rule is

.. math:: \theta \leftarrow \theta - \alpha \nabla_\theta J(\pi_\theta).

In particular, :cite:`cai_mpcbased_2021` shows how to use the Deterministic Policy
Gradient (DPG) algorithm. Estimation of the performance gradient is not trivial, but can
be achieved as

.. math::
      \nabla_\theta J(\pi_\theta) = \mathbb{E} \left[
         \nabla_\theta \pi_\theta(s) \nabla_a Q_{\pi_\theta}(s,a) |_{a=\pi_\theta(s)}
      \right].

The gradient of the policy function can be computed as

.. math::
      \nabla_\theta \pi_\theta(s) = -\nabla_\theta K(y^\star,s,\theta)
         \nabla_y K(y^\star,s,\theta)^{-1} \frac{\partial y}{\partial u_0}

where :math:`y` are all the primal-dual variables, and :math:`K` is the KKT system of
optimal conditions associated with the MPC optimization problem. The action-value
function is instead approximated with the compatible form

.. math::
      Q_{\pi_\theta} \approx Q_\omega = \Psi(s,a)^\top \omega + V_\nu(s)

with :math:`\Psi(s,a) = \nabla_\theta \pi_\theta(s) (a - \pi_\theta(s))` and
:math:`V_{\pi_\theta} \approx V_\nu = \Phi(s)^\top \nu`. :math:`\Phi(s)` is a state
feature vector. Hence, we get that

.. math::
      \nabla_a Q_{\pi_\theta}(s,a) \approx \nabla_a Q_\omega(s,a)
      = \nabla_\theta \pi_\theta(s)^\top \omega.

The unknown parameters :math:`\omega` and :math:`\nu` can be computed in a batch way via
a least-squares regression problem.
