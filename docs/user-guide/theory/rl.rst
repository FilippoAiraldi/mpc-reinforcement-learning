Before we discuss the inner workings of the library, an introduction to the theory
behind the MPC-based RL framework is necessary. This section will provide a brief
overview of the main concepts and ideas that are at its core, and the mathmeticas behind
them.

----------------------
Reinforcement Learning
----------------------

Reinforcement Learning (RL) :cite:`sutton_reinforcement_2018` is a subfield of Machine
Learning that deals with the problem of learning how to make decisions in an environment
in order to maximize some rewards or, as in the context of control, to minimize some
costs. It is heavily related to optimal control theory and Dynamic Programming (DP), so
much so that sometimes it is difficult to distinguish between them. In fact, all of
these fields are concerned with *learning a policy* that will dictate the actions to
take in order to achieve some goal in the given system/environment.

Consider a Markov Decision Process (MDP) defined by state :math:`s`, action :math:`a`,
and a state transition :math:`s \xrightarrow{a} s_+` with the underlying conditional
probability density

.. math::
   \mathbb{P}\left[s_+ | s, a\right] :
   \mathbb{S} \times \mathbb{S} \times \mathbb{A} \rightarrow \left[0, 1\right]

where :math:`\mathbb{S}` and :math:`\mathbb{A}` are the state and action space,
respectively. Such MDP is very generic and can represent, e.g., the model of a
discrete-time system. The performance of a given deterministic policy
:math:`\pi_\theta : \mathbb{S} \rightarrow \mathbb{A}`, parametrized in
:math:`\theta \in \Theta`,
is defined as

.. math::
   J(\pi_\theta) := \mathbb{E} \left[
      \sum_{k=0}^{\infty}{\gamma^k L \bigl(s_k, \pi_\theta(s_k)\bigr)}
   \right]

where :math:`\gamma \in (0,1]` is the discount factor, and
:math:`L : \mathbb{S} \times \mathbb{A} \rightarrow \mathbb{R}` is the stage-cost
function. The goal of RL is then to find the optimal policy

.. math:: \pi_\theta^\star = \arg\min_{\theta \in \Theta} J(\pi_\theta)

by learning from the interaction with the environment. In other words, the algorithm is
only allowed to observed, for each state and action pair :math:`s_k,a_k`, the immediate
cost realization :math:`L(s_k,a_k)` and the next state :math:`s_{k+1}`. Aside the
policy, other important quantities to introduce (here defined in their parametric form,
but the general case holds) are the state value function :math:`V_\theta(s)` and the
state-action value function :math:`Q_\theta(s,a)`. The former is defined as

.. math::
   V_\theta(s) := \mathbb{E} \left[
      \sum_{k=0}^{\infty}{\gamma^k L \bigl(s_k, \pi_\theta(s_k)\bigr) \ | \ s_0 = s}
   \right]

and is used to evaluate the performance of the policy given the current state (which is
imposed as initial), while the latter is defined as

.. math::
   Q_\theta(s,a) := \mathbb{E} \left[
      \sum_{k=0}^{\infty}{\gamma^k L \bigl(s_k, a_k\bigr) \ | \ s_0 = s, a_0 = a}
   \right]

and evaluates the performance of the policy given the current state as well as the
first action. Since it is in general impossible to find and characterise the true
unknown optimal policy and value functions, function approximation techniques (such as
neural networks and, as in this library, MPC) have been employed as a powerful
alternative for tackling this problem.
