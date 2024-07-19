------------
Introduction
------------

This framework, also referred to as *RL with/using MPC*, was first proposed in
:cite:`gros_datadriven_2020` and has so far been shown effective in various
applications, with different learning algorithms and more sound theory, e.g.,
:cite:`cai_mpcbased_2021,esfahani_approximate_2021,zanon_safe_2021,gros_learning_2022`.
It merges two powerful control techinques into a single data-driven one

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
:cite:`esfahani_approximate_2021` and Deterministic Policy Gradient (DPG)
:cite:`cai_mpcbased_2021`.

.. figure:: _static/mpcrl.diagram.light.png
   :alt: Main idea behind the MPC-RL framework
   :align: center
   :width: 50%
   :class: only-light

   Diagram of the MPC-RL framework.

.. figure:: _static/mpcrl.diagram.dark.png
   :alt: Main idea behind the MPC-RL framework
   :align: center
   :width: 50%
   :class: only-dark

   Diagram of the MPC-RL framework.
