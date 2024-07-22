================
Module reference
================

This section contains detailed information about the modules and classes in **mpcrl**.
First, we introduce the main components of the package, which are likely to be the  ones
most users will employe. Then, the rest of the package is presented in details.

.. currentmodule:: mpcrl

------
Agents
------

Base agents
===========

What follows are the base classes for the agents in the package. These are either
nonlearning agents (i.e., :class:`Agent`) or abstract learning agents that provide the
layout for inheriting classes.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :caption: Base agents
   :nosignatures:

   Agent
   LearningAgent
   RlLearningAgent


Reinforcement Learning agents
=============================

These are the learning agents that leverage a reinforcement learning algorithm to tune
the parametrization of the MPC controller. Two very common algorithms are here
implemented: Q-learning and Deterministic Policy Gradient (DPG).

.. autosummary::
   :toctree: generated
   :template: class.rst
   :caption: Reinforcement Learning agents
   :nosignatures:

   LstdDpgAgent
   LstdQLearningAgent

Other learning agents
=====================

We also provide other learning agents that do not use gradient-based approaches to
update their parameters, but rather rely on other global gradient-free optimization
techniques. See also class:`optim.GradientFreeOptimizer`.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :caption: Other learning agents
   :nosignatures:

   GlobOptLearningAgent

----------------
Other submodules
----------------

.. currentmodule:: mpcrl

:mod:`mpcrl` offers a few other components that are not explicitly needed by the agents
and their core functionalities, but can be useful to enhance the base behaviour of
agents via wrappers, or to provide additional methods for, e.g., designing LQR
controllers. To this end, we provide a few utility wrapper classes and utility methods
in the following submodules.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :caption: Other components

   wrappers
   util
