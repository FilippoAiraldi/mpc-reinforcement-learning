================
Module reference
================

This section contains detailed information about the modules and classes in **mpcrl**.
First, we introduce the main components of the package, which are likely to be the  ones
most users will employe. Then, the rest of the package is presented in details.

.. currentmodule:: mpcrl

---------------
Core components
---------------

Here we list the core elements that are used to train and evaluate agents, but are not
the agents themselves, which are presented in the next section. These core components
are paramount in laying the foundation for the agents to be built upon.

# TODO: core and optim here

-----------
Base agents
-----------

What follows are the base classes for the agents in the package. These are either
nonlearning agents (i.e., :class:`Agent`) or abstract learning agents that provide the
layout for inheriting classes.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :caption: Base agents

   Agent
   LearningAgent
   RlLearningAgent

-----------------------------
Reinforcement Learning agents
-----------------------------

These are the learning agents that leverage a reinforcement learning algorithm to tune
the parametrization of the MPC controller. Two very common algorithms are here
implemented: Q-learning and Deterministic Policy Gradient (DPG).

.. autosummary::
   :toctree: generated
   :template: class.rst
   :caption: Reinforcement Learning agents

   LstdDpgAgent
   LstdQLearningAgent

---------------------
Other learning agents
---------------------

We also provide other learning agents that do not use gradient-based approaches to
update their parameters, but rather rely on other global gradient-free optimization
techniques. See also class:`optim.GradientFreeOptimizer`.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :caption: Other learning agents

   GlobOptLearningAgent

----------------
Other submodules
----------------

.. autosummary::
   :toctree: generated
   :template: module.rst
   :caption: Other components

   util

# TODO: wrappers here
