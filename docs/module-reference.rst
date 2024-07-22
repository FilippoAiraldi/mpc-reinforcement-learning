================
Module reference
================

This section contains detailed information about the modules and classes in **mpcrl**.
First, we introduce the main components of the package, which are likely to be the  ones
most users will employe. Then, the rest of the package is presented in details.

Scheduling quantities
---------------------

What if you need to decay or increase your learning rate over time during training?
The following submodule provides a set of schedulers that can be used to update or decay
different quantities, such as learning rates or exploration probability, over time. Most
of the agents will then accept a scheduler as an argument, which will be updated
according to the user-specified way.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :caption: Scheduling quantities

   schedulers

Exceptions
----------

Finally, we also provide two custom warnings and exceptions to signal two distinct and
important events, namely, when the MPC solver fails to find a solution, and when the
update fails (usually the QP solver fails to find a solution). Since the methods
:meth:`mpcrl.Agent.evaluate`, :meth:`mpcrl.LearningAgent.train` and
:meth:`mpcrl.LearningAgent.train_offpolicy` accept the ``raises`` argument, we provide
here both warnings and exceptions that can be raised in case of failures, depending on
the value of said flag. We also provide two utility functions to conveniently raise
these exceptions or warnings.

.. autosummary::
   :toctree: generated
   :template: module.rst
   :caption: Exceptions

   errors

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

---------------
Core components
---------------

Here we list core elements that are used during training and evaluation of agents, but
are not the agents themselves. These can be employed by users to customize the
hyperparameters of the training process, or to modify the behaviour of the agents.

----------
Optimizers
----------

.. automodule:: mpcrl.optim

.. inheritance-diagram::
   mpcrl.optim.base_optimizer.BaseOptimizer
   GradientFreeOptimizer
   GradientBasedOptimizer
   Adam
   GradientDescent
   NetwonMethod
   RMSprop
   :parts: 1

Base optimizers
===============

These are the base abstract optimizer classes that lay the skeleton for the
gradient-based updates of the MPC parametrization. We also offer an interface for
gradient-free optimizers, which can be used to tune the parameters of the MPC controller
via global optimization strategies such as Bayesian Optimization.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :caption: Base optimizers
   :nosignatures:

   mpcrl.optim.base_optimizer.BaseOptimizer
   GradientBasedOptimizer
   GradientFreeOptimizer

Gradient-based optimizers
=========================

.. autosummary::
   :toctree: generated
   :template: class.rst
   :caption: Gradient-based optimizers
   :nosignatures:

   GradientDescent
   NetwonMethod
   Adam
   RMSprop

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
