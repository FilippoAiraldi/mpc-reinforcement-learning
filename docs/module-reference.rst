================
Module reference
================

This page contains all the detailed information about the modules and classes in
:mod:`mpcrl`. First, we will indulge in presenting the core components of the library
that allow us to easily implement Reinforcement Learning algorithms. Then, we will move
to the agents themselves, which contain these algorithms and deploy them to control the
given environments (and possibly learn from interacting with it). Finally, the different
optimization strategies that can be used to update the parameters of the MPC controller
are reported, and the utility functions and wrappers that can be used to enhance the
behaviour of the agents are also presented.

---------------
Core components
---------------

Before jumping into the details of the agents and their Reinforcement Learning
algorithms, we present here the core elements that are used during training and
evaluation, but are not the agents themselves.

.. automodule:: mpcrl.core

We'll start first with the latter, i.e., the building blocks of the library, and only
then move to the other former, i.e., the other core elements that allow to specify
the hyperparameters for our agents.

Building blocks
===============

In this section, we present the building blocks of the package, which are at the core
of the internal workings of the agents and their learning algorithms. These include the
callback mechanisms, the learnable parameters, the scheduling quantities, and our
custom exceptions and warnings.

Callbacks
---------

.. automodule:: mpcrl.core.callbacks

.. autosummary::
   :toctree: generated
   :template: class.rst
   :caption: Callbacks
   :nosignatures:

   CallbackMixin
   AgentCallbackMixin
   LearningAgentCallbackMixin

Learnable parameters
--------------------

.. automodule:: mpcrl.core.parameters

.. currentmodule:: mpcrl

.. autosummary::
   :toctree: generated
   :template: class.rst
   :caption: Learnable parameters
   :nosignatures:

   LearnableParameter
   LearnableParametersDict

Scheduling quantities
---------------------

What if you need to decay or increase your learning rate over time during training?
The following submodule provides a set of schedulers that can be used to update or decay
different quantities, such as learning rates or exploration probability, over time. Most
of the agents will then accept a scheduler as an argument, which will be updated
according to the user-specified way.

.. currentmodule:: mpcrl.core

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

Hyperparameters
===============

Update strategy
---------------

.. automodule:: mpcrl.core.update

.. currentmodule:: mpcrl

.. autosummary::
   :toctree: generated
   :template: class.rst
   :caption: Update strategy
   :nosignatures:

   UpdateStrategy

Experience replay
-----------------

.. automodule:: mpcrl.core.experience

.. currentmodule:: mpcrl

.. autosummary::
   :toctree: generated
   :template: class.rst
   :caption: Experience replay
   :nosignatures:

   ExperienceReplay

Exploring
---------

.. automodule:: mpcrl.core.exploration

.. autosummary::
   :toctree: generated
   :template: class.rst
   :caption: Exploring
   :nosignatures:

   ExplorationStrategy
   NoExploration
   GreedyExploration
   EpsilonGreedyExploration
   OrnsteinUhlenbeckExploration
   StepWiseExploration

Warmstarting the MPC solvers
----------------------------

.. automodule:: mpcrl.core.warmstart

.. currentmodule:: mpcrl

.. autosummary::
   :toctree: generated
   :template: class.rst
   :caption: Warmstarting the MPC solvers
   :nosignatures:

   WarmStartStrategy

.. _module_reference_agents:

------
Agents
------

Agents are the main and, arguably, the most important components of the package. They
deploy the control policies to control the given environments, and, if they are
learning-based, also implement the underlying learning algorithm to tune the parameters
of the control policies.

.. currentmodule:: mpcrl

.. inheritance-diagram::
   Agent
   LearningAgent
   GlobOptLearningAgent
   RlLearningAgent
   LstdDpgAgent
   LstdQLearningAgent
   :parts: 1

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

Here instead are reported the concrete implementations of the gradient-based optimizers
that can be used to update the parameters of the MPC controller. They include both
first-order and second-order methods, whether they require and make use of gradient and
curvature information (i.e., Jacobian and Hessian of some quantity w.r.t. to the
parameters).

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
