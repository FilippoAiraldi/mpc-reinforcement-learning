---------------------
Inheritance hierarchy
---------------------

The main components of the MPC-RL framework are the agents, which are responsible for
interacting with the environment and, in case they are able to do so, learning the
optimal policy from these interactions.

But before jumping into the details of the different agents, it is important to
understand the hierarchy of the different classes that are used to implement the agents
and their relationships. The following diagram shows the inheritance scheme of the
different agents.

.. currentmodule:: mpcrl

.. inheritance-diagram::
   Agent
   LearningAgent
   GlobOptLearningAgent
   RlLearningAgent
   LstdDpgAgent
   LstdQLearningAgent
   :parts: 1


Callbacks
=========

While some of the classes in the diagram are outside the scope of this documentation,
let us notice that the :class:`Agent` and the :class:`LearningAgent`
classes inherit from the mixins :class:`core.callbacks.CallbackMixin` and
:class:`core.callbacks.LearningAgentCallbackMixin`, respectively. These base
classes are fundamental to the implementation of the agents as they provide the backbone
for other functionalities, such as updates and schedulers, to be hooked into each agent
and be called with a specific frequency, e.g., at the end of every episode or after 100
time steps. Of course, this is internally vital for learning agents, as they need to
update their parametrization with a given frequency. Nonetheless, also end users can
benefit from these callbacks: they allow to implement logic that needs to be executed
when specific events occur, e.g., updating disturbance profiles, changing references,
etc.. This topic is discussed further in :ref:`user_guide`'s :ref:`user_guide_callbacks`
and in :ref:`module_reference`'s :ref:`module_reference_callbacks`.


Agents
======

Now, for the agents! As seen in the diagram above, the simplest agent class is
:class:`Agent`. This class implements a basic agent that can interact with an
environment, but not learn from it. From there, the abstract classes
:class:`LearningAgent` and :class:`RlLearningAgent`
are derived, which introduce learning capabilities to the agents. Parallel to latter,
which is oriented towards gradient-based RL solutions, the abstract
:class:`GlobOptLearningAgent` defines the layout for agents that leverage Global
Optimzation (i.e., gradient-free) strategies rather than gradient-based ones. Finally,
the concrete classes :class:`LstdDpgAgent` and :class:`LstdQLearningAgent`
implement the DPG and Q-learning algorithms, respectively, to tune the MPC controller's
parameters. More details about all these agents can be found in the next sections of the
:ref:`user_guide`.
