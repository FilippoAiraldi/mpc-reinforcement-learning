.. currentmodule:: mpcrl

------------------
Base agent classes
------------------

In this section we discuss the various base agent clases that are available in :mod:`mpcrl`
in more details.



:class:`Agent`
--------------

As aforementioned, the basic agent is implemented in :class:`Agent`. This class takes in
an MPC controller, but has no feature for learning from the environment. In fact, it can
evaluate the controller's performance in the environment via the :meth:`Agent.evaluate`,
but it does not accept a :class:`LearnableParametersDict` instance at instantiation,
because it cannot learn. However, :meth:`Agent.__init__` does accept a
``fixed_parameters`` argument that the user can leverage to pass a dictionary of fixed,
non-learned parameters of the MPC controller to the agent.

That said, the :class:`Agent` class provides a method for evaluation of both the
state-value function :math:`V_\theta(s)` and the action-value function
:math:`Q_\theta(s,a)`, both of which are computed with the MPC controller as the
underlying function approximation with parameters :math:`\theta`, though only the
former is used in the evaluation of the agent's performance. See
:meth:`Agent.state_value` and :meth:`Agent.action_value` respectively.

Moreover, the agent accepts various other arguments at instantiation, allowing for
further customization of its behaviour

* ``exploration``: whether and how the MPC policy should be perturbed to induce
  exploration.
* ``warmstart``: how the MPC optimization problem should be warmstarted, if at all.
  This is especially useful when the optimization is highly nonlinear.
* ``use_last_action_on_fail`` specifies how to handle failures of the solver, and which
  action to pass to the environment in such cases.
* ``remove_bounds_on_initial_action`` removes automatically the bounds on the initial
  action in the action-value function approximation.

Overall, this class lays the foundation for the rest of the learning agents, but is not
the focus of the library. Nevertheless, it comes in handy for benchmarking and testing the
MPC controller in the environment prior to the application of any learning strategy, or
to generate expert off-policy rollout sequences.



:class:`LearningAgent`
----------------------

Still an abstract class, :class:`LearningAgent` inherits from :class:`Agent` and adds
features enabling the agent to learn from interactions with the environment. However,
being abstract, it cannot be instantiated directly, and it only offers the foundation
for other concrete implementations. It is agnostic of the underlying learning method,
so it is not meant to be restricted to RL algorithms, but any learning algorithms.

Its constructor adds the following arguments on top of those from :class:`Agent`:

- ``learnable_parameters``: first and foremost, it accepts at instantiation an instance
  of :class:`LearnableParametersDict`. This class is further discussed in
  :ref:`user_guide_learnable_parameters`, but it suffices to say that it must be used to
  indicate the subset of the MPC parametrization that should be learned via the
  learning algorithm.

- ``update_strategy``: this argument defines the update strategy of the learning agent,
  i.e., when and with what frequencies should updates take place. This argument can be
  an :class:`int`, let us say ``n``, at which point the agent will update its parameters
  every ``n`` time steps (if time steps are the default update frequency for that
  class). Otherwise, for further customization, an instance of
  :class:`UpdateStrategy` should be provided, allowing to specify both the frequency and
  the hooking for the updates. For more informations, see :ref:`user_guide_updating`.

- ``experience``: this argument represents the experience replay buffer, used to store
  interactions (but really, anything the user wishes to store) generated via
  interactions with the environment. Again, an :class:`int` can be passed, which will
  prompt the creation of a buffer with size ``n`` which, when sampled, will return all
  its ``n`` items. ``None`` can also be passed, in which case a unitary length buffer is
  created, i.e., the agent does not store any experience aside from the very last one.
  Alternatively, an instance of :class:`ExperienceReplay` can be passed to fine-tune the
  buffer size and sampling strategy. See :ref:`user_guide_experience` for more details.

:class:`LearningAgent` provides two additional methods:

- Of lesser notice, :meth:`LearningAgent.store_experience` can be used to store any
  experience item to the experience buffer, without requiring the user to do so
  manually.

- More importantly, :meth:`LearningAgent.train` is the method that must be called to
  initiate training of the agent. It takes an environment as input and runs the training
  loop, which consists of calling the abstract method
  :meth:`LearningAgent.train_one_episode` for the specified number of episodes.
  :meth:`LearningAgent.train` accepts various other arguments, most notably,
  ``behaviour_policy``, which allows to provide a different policy for the agent to
  learn from. This is the case for off-policy RL algorithms such as Q-learning.

This class also introduces the following abstract methods that must be implemented by
subclasses:

- :meth:`LearningAgent.train_one_episode`: this method is called by
  :meth:`LearningAgent.train` to train the agent for one episode. It must be implemented
  by subclasses, and it is where the actual learning takes place. The implementation
  obviously differs from algorithm to algorithm, but it must take care of calling the
  hooks provided by the mixin class :class:`core.callbacks.AgentCallbackMixin`
  (:meth:`core.callbacks.AgentCallbackMixin.on_mpc_failure`,
  :meth:`core.callbacks.AgentCallbackMixin.on_env_step`,
  :meth:`core.callbacks.AgentCallbackMixin.on_timestep_end`) so that the other
  components are triggered correctly.

- :meth:`LearningAgent.update`: this method is called by the update strategy to update
  the agent's parameters. It must be implemented by subclasses, and it is where the
  actual learning takes place. It assumes that :meth:`LearningAgent.train_one_episode`
  is implemented correctly, triggering the callbacks as needed, and thus triggering also
  the updates. In case the update fails, it must return an error message to be raised
  as an exception or warning, and it must return ``None`` otherwise.



:class:`RlLearningAgent` (gradient-based)
-----------------------------------------

TODO



:class:`GlobOptLearningAgent` (gradient-free)
---------------------------------------------

TODO
