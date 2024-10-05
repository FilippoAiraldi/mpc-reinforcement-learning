-----
Agent
-----

.. currentmodule:: mpcrl

As aforementioned, the basic agent is implemented in :class:`Agent`. This class takes in
an MPC controller, but has no feature for learning from the environment. In fact, it can
evaluate the controller's performance in the environment via the :meth:`Agent.evaluate`,
but it does not accept a :class:`core.parameters.LearnableParametersDict` instance at
instantiation, because it cannot learn. However, :meth:`Agent.__init__` does accept a
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
