"""As it will be clear from the inheritance diagram in :ref:`module_reference_agents`,
all agents are derived from mixin classes that define callbacks and manage hooks
attached to these callbacks. These system allows not only the user to customize the
behaviour of a derived agent every time a callback is triggered, but also to easily
implement and manage all those events and quantities that need to be scheduled during
training and evaluation. Some examples of such events are the decay of the learning rate
or the exploration chances, or when and with which frequency to invoke an update of the
MPC parametrization. Here we list the classes that enable this system, but for an
introduction to the callbacks and how to use them, see :ref:`user_guide_callbacks`."""

from typing import Any, Callable, Literal, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt
from gymnasium import Env

from .errors import raise_or_warn_on_mpc_failure, raise_or_warn_on_update_failure

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def _failure_msg(
    category: Literal["mpc", "update"],
    name: str,
    episode: int,
    timestep: Optional[int],
    status: str,
) -> str:
    """Internal utility for composing message for mpc/update failure."""
    C = category.title()
    if timestep is None:
        return f"{C} failure of {name} at episode {episode}, status: {status}."
    else:
        return (
            f"{C} failure of {name} at episode {episode}, time {timestep}, "
            f"status: {status}."
        )


class CallbackMixin:
    """A class with the particular purpose of creating, storing and deleting hooks
    attached to callbacks.

    Notes
    -----
    A particular note must be included about the `__setstate__` method. When this method
    is used (e.g., via :func:`copy.deepcopy`), the hooks are not copied from the old
    copy. The reason is that the old copy/state's hooks are likely to be pointing to
    methods belonging to old objects' instances. Of course, this is an issue, because if
    the old hooks are used, the new object (created from the state) would reference
    callbacks belonging to the old object. For this reasons, hooks are not copied;
    instead, the method :meth:`_establish_callback_hooks` is automatically called to
    re-establish these, but with respect to the new object(s).
    """

    def __init__(self) -> None:
        self._hooks: dict[str, dict[str, Callable[..., None]]] = {}

    def __setstate__(
        self,
        state: Union[
            None, dict[str, Any], tuple[Optional[dict[str, Any]], dict[str, Any]]
        ],
    ) -> None:
        if isinstance(state, tuple) and len(state) == 2:
            state, slotstate = state
        else:
            slotstate = None
        if state is not None:
            # remove hooks (otherwise, new copies will still be calling the old object)
            state["_hooks"] = {}
            self.__dict__.update(state)
        if slotstate is not None:
            for key, value in slotstate.items():
                setattr(self, key, value)
        # re-establish hooks
        self._establish_callback_hooks()

    def _run_hooks(self, method_name: str, *args: Any) -> None:
        """Runs the internal hooks attached to the given method."""
        if hooks := self._hooks.get(method_name):
            for hook in hooks.values():
                hook(*args)

    def _establish_callback_hooks(self) -> None:
        """This method must be used to perform the connections between callbacks and any
        invokable method (hook). If the object has no hooks, then this method does
        nothing."""

    def _hook_callback(
        self, attachername: str, callbackname: str, func: Callable[..., None]
    ) -> None:
        """Hooks a function to be called each time a callback is invoked.

        Parameters
        ----------
        attachername : str
            The name of the object requesting the hook. Has only info purposes.
        callbackname : str
            Name of the callback to hook to, i.e., the target of the hooking.
        func : Callable
            function to be called when the callback is invoked. Must accept the same
            input arguments as the callback it is hooked to. The return value is
            discarded.

        Raises
        ------
        ValueError
            If an hook with name ``attachername`` is already attached to this callback.
        """
        hook_dict = self._hooks.setdefault(callbackname, {})
        if attachername in hook_dict:
            raise ValueError(
                f"Hook '{attachername}' already attached to callback '{callbackname}'."
            )
        hook_dict[attachername] = func


class AgentCallbackMixin(CallbackMixin):
    """Class with callbacks for agents.

    In particular, this class defines the following callbacks:

    - :meth:`on_mpc_failure`, invoked when an MPC solver fails
    - :meth:`on_validation_start`, invoked when validation starts (see
      :meth:`mpcrl.Agent.evaluate`)
    - :meth:`on_validation_end`, invoked when validation ends
    - :meth:`on_episode_start`, invoked when a training or validation episode starts
    - :meth:`on_episode_end`, invoked when a training or validation episode ends
    - :meth:`on_env_step`, invoked when a training or validation episode steps, i.e.,
      after :func:`gymnasium.Env.step`
    - :meth:`on_timestep_end`, invoked when the current simulation's time step reaches
      an end, i.e., after having stepped the environment and done all the internal
      computations according to the algorithm.
    """

    def on_mpc_failure(
        self, episode: int, timestep: Optional[int], status: str, raises: bool
    ) -> None:
        """Callback in case of failure of the MPC solver.

        Parameters
        ----------
        episode : int
            Number of the episode when the failure happened.
        timestep : int or None
            Timestep of the current episode when the failure happened. Can be ``None``,
            in case the error occurs inter-episodically or no notion of time step is
            available.
        status : str
            Status of the solver that failed.
        raises : bool
            Whether the failure should be raised as exception (``True``) or as a warning
            (``False``).
        """
        name: str = getattr(self, "name", "agent")
        raise_or_warn_on_mpc_failure(
            _failure_msg("mpc", name, episode, timestep, status),
            raises,
        )
        self._run_hooks("on_mpc_failure", episode, timestep, status, raises)

    def on_validation_start(self, env: Env[ObsType, ActType]) -> None:
        """Callback called at the beginning of the validation process (see
        :meth:`mpcrl.Agent.evaluate`)

        Parameters
        ----------
        env : gym env
            A gym environment where the agent is being validated on.
        """
        self._run_hooks("on_validation_start", env)

    def on_validation_end(
        self, env: Env[ObsType, ActType], returns: npt.NDArray[np.floating]
    ) -> None:
        """Callback called at the end of the validation process (see
        :meth:`mpcrl.Agent.evaluate`).

        Parameters
        ----------
        env : gym env
            A gym environment where the agent has been validated on.
        returns : array of double
            Each episode's cumulative rewards.
        """
        self._run_hooks("on_validation_end", env, returns)

    def on_episode_start(
        self, env: Env[ObsType, ActType], episode: int, state: ObsType
    ) -> None:
        """Callback called at the beginning of each episode in the training or
        validation process (see :meth:`mpcrl.Agent.evaluate`,
        :meth:`mpcrl.LearningAgent.train` and
        :meth:`mpcrl.LearningAgent.train_offpolicy`).

        Parameters
        ----------
        env : gym env
            A gym environment where the agent is being trained on.
        episode : int
            Number of the training episode.
        state : ObsType
            Starting state for this episode.
        """
        self._run_hooks("on_episode_start", env, episode, state)

    def on_episode_end(
        self, env: Env[ObsType, ActType], episode: int, rewards: float
    ) -> None:
        """Callback called at the end of each episode in the training or evaluation
        process (see :meth:`mpcrl.Agent.evaluate`, :meth:`mpcrl.LearningAgent.train` and
        :meth:`mpcrl.LearningAgent.train_offpolicy`).

        Parameters
        ----------
        env : gym env
            A gym environment where the agent is being trained on.
        episode : int
            Number of the training episode.
        rewards : float
            Cumulative rewards for this episode.
        """
        self._run_hooks("on_episode_end", env, episode, rewards)

    def on_env_step(
        self, env: Env[ObsType, ActType], episode: int, timestep: int
    ) -> None:
        """Callback called after each call to :func:`gymnasium.Env.step`.

        Parameters
        ----------
        env : gym env
            A gym environment where the agent is being trained on.
        episode : int
            Number of the training episode.
        timestep : int
            Time instant of the current training episode.
        """
        self._run_hooks("on_env_step", env, episode, timestep)

    def on_timestep_end(
        self, env: Env[ObsType, ActType], episode: int, timestep: int
    ) -> None:
        """Callback called at the end of each time iteration. It is called with the same
        frequency as :meth:`on_env_step`, but with different timing.

        Parameters
        ----------
        env : gym env
            A gym environment where the agent is being trained on.
        episode : int
            Number of the training episode.
        timestep : int
            Time instant of the current training episode.
        """
        self._run_hooks("on_timestep_end", env, episode, timestep)


class LearningAgentCallbackMixin(AgentCallbackMixin):
    """Class with callbacks for learning agents.

    In particular, this class defines, on top of the callbacks from
    :class:`AgentCallbackMixin`, the additional following callbacks:

    - :meth:`on_update_failure`, invoked when an update of the parametrization fails
    - :meth:`on_training_start`, invoked when training starts (see
      :meth:`mpcrl.LearningAgent.train` and :meth:`mpcrl.LearningAgent.train_offpolicy`)
    - :meth:`on_training_end`, invoked when training ends
    - :meth:`on_update`, invoked after each update of the parametrization.
    """

    def on_update_failure(
        self, episode: int, timestep: Optional[int], errormsg: str, raises: bool
    ) -> None:
        """Callback in case of update failure.

        Parameters
        ----------
        episode : int
            Number of the episode when the failure happened.
        timestep : int or None
            Timestep of the current episode when the failure happened. Can be ``None``
            in case the update occurs inter-episodically or no notion of time step is
            available.
        errormsg : str
            Error message of the update failure.
        raises : bool
            Whether the failure should be raised as exception (``True``) or as a warning
            (``False``).
        """
        name: str = getattr(self, "name", "agent")
        raise_or_warn_on_update_failure(
            _failure_msg("update", name, episode, timestep, errormsg),
            raises,
        )
        self._run_hooks("on_update_failure", episode, timestep, errormsg, raises)

    def on_training_start(self, env: Env[ObsType, ActType]) -> None:
        """Callback called at the beginning of the training process.

        Parameters
        ----------
        env : gym env
            A gym environment where the agent is being trained on.
        """
        self._run_hooks("on_training_start", env)

    def on_training_end(
        self, env: Env[ObsType, ActType], returns: npt.NDArray[np.floating]
    ) -> None:
        """Callback called at the end of the training process.

        Parameters
        ----------
        env : gym env
            A gym environment where the agent has been trained on.
        returns : array of double
            Each episode's cumulative rewards.
        """
        self._run_hooks("on_training_end", env, returns)

    def on_update(self) -> None:
        """Callback called after each :func:`mpcrl.LearningAgent.update`.

        This callback is especially useful for, e.g., decaying exploration probabilities
        or learning rates."""
        self._run_hooks("on_update")
