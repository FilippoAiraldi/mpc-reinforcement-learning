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
    """A class with the particular purpose of creating, storing and deleting hooks.
    Particularly touchy is when the state is set, and the hooks need to be reestablished
    automatically. In fact, if the old hooks are used, the new object (created from the
    state) would reference callbacks belonging to the old agent. In this way, the
    callbacks are linked to the new instance.
    """

    def __init__(self) -> None:
        self._hooks: dict[str, list[tuple[str, Callable[..., None]]]] = {}

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
        self.establish_callback_hooks()

    def _run_hooks(self, method_name: str, *args: Any) -> None:
        """Runs the internal hooks attached to the given method."""
        if hooks := self._hooks.get(method_name):
            for hook in hooks.values():
                hook(*args)

    def establish_callback_hooks(self) -> None:
        """This method must be used to perform the connections between callbacks and any
        invokable method (hook). If the object has no hooks, then this method does
        nothing."""

    def hook_callback(
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
            input arguments as the callback it is hooked to. Moreover, the return value
            is discarded.

        Raises
        ------
        ValueError
            If an hook with name `attachername` is already attached to this callback.
        """
        hook_dict = self._hooks.setdefault(callbackname, {})
        if attachername in hook_dict:
            raise ValueError(
                f"Hook '{attachername}' already attached to callback '{callbackname}'."
            )
        hook_dict[attachername] = func


class AgentCallbackMixin(CallbackMixin):
    """Callbacks for agents."""

    def on_mpc_failure(
        self, episode: int, timestep: Optional[int], status: str, raises: bool
    ) -> None:
        """Callback in case of MPC failure.

        Parameters
        ----------
        episode : int
            Number of the episode when the failure happened.
        timestep : int or None
            Timestep of the current episode when the failure happened. Can be `None` in
            case the error occurs inter-episodically.
        status : str
            Status of the solver that failed.
        raises : bool
            Whether the failure should be raised as exception.
        """
        name: str = getattr(self, "name", "agent")
        raise_or_warn_on_mpc_failure(
            _failure_msg("mpc", name, episode, timestep, status),
            raises,
        )
        self._run_hooks("on_mpc_failure", episode, timestep, status, raises)

    def on_validation_start(self, env: Env[ObsType, ActType]) -> None:
        """Callback called at the beginning of the validation process.

        Parameters
        ----------
        env : gym env
            A gym environment where the agent is being validated on.
        """
        self._run_hooks("on_validation_start", env)

    def on_validation_end(
        self, env: Env[ObsType, ActType], returns: npt.NDArray[np.floating]
    ) -> None:
        """Callback called at the end of the validation process.

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
        """Callback called at the beginning of each episode in the training process.

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
        """Callback called at the end of each episode in the training process.

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
        """Callback called after each `env.step`.

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
        frequency as `env.step`, but with different timing.

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
    """Callbacks for learning agents."""

    def on_update_failure(
        self, episode: int, timestep: Optional[int], errormsg: str, raises: bool
    ) -> None:
        """Callback in case of update failure.

        Parameters
        ----------
        episode : int
            Number of the episode when the failure happened.
        timestep : int or None
            Timestep of the current episode when the failure happened. Can be `None` in
            case the update occurs inter-episodically.
        errormsg : str
            Error message of the update failure.
        raises : bool
            Whether the failure should be raised as exception.
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
        """Callback called after each `agent.update`. Use this callback for, e.g.,
        decaying exploration probabilities or learning rates."""
        self._run_hooks("on_update")
