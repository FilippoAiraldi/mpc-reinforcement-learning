from functools import wraps
from inspect import getmembers, isfunction
from operator import itemgetter
from typing import Any, Callable, Literal, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt
from gymnasium import Env

from mpcrl.core.errors import (
    raise_or_warn_on_mpc_failure,
    raise_or_warn_on_update_failure,
)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def _failure_msg(
    type: Literal["mpc", "update"],
    name: str,
    episode: int,
    timestep: Optional[int],
    status: str,
) -> str:
    """Internal utility for composing message for mpc/update failure."""
    p = type.title()
    if timestep is None:
        return f"{p} failure of {name} at episode {episode}, status: {status}."
    else:
        return (
            f"{p} failure of {name} at episode {episode}, time {timestep}, "
            f"status: {status}."
        )


class AgentCallbacks:
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

    def on_validation_start(self, env: Env[ObsType, ActType]) -> None:
        """Callback called at the beginning of the validation process.

        Parameters
        ----------
        env : gym env
            A gym environment where the agent is being validated on.
        """

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


class LearningAgentCallbacks:
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

    def on_training_start(self, env: Env[ObsType, ActType]) -> None:
        """Callback called at the beginning of the training process.

        Parameters
        ----------
        env : gym env
            A gym environment where the agent is being trained on.
        """

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

    def on_update(self) -> None:
        """Callaback called after each `agent.update`. Use this callback for, e.g.,
        decaying exploration probabilities or learning rates."""


_pred = lambda o: isfunction(o) and o.__name__.startswith("on_")
_AGENT_CALLBACKS, _LEARNING_AGENT_CALLBACKS = (
    set(map(itemgetter(0), getmembers(cls, _pred)))
    for cls in (AgentCallbacks, LearningAgentCallbacks)
)
_ALL_CALLBACKS = set.union(_AGENT_CALLBACKS, _LEARNING_AGENT_CALLBACKS)
del _pred


class RemovesCallbackHooksInState:
    """A class with the particular purpose of removing hooks when setting the state
    and re-establishing them automatically. In fact, if the old hooks are used, the new
    object (created from the state) would reference callbacks belonging to the old
    agent. In this way, the callbacks are linked to the new instance.
    """

    def hook_callback(
        self, attachername: str, callbackname: str, func: Callable
    ) -> None:
        """Hooks a function to be called each time an agent's callback is invoked.

        Parameters
        ----------
        attachername : str
            The name of the object requesting the hook. Has only info purposes.
        callbackname : str
            Name of the callback to hook to.
        func : Callable
            function to be called when the callback is invoked. Must accept the same
            input arguments as the callback it is hooked to. Moreover, the return value
            is discarded.
        """

        def decorate(method: Callable) -> Callable:
            @wraps(method)
            def wrapper(*args, **kwargs):
                out = method(*args, **kwargs)
                func(*args, **kwargs)
                return out

            wrapper.attacher = attachername  # type: ignore[attr-defined]
            return wrapper

        setattr(self, callbackname, decorate(getattr(self, callbackname)))

    def establish_callback_hooks(self) -> None:
        """This method must be used to perform the connections between callbacks and any
        invokable method (hook). If the object has no hooks, then this method does
        nothing."""

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
            # remove wrapped methods for callbacks due to this object (otherwise, new
            # copies will still be calling the old object).
            for name in _ALL_CALLBACKS:
                state.pop(name, None)  # type: ignore[union-attr]
            self.__dict__.update(state)  # type: ignore[arg-type]
        if slotstate is not None:
            for key, value in slotstate.items():
                setattr(self, key, value)
        # re-establish hooks
        self.establish_callback_hooks()
