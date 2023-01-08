from inspect import getmembers, isfunction
from itertools import chain
from operator import itemgetter
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from mpcrl.core.errors import (
    raise_or_warn_on_mpc_failure,
    raise_or_warn_on_update_failure,
)
from mpcrl.util.types import ActType, GymEnvLike, ObsType


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
        if timestep is None:
            msg = f"Mpc failure at episode {episode}, status: {status}."
        else:
            msg = (
                f"Mpc failure at episode {episode}, time {timestep}, "
                f"status: {status}."
            )
        raise_or_warn_on_mpc_failure(msg, raises)

    def on_validation_start(self, env: GymEnvLike[ObsType, ActType]) -> None:
        """Callback called at the beginning of the validation process.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent is being validated on.
        """

    def on_validation_end(
        self, env: GymEnvLike[ObsType, ActType], returns: npt.NDArray[np.double]
    ) -> None:
        """Callback called at the end of the validation process.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent has been validated on.
        returns : array of double
            Each episode's cumulative rewards.
        """

    def on_episode_start(self, env: GymEnvLike[ObsType, ActType], episode: int) -> None:
        """Callback called at the beginning of each episode in the training process.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent is being trained on.
        episode : int
            Number of the training episode.
        """

    def on_episode_end(
        self, env: GymEnvLike[ObsType, ActType], episode: int, rewards: float
    ) -> None:
        """Callback called at the end of each episode in the training process.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent is being trained on.
        episode : int
            Number of the training episode.
        rewards : float
            Cumulative rewards for this episode.
        """

    def on_env_step(
        self, env: GymEnvLike[ObsType, ActType], episode: int, timestep: int
    ) -> None:
        """Callback called after each `env.step`.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent is being trained on.
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
        if timestep is None:
            msg = f"Update failed at episode {episode}, status: {errormsg}."
        else:
            msg = (
                f"Update failed at episode {episode}, time {timestep}, "
                f"status: {errormsg}."
            )
        raise_or_warn_on_update_failure(msg, raises)

    def on_training_start(self, env: GymEnvLike[ObsType, ActType]) -> None:
        """Callback called at the beginning of the training process.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent is being trained on.
        """

    def on_training_end(
        self, env: GymEnvLike[ObsType, ActType], returns: npt.NDArray[np.double]
    ) -> None:
        """Callback called at the end of the training process.

        Parameters
        ----------
        env : gym env
            A gym-like environment where the agent has been trained on.
        returns : array of double
            Each episode's cumulative rewards.
        """

    def on_update(self) -> None:
        """Callaback called after each `agent.update`. Use this callback for, e.g.,
        decaying exploration probabilities or learning rates."""


ALL_CALLBACKS = tuple(
    map(
        itemgetter(0),
        chain.from_iterable(
            (
                getmembers(
                    cls,
                    predicate=lambda o: isfunction(o) and o.__name__.startswith("on_"),
                )
                for cls in (AgentCallbacks, LearningAgentCallbacks)
            )
        ),
    )
)


class RemovesCallbackHooksInState:
    """A class with the particular purpose of removing hooks when setting the state
    and re-establishing them automatically. In fact, if the old hooks are used, the new
    object (created from the state) would reference callbacks belonging to the old
    agent. In this way, the callbacks are linked to the new instance.
    """

    def establish_callback_hooks(self) -> None:
        """This method must be used to perform the connections between callbacks and any
        invokable method (hook). If the object has no hooks, then this method does
        nothing."""

    def __setstate__(
        self,
        state: Union[
            None, Dict[str, Any], Tuple[Optional[Dict[str, Any]], Dict[str, Any]]
        ],
    ) -> None:
        if isinstance(state, tuple) and len(state) == 2:
            state, slotstate = state
        else:
            slotstate = None
        if state is not None:
            # remove wrapped methods for callbacks due to this object (otherwise, new
            # copies will still be calling the old object).
            for name in ALL_CALLBACKS:
                state.pop(name, None)  # type: ignore[union-attr]
            self.__dict__.update(state)  # type: ignore[arg-type]
        if slotstate is not None:
            for key, value in slotstate.items():
                setattr(self, key, value)
        # re-perform hooks
        self.establish_callback_hooks()
