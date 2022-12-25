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
        self, episode: int, timestep: int, status: str, raises: bool
    ) -> None:
        """Callback in case of MPC failure.

        Parameters
        ----------
        episode : int
            Number of the episode when the failure happened.
        timestep : int
            Timestep of the current episode when the failure happened.
        status : str
            Status of the solver that failed.
        raises : bool
            Whether the failure should be raised as exception.
        """
        raise_or_warn_on_mpc_failure(
            f"mpc failure at episode {episode}, time {timestep}, status: {status}.",
            raises,
        )

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
        self, episode: int, timestep: int, errormsg: str, raises: bool
    ) -> None:
        """Callback in case of update failure.

        Parameters
        ----------
        episode : int
            Number of the episode when the failure happened.
        timestep : int
            Timestep of the current episode when the failure happened.
        errormsg : str
            Error message of the update failure.
        raises : bool
            Whether the failure should be raised as exception.
        """
        raise_or_warn_on_update_failure(
            f"Update failed at episode {episode}, time {timestep}, status: {errormsg}.",
            raises,
        )

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
