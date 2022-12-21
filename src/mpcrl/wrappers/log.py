import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

from mpcrl.agents.agent import Agent
from mpcrl.util.types import ActType, GymEnvLike, ObsType
from mpcrl.wrappers.wrapper import SymType, Wrapper


class Log(Wrapper[SymType]):
    """A wrapper class for logging information about an agent."""

    __slots__ = ("logger", "precision")

    def __init__(
        self,
        agent: Agent[SymType],
        log_name: Optional[str] = None,
        level: int = logging.INFO,
        to_file: bool = False,
        mode: str = "a",
        precision: int = 3,
    ) -> None:
        """Creates a logger wrapper.

        Parameters
        ----------
        agent : Agent or inheriting
            Agent to wrap.
        log_name : str, optional
            Name of the logger. If not provided, the name of the agent is used.
        level : int, optional
            The logging level, by default `DEBUG`.
        to_file : bool, optional
            Whether to write the log also to a file in the current directory. By
            default, `False`.
        mode : str, optional
            The mode for opening the logging faile, in case `to_file=True`.
        precision : int, optional
            Precision for printing floats, by default 3.

        Returns
        -------
        logging.Logger
            The newly created logger.
        """
        super().__init__(agent)
        name = log_name if log_name is not None else agent.name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(name)s@%(asctime)s> %(message)s", datefmt="%Y-%m-%d,%H:%M:%S"
        )
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        if to_file:
            fh = logging.FileHandler(f"{name}.txt", mode=mode)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        self.precision = precision
        self.logger.info("logger created.")

    # callbacks for Agent

    def on_mpc_failure(
        self, episode: int, timestep: int, status: str, raises: bool
    ) -> None:
        """See `agent.on_mpc_failure`."""
        m = self.logger.error if raises else self.logger.warning
        m(f"MPC failure at episode {episode}, time {timestep}, status: {status}.")
        self.agent.on_mpc_failure(episode, timestep, status, raises)

    def on_validation_start(self, env: GymEnvLike[ObsType, ActType]) -> None:
        """See `agent.on_validation_start`."""
        self.logger.debug(f"validation on {env.__class__.__name__} started.")
        self.agent.on_validation_start(env)

    def on_validation_end(
        self, env: GymEnvLike[ObsType, ActType], returns: npt.NDArray[np.double]
    ) -> None:
        """See `agent.on_validation_end`."""
        self.logger.info(
            f"validation on {env.__class__.__name__} concluded with "
            f"returns={np.array2string(returns, precision=self.precision)}."
        )
        self.agent.on_validation_end(env, returns)

    def on_episode_start(self, env: GymEnvLike[ObsType, ActType], episode: int) -> None:
        """See `agent.on_episode_start`."""
        self.logger.debug(f"episode {episode} started.")
        self.agent.on_episode_start(env, episode)

    def on_episode_end(
        self, env: GymEnvLike[ObsType, ActType], episode: int, rewards: float
    ) -> None:
        """See `agent.on_episode_end`."""
        self.logger.info(
            f"episode {episode} ended with rewards={rewards:.{self.precision}f}."
        )
        self.on_episode_end(env, episode, rewards)

    def on_env_step(
        self, env: GymEnvLike[ObsType, ActType], episode: int, timestep: int
    ) -> None:
        """See `agent.on_env_step`."""
        self.logger.debug(f"episode {episode} stepped at time {timestep}.")
        self.on_env_step(env, episode, timestep)

    # callbacks for LearningAgent

    def on_update_failure(
        self, episode: int, timestep: int, errormsg: str, raises: bool
    ) -> None:
        """See `learningagent.on_update_failure`."""
        m = self.logger.error if raises else self.logger.warning
        m(f"Update failed at episode {episode}, time {timestep}, status: {errormsg}.")
        self.agent.on_update_failure(episode, timestep, errormsg, raises)

    def on_training_start(self, env: GymEnvLike[ObsType, ActType]) -> None:
        """See `learningagent.on_training_start`."""
        self.logger.debug(f"training on {env.__class__.__name__} started.")
        self.agent.on_training_start(env)

    def on_training_end(
        self, env: GymEnvLike[ObsType, ActType], returns: npt.NDArray[np.double]
    ) -> None:
        """See `learningagent.on_training_end`."""
        self.logger.info(
            f"training on {env.__class__.__name__} concluded with "
            f"returns={np.array2string(returns, precision=self.precision)}."
        )
        self.agent.on_training_end(env, returns)

    def on_update(self) -> None:
        """See `learningagent.on_update`."""
        # assert isinstance(self.agent, LearningAgent)
        parsstr = self.agent.learnable_parameters.stringify()
        self.logger.info(f"updated parameters: {parsstr}.")
        self.agent.on_update()
