import logging
from itertools import repeat
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt

from mpcrl.agents.agent import Agent
from mpcrl.util.iters import bool_cycle
from mpcrl.util.types import ActType, GymEnvLike, ObsType
from mpcrl.wrappers.wrapper import SymType, Wrapper

_FALSE_ITER = repeat(False)


class Log(Wrapper[SymType]):
    """A wrapper class for logging information about an agent."""

    __slots__ = ("logger", "precision", "log_frequencies")

    def __init__(
        self,
        agent: Agent[SymType],
        log_name: Optional[str] = None,
        level: int = logging.INFO,
        to_file: bool = False,
        mode: str = "a",
        precision: int = 3,
        log_frequencies: Optional[Dict[str, int]] = None,
    ) -> None:
        """Creates a logger wrapper.

        Parameters
        ----------
        agent : Agent or inheriting
            Agent to wrap.
        log_name : str, optional
            Name of the logger. If not provided, the name of the agent is used.
        level : int, optional
            The logging level, by default `INFO`.
        to_file : bool, optional
            Whether to write the log also to a file in the current directory. By
            default, `False`.
        mode : str, optional
            The mode for opening the logging faile, in case `to_file=True`.
        precision : int, optional
            Precision for printing floats, by default 3.
        log_frequencies : int, optional
            Dict containing for each logging call its corresponding frequency. The calls
            for which a frequency can be set are:
             - `on_episode_start`
             - `on_episode_end`
             - `on_env_step`
             - `on_update`.
            If an entry is not found in the dict, it is assumed that its call is never
            logged.

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
        if log_frequencies is None:
            self.log_frequencies = {}
        else:
            self.log_frequencies = {
                name: bool_cycle(freq) for name, freq in log_frequencies.items()
            }
        self.logger.info("logger created.")

    # callbacks for Agent

    def on_mpc_failure(
        self, episode: int, timestep: Optional[int], status: str, raises: bool
    ) -> None:
        """See `agent.on_mpc_failure`."""
        m = self.logger.error if raises else self.logger.warning
        m(f"Mpc failure at episode {episode}, time {timestep}, status: {status}.")
        self.agent.on_mpc_failure(episode, timestep, status, raises)

    def on_validation_start(self, env: GymEnvLike[ObsType, ActType]) -> None:
        """See `agent.on_validation_start`."""
        self.logger.debug(f"validation on {env} started.")
        self.agent.on_validation_start(env)

    def on_validation_end(
        self, env: GymEnvLike[ObsType, ActType], returns: npt.NDArray[np.double]
    ) -> None:
        """See `agent.on_validation_end`."""
        S = np.array2string(returns, precision=self.precision)
        self.logger.info(f"validation on {env} concluded with returns={S}.")
        self.agent.on_validation_end(env, returns)

    def on_episode_start(self, env: GymEnvLike[ObsType, ActType], episode: int) -> None:
        """See `agent.on_episode_start`."""
        if next(self.log_frequencies.get("on_episode_start", _FALSE_ITER)):
            self.logger.debug(f"episode {episode} started.")
        self.agent.on_episode_start(env, episode)

    def on_episode_end(
        self, env: GymEnvLike[ObsType, ActType], episode: int, rewards: float
    ) -> None:
        """See `agent.on_episode_end`."""
        if next(self.log_frequencies.get("on_episode_end", _FALSE_ITER)):
            self.logger.info(
                f"episode {episode} ended with rewards={rewards:.{self.precision}f}."
            )
        self.agent.on_episode_end(env, episode, rewards)

    def on_env_step(
        self, env: GymEnvLike[ObsType, ActType], episode: int, timestep: int
    ) -> None:
        """See `agent.on_env_step`."""
        if next(self.log_frequencies.get("on_env_step", _FALSE_ITER)):
            self.logger.debug(f"episode {episode} stepped at time {timestep}.")
        self.agent.on_env_step(env, episode, timestep)

    # callbacks for LearningAgent

    def on_update_failure(
        self, episode: int, timestep: Optional[int], errormsg: str, raises: bool
    ) -> None:
        """See `learningagent.on_update_failure`."""
        m = self.logger.error if raises else self.logger.warning
        m(f"Update failed at episode {episode}, time {timestep}, status: {errormsg}.")
        self.agent.on_update_failure(episode, timestep, errormsg, raises)

    def on_training_start(self, env: GymEnvLike[ObsType, ActType]) -> None:
        """See `learningagent.on_training_start`."""
        self.logger.debug(f"training on {env} started.")
        self.agent.on_training_start(env)

    def on_training_end(
        self, env: GymEnvLike[ObsType, ActType], returns: npt.NDArray[np.double]
    ) -> None:
        """See `learningagent.on_training_end`."""
        S = np.array2string(returns, precision=self.precision)
        self.logger.info(f"training on {env} concluded with returns={S}.")
        self.agent.on_training_end(env, returns)

    def on_update(self) -> None:
        """See `learningagent.on_update`."""
        # assert isinstance(self.agent, LearningAgent)
        if next(self.log_frequencies.get("on_update", _FALSE_ITER)):
            S = self.agent.learnable_parameters.stringify()
            self.logger.info(f"updated parameters: {S}.")
        self.agent.on_update()
