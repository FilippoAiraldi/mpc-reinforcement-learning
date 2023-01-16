import logging
from itertools import chain, repeat
from typing import Dict, Iterator, Optional, TypeVar

import numpy as np
import numpy.typing as npt
from gymnasium import Env

from mpcrl.agents.learning_agent import ExpType, LearningAgent
from mpcrl.core.callbacks import ALL_CALLBACKS
from mpcrl.util.iters import bool_cycle
from mpcrl.wrappers.agents.wrapper import LearningWrapper, SymType

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
_FALSE_ITER = repeat(False)


class Log(LearningWrapper[SymType, ExpType]):
    """A wrapper class for logging information about an agent."""

    __slots__ = ("logger", "precision", "log_frequencies")

    def __init__(
        self,
        agent: LearningAgent[SymType, ExpType],
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
        agent : LearningAgent or inheriting
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
        """
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

        self.log_frequencies: Dict[str, Iterator[bool]] = {}
        if log_frequencies is not None:
            for name, freq in log_frequencies.items():
                self.log_frequencies[name] = bool_cycle(freq)

        super().__init__(agent)

    def establish_callback_hooks(self) -> None:
        super().establish_callback_hooks()
        # hook only the callbacks for which a frequency was given + the mandatory ones
        repr_self = repr(self)
        specified = self.log_frequencies.keys()
        mandatory = set(ALL_CALLBACKS).difference(specified)
        for name in chain(specified, mandatory):
            self.hook_callback(
                repr_self,
                name,
                getattr(self, f"_{name}"),
                args_idx=slice(None),
                kwargs_keys="all",
            )

    # callbacks for Agent

    def _on_mpc_failure(
        self, episode: int, timestep: Optional[int], status: str, raises: bool
    ) -> None:
        m = self.logger.error if raises else self.logger.warning
        m(f"Mpc failure at episode {episode}, time {timestep}, status: {status}.")

    def _on_validation_start(self, env: Env[ObsType, ActType]) -> None:
        self.logger.debug(f"validation of {env} started.")

    def _on_validation_end(
        self, env: Env[ObsType, ActType], returns: npt.NDArray[np.double]
    ) -> None:
        S = np.array2string(returns, precision=self.precision)
        self.logger.info(f"validation of {env} concluded with returns={S}.")

    def _on_episode_start(self, env: Env[ObsType, ActType], episode: int) -> None:
        if next(self.log_frequencies.get("on_episode_start", _FALSE_ITER)):
            self.logger.debug(f"episode {episode} started.")

    def _on_episode_end(
        self, env: Env[ObsType, ActType], episode: int, rewards: float
    ) -> None:
        if next(self.log_frequencies.get("on_episode_end", _FALSE_ITER)):
            self.logger.info(
                f"episode {episode} ended with rewards={rewards:.{self.precision}f}."
            )

    def _on_env_step(
        self, env: Env[ObsType, ActType], episode: int, timestep: int
    ) -> None:
        if next(self.log_frequencies.get("on_env_step", _FALSE_ITER)):
            self.logger.debug(f"episode {episode} stepped at time {timestep}.")

    # callbacks for LearningAgent

    def _on_update_failure(
        self, episode: int, timestep: Optional[int], errormsg: str, raises: bool
    ) -> None:
        m = self.logger.error if raises else self.logger.warning
        m(f"Update failed at episode {episode}, time {timestep}, status: {errormsg}.")

    def _on_training_start(self, env: Env[ObsType, ActType]) -> None:
        self.logger.debug(f"training of {env} started.")

    def _on_training_end(
        self, env: Env[ObsType, ActType], returns: npt.NDArray[np.double]
    ) -> None:
        S = np.array2string(returns, precision=self.precision)
        self.logger.info(f"training of {env} concluded with returns={S}.")

    def _on_update(self) -> None:
        if next(self.log_frequencies.get("on_update", _FALSE_ITER)):
            S = self.agent.learnable_parameters.stringify()
            self.logger.info(f"updated parameters: {S}.")
