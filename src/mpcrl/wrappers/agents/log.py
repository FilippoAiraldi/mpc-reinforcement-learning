import logging
from itertools import chain
from typing import Dict, Iterable, Iterator, Optional, Set, TypeVar

import numpy as np
import numpy.typing as npt
from gymnasium import Env

from mpcrl.agents.agent import Agent
from mpcrl.agents.learning_agent import LearningAgent
from mpcrl.core.callbacks import _LEARNING_AGENT_CALLBACKS
from mpcrl.util.iters import bool_cycle
from mpcrl.wrappers.agents.wrapper import SymType, Wrapper

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
_MANDATORY_CALLBACKS = {
    "on_mpc_failure",
    "on_validation_start",
    "on_validation_end",
    "on_update_failure",
    "on_training_start",
    "on_training_end",
}


class Log(Wrapper[SymType]):
    """A wrapper class for logging information about an agent."""

    __slots__ = ("logger", "precision", "log_frequencies", "exclude_mandatory")

    def __init__(
        self,
        agent: Agent[SymType],
        log_name: Optional[str] = None,
        level: int = logging.INFO,
        to_file: bool = False,
        mode: str = "a",
        precision: int = 3,
        log_frequencies: Optional[Dict[str, int]] = None,
        exclude_mandatory: Optional[Iterable[str]] = None,
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
        exclude_mandatory : iterable of str, optional
            An iterable of strings that contains the mandatory callbacks to be excluded.
            The mandatory callbacks that can be excluded are:
             - `on_mpc_failure`
             - `on_validation_start`
             - `on_validation_end`
             - `on_update_failure`
             - `on_training_start`
             - `on_training_end`.
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

        # store excluded-mandatory-callbacks and callbacks-with-frequency
        self.exclude_mandatory: Set[str] = (
            set() if exclude_mandatory is None else set(exclude_mandatory)
        )
        self.log_frequencies: Dict[str, Iterator[bool]] = {}
        if log_frequencies is not None:
            for name, freq in log_frequencies.items():
                if name not in _MANDATORY_CALLBACKS:
                    self.log_frequencies[name] = bool_cycle(freq)

        # if the agent is non-learning, make sure that both mandatory and callbacks with
        # frequencies do not lead to callbacks reserved to only learning agents.
        if not isinstance(agent, LearningAgent):
            for cb in _LEARNING_AGENT_CALLBACKS:
                self.exclude_mandatory.add(cb)
                self.log_frequencies.pop(cb, None)
        super().__init__(agent)

    def establish_callback_hooks(self) -> None:
        super().establish_callback_hooks()
        # hook only the callbacks for which a frequency was given + the mandatory ones
        repr_self = repr(self)
        optional = self.log_frequencies.keys()
        mandatory = _MANDATORY_CALLBACKS.difference(self.exclude_mandatory)
        for name in chain(optional, mandatory):
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
        self, env: Env[ObsType, ActType], returns: npt.NDArray[np.floating]
    ) -> None:
        S = np.array2string(returns, precision=self.precision)
        self.logger.info(f"validation of {env} concluded with returns={S}.")

    def _on_episode_start(self, env: Env[ObsType, ActType], episode: int) -> None:
        if next(self.log_frequencies["on_episode_start"]):
            self.logger.debug(f"episode {episode} started.")

    def _on_episode_end(
        self, env: Env[ObsType, ActType], episode: int, rewards: float
    ) -> None:
        if next(self.log_frequencies["on_episode_end"]):
            self.logger.info(
                f"episode {episode} ended with rewards={rewards:.{self.precision}f}."
            )

    def _on_env_step(
        self, env: Env[ObsType, ActType], episode: int, timestep: int
    ) -> None:
        if next(self.log_frequencies["on_env_step"]):
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
        self, env: Env[ObsType, ActType], returns: npt.NDArray[np.floating]
    ) -> None:
        S = np.array2string(returns, precision=self.precision)
        self.logger.info(f"training of {env} concluded with returns={S}.")

    def _on_update(self) -> None:
        if next(self.log_frequencies["on_update"]):
            S = self.agent.learnable_parameters.stringify()
            self.logger.info(f"updated parameters: {S}.")
