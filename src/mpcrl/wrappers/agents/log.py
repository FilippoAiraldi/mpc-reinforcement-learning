import logging
from collections.abc import Iterable, Iterator
from inspect import getmembers, isfunction
from itertools import chain
from operator import itemgetter
from typing import Callable, Optional, TypeVar

import numpy as np
import numpy.typing as npt
from gymnasium import Env

from ...agents.common.agent import Agent
from ...agents.common.learning_agent import LearningAgent
from ...core.callbacks import (
    AgentCallbackMixin,
    LearningAgentCallbackMixin,
    _failure_msg,
)
from ...util.iters import bool_cycle
from .wrapper import SymType, Wrapper

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

_pred = lambda o: isfunction(o) and o.__name__.startswith("on_")
_AGENT_CALLBACKS = set(map(itemgetter(0), getmembers(AgentCallbackMixin, _pred)))
_LEARNING_AGENT_CALLBACKS = set.difference(
    set(map(itemgetter(0), getmembers(LearningAgentCallbackMixin, _pred))),
    _AGENT_CALLBACKS,
)
del _pred


def _generate_method_caller(m: Callable) -> Callable:
    """Returns a method that calls the given method `m`."""

    def method_caller(*args, **kwargs):
        return m(*args, **kwargs)

    return method_caller


class Log(Wrapper[SymType]):
    """A wrapper class for logging information about an agent.

    Parameters
    ----------
    agent : LearningAgent or inheriting
        Agent to wrap.
    log_name : str, optional
        Name of the logger. If not provided, the name of the agent is used.
    level : int, optional
        The logging level, by default :attr:`logging.INFO`.
    to_file : bool, optional
        Whether to write the log also to a file in the current directory. By
        default, ``False``.
    mode : str, optional
        The mode for opening the logging faile, in case ``to_file=True``. By default, it
        appends to the file, if already present.
    precision : int, optional
        Precision for printing floats, by default ``3``.
    log_frequencies : dict of (str, int), optional
        A dict containing, for each logging call hook, its corresponding frequency. The
        calls for which a frequency can be set are:

        - ``"on_episode_start"``
        - ``"on_episode_end"``
        - ``"on_env_step"``
        - ``"on_timestep_end"``
        - ``"on_update"``.

        If this dictionary does not contain an entry for a specific call, the call is
        assumed to be never logged.
    exclude_mandatory : iterable of str, optional
        An iterable of strings that contains the default mandatory callbacks to be
        excluded. These mandatory callbacks that can be excluded are:

        - ``"on_mpc_failure"``
        - ``"on_validation_start"``
        - ``"on_validation_end"``
        - ``"on_update_failure"``
        - ``"on_training_start"``
        - ``"on_training_end"``.
    """

    def __init__(
        self,
        agent: Agent[SymType],
        log_name: Optional[str] = None,
        level: int = logging.INFO,
        to_file: bool = False,
        mode: str = "a",
        precision: int = 3,
        log_frequencies: Optional[dict[str, int]] = None,
        exclude_mandatory: Optional[Iterable[str]] = None,
    ) -> None:
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
        self.exclude_mandatory: set[str] = (
            set() if exclude_mandatory is None else set(exclude_mandatory)
        )
        self.log_frequencies: dict[str, Iterator[bool]] = {}
        if log_frequencies is not None:
            for name, freq in log_frequencies.items():
                if name not in _MANDATORY_CALLBACKS:
                    self.log_frequencies[name] = bool_cycle(freq)

        # if the agent is non-learning, make sure that both mandatory and callbacks with
        # frequencies do not lead to callbacks reserved to only learning agents.
        if not isinstance(agent.unwrapped, LearningAgent):
            for cb in _LEARNING_AGENT_CALLBACKS:
                self.exclude_mandatory.add(cb)
                self.log_frequencies.pop(cb, None)
        super().__init__(agent)

    def _establish_callback_hooks(self) -> None:
        super()._establish_callback_hooks()
        # hook only the callbacks for which a frequency was given + the mandatory ones
        repr_self = repr(self)
        optional_cbs = self.log_frequencies.keys()
        mandatory_cbs = _MANDATORY_CALLBACKS.difference(self.exclude_mandatory)
        for name in chain(optional_cbs, mandatory_cbs):
            method = getattr(self, f"_{name}")
            self._hook_callback(repr_self, name, _generate_method_caller(method))

    # NOTE: callbacks for Agent

    def _on_mpc_failure(
        self, episode: int, timestep: Optional[int], status: str, raises: bool
    ) -> None:
        m = self.logger.error if raises else self.logger.warning
        m(_failure_msg("mpc", self.agent.name, episode, timestep, status))

    def _on_validation_start(self, env: Env[ObsType, ActType]) -> None:
        self.logger.debug("validation of %s started.", env)

    def _on_validation_end(
        self, env: Env[ObsType, ActType], returns: npt.NDArray[np.floating]
    ) -> None:
        S = np.array2string(returns, precision=self.precision)
        self.logger.info("validation of %s concluded with returns=%s.", env, S)

    def _on_episode_start(
        self, env: Env[ObsType, ActType], episode: int, state: ObsType
    ) -> None:
        if next(self.log_frequencies["on_episode_start"]):
            S = np.array2string(state, precision=self.precision)
            self.logger.debug("episode %d started with state=%s.", episode, S)

    def _on_episode_end(
        self, env: Env[ObsType, ActType], episode: int, rewards: float
    ) -> None:
        if next(self.log_frequencies["on_episode_end"]):
            self.logger.info(
                "episode %d ended with rewards=%.*f.", episode, self.precision, rewards
            )

    def _on_env_step(
        self, env: Env[ObsType, ActType], episode: int, timestep: int
    ) -> None:
        if next(self.log_frequencies["on_env_step"]):
            self.logger.debug(
                "env stepped in episode %d at time %d.", episode, timestep
            )

    def _on_timestep_end(
        self, env: Env[ObsType, ActType], episode: int, timestep: int
    ) -> None:
        if next(self.log_frequencies["on_timestep_end"]):
            self.logger.debug("episode %d stepped at time %d.", episode, timestep)

    # NOTE: callbacks for LearningAgent

    def _on_update_failure(
        self, episode: int, timestep: Optional[int], errormsg: str, raises: bool
    ) -> None:
        (self.logger.error if raises else self.logger.warning)(
            "_failure_msg('update', %s, %d, %s, %s)",
            self.agent.name,
            episode,
            timestep,
            errormsg,
        )

    def _on_training_start(self, env: Env[ObsType, ActType]) -> None:
        self.logger.debug("training of %s started.", env)

    def _on_training_end(
        self, env: Env[ObsType, ActType], returns: npt.NDArray[np.floating]
    ) -> None:
        S = np.array2string(returns, precision=self.precision)
        self.logger.info("training of %s concluded with returns=%s.", env, S)

    def _on_update(self) -> None:
        if next(self.log_frequencies["on_update"]):
            S = self.agent.learnable_parameters.stringify()
            self.logger.info("updated parameters: %s.", S)
