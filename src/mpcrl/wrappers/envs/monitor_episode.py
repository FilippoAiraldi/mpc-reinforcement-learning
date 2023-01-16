from collections import deque
from time import perf_counter
from typing import (
    Any,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    SupportsFloat,
    Tuple,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
from gymnasium import Env, Wrapper

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


def compact_dicts_into_one(
    dicts: Iterable[Dict[str, Any]], fill_value: Any = None
) -> Dict[str, List[Any]]:
    """Compacts an iterable of dictionaries into a single dict with lists of entries. If
    an entry is missing for any given dict, `fill_value` is used in place.

    Parameters
    ----------
    dicts : iterable of dicts[str, any]
        Dictionaries to be compacted into a single one.
    fill_value : any, optional
        The value to be used to fill in missing values.

    Returns
    -------
    Dict[str, list of any | fill_value]
        A unique dictionary made from all the passed dicts.
    """
    out: Dict[str, List[Any]] = {}
    for i, dict in enumerate(dicts):
        for k, v in dict.items():
            if k in out:
                out[k].append(v)
            else:
                out[k] = [fill_value] * i + [v]
        for k in out.keys() - dict.keys():
            out[k].append(fill_value)
    return out


class MonitorEpisode(Wrapper[ObsType, ActType]):
    """This wrapper keeps track of
        - observations
        - actions
        - costs/rewards
        - episode length
        - episode execution time
    that the environment is subject to during the learning process.
    """

    def __init__(
        self, env: Env[ObsType, ActType], deque_size: Optional[int] = None
    ) -> None:
        """This wrapper will keep track of observations, actions and rewards as well as
        episode length and execution time.

        Parameters
        ----------
        env : Env[ObsType, ActType]
            The environment to apply the wrapper to.
        deque_size : int, optional
            The maximum number of episodes to hold as historical data in the internal
            deques. By default, `None`, i.e., unlimited.
        """
        super().__init__(env)
        # long-term storages
        self.observations: Deque[npt.NDArray[ObsType]] = deque(  # type: ignore
            maxlen=deque_size
        )
        self.actions: Deque[npt.NDArray[ActType]] = deque(  # type: ignore
            maxlen=deque_size
        )
        self.rewards: Deque[npt.NDArray[np.floating]] = deque(maxlen=deque_size)
        self.infos: Deque[Dict[str, List[Any]]] = deque(maxlen=deque_size)
        self.episode_lengths: Deque[int] = deque(maxlen=deque_size)
        self.exec_times: Deque[float] = deque(maxlen=deque_size)
        # current-episode-storages
        self.ep_observations: List[ObsType] = []
        self.ep_actions: List[ActType] = []
        self.ep_rewards: List[SupportsFloat] = []
        self.ep_infos: List[Dict[str, Any]] = []
        self.t0: float = perf_counter()
        self.ep_length: int = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Resets the environment and resets the current data accumulators."""
        observation, info = super().reset(seed=seed, options=options)
        self._clear_ep_data()
        self.ep_observations.append(observation)
        return observation, info

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        # sourcery skip: extract-method
        """Steps through the environment, accumulating the episode data."""
        obs, reward, terminated, truncated, info = super().step(action)

        # accumulate data
        self.ep_observations.append(obs)
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)
        self.ep_infos.append(info)
        self.ep_length += 1

        # if episode is done, save the current data to history
        if terminated or truncated:
            # append data
            self.observations.append(np.asarray(self.ep_observations))
            self.actions.append(np.asarray(self.ep_actions))
            self.rewards.append(np.asarray(self.ep_rewards))
            self.infos.append(compact_dicts_into_one(self.ep_infos))
            self.episode_lengths.append(self.ep_length)
            self.exec_times.append(perf_counter() - self.t0)

            # clear this episode's data
            self._clear_ep_data()

        return obs, reward, terminated, truncated, info

    def _clear_ep_data(self) -> None:
        # clear this episode's lists and reset counters
        self.ep_observations.clear()
        self.ep_actions.clear()
        self.ep_rewards.clear()
        self.ep_infos.clear()
        self.t0 = perf_counter()
        self.ep_length = 0
