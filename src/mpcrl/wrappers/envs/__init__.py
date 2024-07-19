"""Wrappers to record information about the environment.

This submodule contains wrappers to record information about call arguments to and
return arguments from environment's :func:`gymnasium.Env.reset` and
:func:`gymnasium.Env.step`. In particular

- the :class:`MonitorInfos` wrapper records the information dictionaries returned by the
  environment

- the :class:`MonitorEpisodes` wrapper records the observations, actions, rewards,
  episode lengths, and execution times of each episode.
"""

__all__ = ["MonitorEpisodes", "MonitorInfos"]

from .monitor_episodes import MonitorEpisodes
from .monitor_infos import MonitorInfos
