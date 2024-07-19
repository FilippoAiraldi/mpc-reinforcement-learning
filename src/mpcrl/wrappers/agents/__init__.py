"""Wrappers for learning and non-learning agents.

First of all, this submodule introduces two base classes for wrappers to be appplied to
agents, not environments. These classes are :class:`mpcrl.wrappers.agents.Wrapper` and
:class:`mpcrl.wrappers.agents.LearningWrapper`. The former is a wrapper for non-learning
agents, while the latter is a wrapper for learning agents. Both classes define an
interface equal to the agent's, so that they can be used as agents themselves
seamlessly.

Outside of these base classes, this submodule provides a few concrete wrappers for
logging information about the agent's learning/evaluation process, recording the updates
of the MPC parametrization during learning, and launching periodic evaluations of the
agent during training.
"""

__all__ = ["Evaluate", "LearningWrapper", "Log", "RecordUpdates", "Wrapper"]

from .evaluate import Evaluate
from .log import Log
from .record_updates import RecordUpdates
from .wrapper import LearningWrapper, Wrapper
