from typing import Dict, List

import numpy as np
import numpy.typing as npt

from mpcrl.agents.learning_agents import ExpType, LearningAgent
from mpcrl.wrappers.wrapper import LearningWrapper, SymType


class RecordUpdates(LearningWrapper[SymType, ExpType]):
    """Wrapper for recording the history of updates by the learning agent."""

    __slots__ = ("updates_history",)

    def __init__(self, agent: LearningAgent[SymType, ExpType]) -> None:
        """Instantiates the recorder.

        Parameters
        ----------
        agent : LearningAgent or inheriting
            The agent whose updates need recording.
        """
        super().__init__(agent)
        self.updates_history: Dict[str, List[npt.NDArray[np.double]]] = {
            p.name: [p.value] for p in agent.learnable_parameters.values()
        }
        self.unwrapped._hook_callbacks("on_update", self._store_learnable_parameters)

    def _store_learnable_parameters(self) -> None:
        """Internal utility to store the current parameters in memory."""
        for par in self.agent.learnable_parameters.values():
            self.updates_history[par.name].append(par.value)
