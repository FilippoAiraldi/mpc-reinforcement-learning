import numpy as np
import numpy.typing as npt

from ...agents.common.learning_agent import ExpType, LearningAgent
from .wrapper import LearningWrapper, SymType


class RecordUpdates(LearningWrapper[SymType, ExpType]):
    """Wrapper for recording the history of updates by the learning agent."""

    def __init__(self, agent: LearningAgent[SymType, ExpType]) -> None:
        """Instantiates the recorder.

        Parameters
        ----------
        agent : LearningAgent or inheriting
            The agent whose updates need recording.
        """
        super().__init__(agent)
        self.updates_history: dict[str, list[npt.NDArray[np.floating]]] = {
            p.name: [p.value] for p in agent.learnable_parameters.values()
        }

    def _on_update(self, *_, **__) -> None:
        """Internal utility to store the current parameters in memory."""
        for par in self.agent.learnable_parameters.values():
            self.updates_history[par.name].append(par.value)

    def establish_callback_hooks(self) -> None:
        super().establish_callback_hooks()
        # connect the agent's on_update callback to this wrapper storing action
        self.hook_callback(repr(self), "on_update", self._on_update)
