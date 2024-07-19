import numpy as np
import numpy.typing as npt

from ...agents.common.learning_agent import ExpType, LearningAgent
from .wrapper import LearningWrapper, SymType


class RecordUpdates(LearningWrapper[SymType, ExpType]):
    """Wrapper for recording the history of updated parametrizations by the learning
    agent.

    In other words, it records the new value of the parameters in
    :attr:`mpcrl.LearningAgent.learnable_parameters` after every call to
    :meth:`mpcrl.LearningAgent.update`. This information can be retrieved from the
    attribute :attr:`updates_history`.

    Parameters
    ----------
    agent : LearningAgent or subclass
        The agent whose updates need recording.
    """

    def __init__(self, agent: LearningAgent[SymType, ExpType]) -> None:
        super().__init__(agent)
        self.updates_history: dict[str, list[npt.NDArray[np.floating]]] = {
            p.name: [p.value] for p in agent.learnable_parameters.values()
        }

    def _on_update(self, *_, **__) -> None:
        for par in self.agent.learnable_parameters.values():
            self.updates_history[par.name].append(par.value)

    def _establish_callback_hooks(self) -> None:
        super()._establish_callback_hooks()
        # connect the agent's on_update callback to this wrapper storing action
        self._hook_callback(repr(self), "on_update", self._on_update)
