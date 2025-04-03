import numpy as np
import numpy.typing as npt

from ...agents.common.learning_agent import ExpType, LearningAgent
from ...util.iters import bool_cycle
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
    frequency : int, optional
        The frequency of recording the updates. If the frequency is set to ``1``, all
        updates are recorded. If the frequency is set to ``2``, every second update is
        recorded, and so on. By default, ``1``. Note that the first values of the
        parameters are always recorded.
    """

    def __init__(
        self,
        agent: LearningAgent[SymType, ExpType],
        frequency: int = 1,
    ) -> None:
        super().__init__(agent)
        self._record_cycle = bool_cycle(frequency)
        self.updates_history: dict[str, list[npt.NDArray[np.floating]]] = {
            p.name: [p.value] for p in agent.learnable_parameters.values()
        }

    def _on_update(self, *_, **__) -> None:
        if next(self._record_cycle):
            for par in self.agent.learnable_parameters.values():
                self.updates_history[par.name].append(par.value)

    def _establish_callback_hooks(self) -> None:
        super()._establish_callback_hooks()
        # connect the agent's on_update callback to this wrapper storing action
        self._hook_callback(repr(self), "on_update", self._on_update)
