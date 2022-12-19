from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt

from mpcrl.agents.learning_agent import ExpType, LearningAgent
from mpcrl.wrappers.wrapper import LearningWrapper, SymType


class RecordUpdates(LearningWrapper[SymType, ExpType]):

    __slots__ = ("learnable_parameters_history",)

    def __init__(self, agent: LearningAgent[SymType, ExpType]) -> None:
        super().__init__(agent)
        self.learnable_parameters_history: Dict[str, List[npt.NDArray[np.double]]] = {
            p.name: [p.value] for p in agent.learnable_parameters.values()
        }

    def update(self) -> Optional[str]:
        """See `LearningAgent.update`."""
        out = self.agent.update()
        for par in self.agent.learnable_parameters.values():
            self.learnable_parameters_history[par.name].append(par.value)
        return out
