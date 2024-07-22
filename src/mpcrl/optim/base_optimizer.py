from typing import Any, Generic

import numpy as np

from ..core.parameters import LearnableParametersDict, SymType


class BaseOptimizer(Generic[SymType]):
    """Base class for optimization algorithms.

    This class contains useful methods for, e.g., initializing the optimizer, retrieving
    bounds on the learnable parameters, etc.

    Parameters
    ----------
    max_percentage_update : float, optional
        A positive float that specifies the maximum percentage change the learnable
        parameters can experience in each update. For example,
        ``max_percentage_update=0.5`` means that the parameters can be updated by up
        to 50% of their current value. By default, it is set to ``+inf``.
    """

    def __init__(self, max_percentage_update: float = float("+inf")) -> None:
        self.max_percentage_update = max_percentage_update
        self.learnable_parameters: LearnableParametersDict[SymType] = None

    def set_learnable_parameters(self, pars: LearnableParametersDict[SymType]) -> None:
        """Makes the optimization class aware of the dictionary of the learnable
        parameters whose values are to be updated.

        Parameters
        ----------
        pars : :class`mpcrl.LearnableParametersDict`
            The dictionary of the learnable parameters.
        """
        self.learnable_parameters = pars
        self._update_solver = self._init_update_solver()

    def _get_update_bounds(
        self, theta: np.ndarray, eps: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Internal utility to retrieve the current bounds on the learnable parameters.
        Only useful if the update problem is not unconstrained, i.e., there are either
        some lower- or upper-bounds, or a maximum percentage update was given."""
        lb = self.learnable_parameters.lb - theta
        ub = self.learnable_parameters.ub - theta
        perc = self.max_percentage_update
        if perc != float("+inf"):
            max_update_delta = np.maximum(np.abs(perc * theta), eps)
            lb = np.maximum(lb, -max_update_delta)
            ub = np.minimum(ub, +max_update_delta)
        return lb, ub

    def _init_update_solver(self) -> Any:
        """Internal utility to initialize whatever solver is necessary to perform an
        update according to this learning strategy."""

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        mp = self.max_percentage_update
        return f"{cn}(max%={mp})"
