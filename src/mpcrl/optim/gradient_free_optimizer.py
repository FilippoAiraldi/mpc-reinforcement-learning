from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from .base_optimizer import BaseOptimizer, SymType


class GradientFreeOptimizer(BaseOptimizer[SymType], ABC):
    """Base class for gradient-free optimization algorithms, e.g., Bayesian
    Optimization.

    This optimizer adopts the ask-tell interface, i.e., it must implement the
    :meth:`GradientFreeOptimizer.ask` and :meth:`GradientFreeOptimizer.tell` methods.
    The former allows the agent to ask for a new set of parameters to evaluate, while
    the latter allows the agent to tell the optimizer the values of the objective
    function(s) for the set of parameters it asked for.
    """

    prefers_dict: bool
    """A flag that specifies whether the optimizer prefers to receive the learnable
    parameters as a dictionary of names and values or as a single concatenated array."""

    @abstractmethod
    def ask(
        self,
    ) -> tuple[
        Union[dict[str, npt.ArrayLike], npt.ArrayLike],
        Optional[str],
    ]:
        """Asks the learning agent for a new set of parameters to evaluate.

        Returns
        -------
        dict of (str, 1d arrays) or a single 1d array
            A dictionary of learnable parameter names and their corresponding values.
            Or a single array that results from the concatenation of the parameter
            values.
        str, optional (default=None)
            A string that specifies the status of the optimizer. This is useful to
            communicate to the learning agent whether the optimization algorithm has
            encountered, e.g., some error or failure.
        """

    @abstractmethod
    def tell(
        self,
        values: Union[dict[str, npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        objective: Union[float, npt.NDArray[np.floating]],
    ) -> None:
        """Tells the learning agent the values of the objective function for the set of
        parameters it asked for.

        Parameters
        ----------
        values : dict of (str, 1d arrays) or a single 1d array
            A dictionary of learnable parameter names and their corresponding values for
            which the objective function(s) was (were) evaluated. Or a single array that
            results from the concatenation of the parameter values. This depends on the
            optimizer's :attr:`prefers_dict` class attribute.
        objective : float or array
            Value(s) of the objective function(s) for the set of parameters. Can be
            single-objective or multi-objective.
        """
