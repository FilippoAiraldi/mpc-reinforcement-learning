from typing import Any, Dict, Generic, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T")  # most like, T is cs.SX or MX


class LearnableParameter(Generic[T]):
    """A 1D parameter that is learnable, that is, it can be adjusted via . This class
    is useful for managing symbols, bounds and value of learnable parameters.

    Parameters
    ----------
    name : str
        Name of the learnable parameter.
    size : int
        Size of the 1D parameter vector.
    value : array_like
        Starting value of the parameter. This can then be updated via `update` method.
    lb : array_like, optional
        Lower bound of the parameter values. If not specified, it is unbounded.
    ub : array_like, optional
        Upper bound of the parameter values. If not specified, it is unbounded.
    syms : dict[str, Any], optional
        An optional dictionary contaring references to different symbols linked to this
        parameter. For example, in MPC-based RL, it is convenient to keep a reference to
        the parameter's symbol for the V function and for the Q function, i.e.,
        `syms = {"V": V_par_sym, "Q": Q_par_sym}`

    Raises
    ------
    ValueError
        Raises if `value`, `lb` or `ub` cannot be broadcasted to a 1D vector with shape
        equal to `(size,)`.
    """

    __slots__ = (
        "name",
        "size",
        "value",
        "lb",
        "ub",
        "syms",
    )

    def __init__(
        self,
        name: str,
        size: int,
        value: npt.ArrayLike,
        lb: npt.ArrayLike = -np.inf,
        ub: npt.ArrayLike = +np.inf,
        syms: Dict[str, Any] = None,
    ) -> None:
        """_summary_

        Parameters
        ----------
        name : str
            Name of the learnable parameter.
        size : int
            Size of the 1D parameter vector.
        value : array_like
            Starting value of the parameter. This can then be updated via `update`.
        lb : array_like, optional
            Lower bound of the parameter values. If not specified, it is unbounded.
        ub : array_like, optional
            Upper bound of the parameter values. If not specified, it is unbounded.
        syms : dict[str, Any], optional
            An optional dictionary contaring references to different symbols linked to
            this parameter. For example, in MPC-based RL, it is convenient to keep a
            reference to the parameter's symbol for the V function and for the Q
            function, i.e., `syms = {"V": V_par_sym, "Q": Q_par_sym}`

        Raises
        ------
        ValueError
            Raises if `value`, `lb` or `ub` cannot be broadcasted to a 1D vector with
            shape equal to `(size,)`.
        """
        super().__init__()
        self.name = name
        self.size = size
        self.syms = syms
        shape = (size,)
        self.lb = np.broadcast_to(lb, shape)
        self.ub = np.broadcast_to(ub, shape)
        self.update(value)

    def update(self, v: npt.ArrayLike) -> None:
        """Updates the parameter value with a new value.

        Parameters
        ----------
        new_value : array_like
            New value of the parameter.

        Raises
        ------
        ValueError
            Raises if `new_value` cannot be broadcasted to a 1D vector with shape equal
            to `(size,)`; or if it does not lie inside the upper and lower bounds within
            the specified tolerances.
        """
        v = np.broadcast_to(v, (self.size,))
        lb = self.lb
        ub = self.ub
        if ((v < lb) & ~np.isclose(v, lb)).any() or (
            (v > ub) & ~np.isclose(v, ub)
        ).any():
            raise ValueError(f"Updated parameter {self.name} is outside bounds.")
        self.value = np.clip(v, lb, ub)
