from functools import cached_property
from itertools import chain
from typing import Dict, Generic, Iterable, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt
from csnlp.core.cache import invalidate_cache

T = TypeVar("T")  # most likely, T is cs.SX or MX


class LearnableParameter(Generic[T]):
    """A 1D parameter that is learnable, that is, it can be adjusted via . This class
    is useful for managing symbols, bounds and value of learnable parameters."""

    __slots__ = (
        "name",
        "size",
        "value",
        "lb",
        "ub",
        "sym",
    )

    def __init__(
        self,
        name: str,
        size: int,
        value: npt.ArrayLike,
        lb: npt.ArrayLike = -np.inf,
        ub: npt.ArrayLike = +np.inf,
        sym: Optional[T] = None,
    ) -> None:
        """Instantiates a learnable parameter.

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
        sym : T, optional
            An optional reference to a symbolic variable representing this parameter.

        Raises
        ------
        ValueError
            Raises if `value`, `lb` or `ub` cannot be broadcasted to a 1D vector with
            shape equal to `(size,)`.
        """
        super().__init__()
        self.name = name
        self.size = size
        self.sym = sym
        shape = (size,)
        self.lb: npt.NDArray[np.double] = np.broadcast_to(lb, shape)
        self.ub: npt.NDArray[np.double] = np.broadcast_to(ub, shape)
        self._update_value(value)

    def _update_value(self, v: npt.ArrayLike) -> None:
        """Internal utility for updating the parameter value with a new value.

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
        self.value: npt.NDArray[np.double] = np.clip(v, lb, ub)

    def __str__(self) -> str:
        return f"<{self.name}(size={self.size})>"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name},size={self.size})>"


class LearnableParametersDict(Dict[str, LearnableParameter[T]], Generic[T]):
    """Dict-based collection of `LearnableParameter` instances that simplifies the
    process of managing and updating these. The dict contains pairs of parameter's name
    vs parameter's instance.

    Note: to speed up computations, properties of this class are often cached for faster
    calls to the same methods. However, these are cleared when the underlying dict is
    modified."""

    def __init__(self, pars: Optional[Iterable[LearnableParameter[T]]] = None):
        """Initializes the collection of learnable parameters.

        Parameters
        ----------
        pars : iterable of LearnableParameter, optional
            An optional iterable of parameters to insert into the dict by their names.
        """
        if pars is None:
            super().__init__()
        else:
            super().__init__(map(lambda p: (p.name, p), chain(pars)))

    @cached_property
    def size(self) -> int:
        """Gets the overall size of all the learnable parameters."""
        return sum(p.size for p in self.values())

    @cached_property
    def lb(self) -> npt.NDArray[np.double]:
        """Gets the lower bound of all the learnable parameters, concatenated."""
        if len(self) == 0:
            return np.asarray([])
        return np.concatenate(tuple(p.lb for p in self.values()))

    @cached_property
    def ub(self) -> npt.NDArray[np.double]:
        """Gets the upper bound of all the learnable parameters, concatenated."""
        if len(self) == 0:
            return np.asarray([])
        return np.concatenate(tuple(p.ub for p in self.values()))

    @cached_property
    def value(self) -> npt.NDArray[np.double]:
        """Gets the values of all the learnable parameters, concatenated."""
        if len(self) == 0:
            return np.asarray([])
        return np.concatenate(tuple(p.value for p in self.values()))

    @cached_property
    def sym(self) -> Dict[str, Optional[T]]:
        """Gets symbols of all the learnable parameters, in a dict. If one parameter
        does not possess the symbol, `None` is put."""
        return {
            parname: None if par.sym is None else par.sym
            for parname, par in self.items()
        }

    @invalidate_cache(value)
    def update_values(
        self, new_values: Union[npt.ArrayLike, Dict[str, npt.ArrayLike]]
    ) -> None:
        """Updates the value of each parameter

        Parameters
        ----------
        new_values : array_like or dict[str, array_like]
            The parameters' new values, either as a single concatenated array (which
            will be splitted according to the sizes and each piece sequentially assigned
            to each parameter), or as a dict of parameter's name vs parameter's new
            value.

        Raises
        ------
        ValueError
            In case of array-like, raises if `new_values` cannot be split according to
            the sizes of parameters; or if the new values cannot be broadcasted to 1D
            vectors according to each parameter's size; or if the new values lie outside
            either the lower or upper bounds of each parameter.
        """
        if isinstance(new_values, dict):
            for parname, new_value in new_values.items():
                self[parname]._update_value(new_value)
        else:
            cumsizes = np.cumsum([p.size for p in self.values()])[:-1]
            values_ = np.split(new_values, cumsizes)
            for par, value in zip(self.values(), values_):
                par._update_value(value)

    __cache_decorator = invalidate_cache(size, lb, ub, value, sym)

    @__cache_decorator
    def __setitem__(self, name: str, par: LearnableParameter) -> None:
        assert name == par.name, f"Key '{name}' must match parameter name '{par.name}'."
        return super().__setitem__(name, par)

    @__cache_decorator
    def update(
        self, pars: Iterable[LearnableParameter[T]], *args: LearnableParameter[T]
    ) -> None:
        return super().update(map(lambda p: (p.name, p), chain(pars, args)))

    @__cache_decorator
    def setdefault(self, par: LearnableParameter[T]) -> LearnableParameter[T]:
        return super().setdefault(par.name, par)

    __delitem__ = __cache_decorator(dict.__delitem__)
    pop = __cache_decorator(dict.pop)
    popitem = __cache_decorator(dict.popitem)
    clear = __cache_decorator(dict.clear)
