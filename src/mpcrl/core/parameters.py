"""Given an MPC controller with several symbolic parameters (some meant to be learned,
some other not), we need a way to specify to the agent of choice which of these are
indeed learnable. This is done by the use of the two classes introduced in this
submodule.

Namely, :class:`LearnableParameter` allows to embed a single parameter and its
information, while :class:`LearnableParametersDict` is a dictionary-like class that
contains several of these :class:`LearnableParameter` instances, and offers different
properties and methods to manage them in bulk.

See also :ref:`user_guide_learnable_parameters`."""

from collections.abc import Iterable
from functools import cached_property
from itertools import chain
from numbers import Integral
from typing import Any, Generic, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt
from csnlp.core.cache import invalidate_cache
from csnlp.util.io import SupportsDeepcopyAndPickle

from ..util.math import summarize_array

SymType = TypeVar("SymType")  # most likely, T is cs.SX or MX


class LearnableParameter(SupportsDeepcopyAndPickle, Generic[SymType]):
    """A parameter that is learnable, that is, it can be adjusted via RL or any other
    learning strategy. This class is useful for managing symbols, bounds and value of
    the learnable parameter.

    Parameters
    ----------
    name : str
        Name of the learnable parameter.
    shape : int or tuple of ints
        Shape of the parameter.
    value : array_like
        Starting value of the parameter.
    lb : array_like, optional
        Lower bound of the parameter values. If not specified, it is unbounded.
    ub : array_like, optional
        Upper bound of the parameter values. If not specified, it is unbounded.
    sym : T, optional
        An optional reference to a symbolic variable representing this parameter.

    Raises
    ------
    ValueError
        Raises if ``value``, ``lb`` or ``ub`` cannot be broadcasted to a 1D vector with
        shape equal to ``shape``; or if the shape of the symbolic variable ``sym`` does
        not match the shape of the parameter.
    """

    def __init__(
        self,
        name: str,
        shape: Union[int, tuple[int, ...]],
        value: npt.ArrayLike,
        lb: npt.ArrayLike = -np.inf,
        ub: npt.ArrayLike = +np.inf,
        sym: Optional[SymType] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.shape: tuple[int, ...] = (shape,) if isinstance(shape, Integral) else shape
        self.sym = sym
        self.lb: npt.NDArray[np.floating] = np.broadcast_to(lb, shape)
        self.ub: npt.NDArray[np.floating] = np.broadcast_to(ub, shape)
        self._check_sym_shape()
        self._update_value(value)

    @property
    def size(self) -> int:
        """Gets the number of elements in the parameter."""
        return np.prod(self.shape, dtype=int).item()

    def _update_value(self, v: npt.ArrayLike, **is_close_kwargs: Any) -> None:
        """Internal utility for updating the parameter value with a new value.

        Parameters
        ----------
        new_value : array_like
            New value of the parameter.
        is_close_kwargs
            Additional kwargs for :func:`numpy.isclose`, e.g., ``rtol`` and ``atol``,
            for checking numerical values close to a bound.

        Raises
        ------
        ValueError
            Raises if ``new_value`` cannot be broadcasted to a 1D vector with shape
            equal to ``shape``; or if it does not lie inside the upper and lower bounds
            within the specified tolerances.
        """
        v = np.broadcast_to(v, self.shape)
        lb = self.lb
        ub = self.ub
        if ((v < lb) & ~np.isclose(v, lb, **is_close_kwargs)).any() or (
            (v > ub) & ~np.isclose(v, ub, **is_close_kwargs)
        ).any():
            raise ValueError(
                f"Updated parameter `{self.name}` outside bounds: {lb} <= {v} <= {ub}."
            )
        self.value: npt.NDArray[np.floating] = np.clip(v, lb, ub)

    def _check_sym_shape(self) -> None:
        """Internal utility for checking that the shape of the symbolic variable matches
        the shape of the parameter."""
        if self.sym is None or not hasattr(self.sym, "shape"):
            return
        sym_shape = tuple(self.sym.shape)
        shape = self.shape + tuple(1 for _ in range(len(sym_shape) - len(self.shape)))
        if sym_shape != shape:
            raise ValueError("Shape of `sym` does not match `shape`.")

    def __str__(self) -> str:
        return f"<{self.name}(shape={self.shape})>"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name},shape={self.shape})>"


class LearnableParametersDict(
    dict[str, LearnableParameter[SymType]], SupportsDeepcopyAndPickle
):
    """:class:`dict`-based collection of :class:`LearnableParameter` instances that
    simplifies the process of managing and updating these. The dict contains pairs of
    parameter's name and parameter's instance.

    Parameters can be retrieved as a normal dictionary by their names, but the class
    also offers several properties that are useful for managing the parameters in bulk,
    such as :attr:`lb`, :attr:`ub`, :attr:`value`, :attr:`value_as_dict` and
    :attr:`sym`. With a single call to :meth:`update_values`, the values of the
    parameters can also be updated.

    Parameters
    ----------
    pars : iterable of :class:`LearnableParameter`, optional
        An optional iterable of parameters to insert into the dict by their names.

    Notes
    -----
    To speed up computations, properties of this class are often cached for faster
    calls to the same methods. However, these are automatically cleared when the
    underlying dict is modified.
    """

    def __init__(self, pars: Optional[Iterable[LearnableParameter[SymType]]] = None):
        if pars is None:
            dict.__init__(self)
        else:
            dict.__init__(self, map(lambda p: (p.name, p), chain(pars)))
        SupportsDeepcopyAndPickle.__init__(self)

    @cached_property
    def size(self) -> int:
        """Gets the overall size of all the learnable parameters."""
        return sum(p.size for p in self.values())

    @cached_property
    def lb(self) -> npt.NDArray[np.floating]:
        """Gets the lower bound of all the learnable parameters, concatenated."""
        return (
            np.concatenate([p.lb.reshape(-1, order="F") for p in self.values()])
            if self
            else np.empty(0)
        )

    @cached_property
    def ub(self) -> npt.NDArray[np.floating]:
        """Gets the upper bound of all the learnable parameters, concatenated."""
        return (
            np.concatenate([p.ub.reshape(-1, order="F") for p in self.values()])
            if self
            else np.empty(0)
        )

    @cached_property
    def value(self) -> npt.NDArray[np.floating]:
        """Gets the values of all the learnable parameters, concatenated."""
        return (
            np.concatenate([p.value.reshape(-1, order="F") for p in self.values()])
            if self
            else np.empty(0)
        )

    @cached_property
    def value_as_dict(self) -> dict[str, npt.NDArray[np.floating]]:
        """Gets the values of all the learnable parameters as a :class:`dict`."""
        return {p.name: p.value for p in self.values()}

    @cached_property
    def sym(self) -> dict[str, Optional[SymType]]:
        """Gets symbols of all the learnable parameters, in a dict. If one parameter
        does not possess the symbol, ``None`` is put."""
        return {
            parname: None if par.sym is None else par.sym
            for parname, par in self.items()
        }

    @invalidate_cache(value, value_as_dict)
    def update_values(
        self,
        new_values: Union[npt.ArrayLike, dict[str, npt.ArrayLike]],
        **is_close_kwargs: Any,
    ) -> None:
        """Updates the value of each parameter

        Parameters
        ----------
        new_values : array_like or dict of (str, array_like)
            The parameters' new values, either as a single concatenated array (which
            will be splitted according to the sizes and each piece sequentially assigned
            to each parameter), or as a dict of parameter's name vs parameter's new
            value.
        is_close_kwargs
            Additional kwargs for :func:`numpy.isclose`, e.g., ``rtol`` and ``atol``,
            for checking numerical values of parameters close to a bound.

        Raises
        ------
        ValueError
            In case of array-like, raises if ``new_values`` cannot be split according to
            the sizes of parameters; or if the new values cannot be broadcasted to 1D
            vectors according to each parameter's size; or if the new values lie outside
            either the lower or upper bounds of each parameter.
        """
        if isinstance(new_values, dict):
            for parname, new_value in new_values.items():
                self[parname]._update_value(new_value, **is_close_kwargs)
        else:
            cumsizes = np.cumsum([p.size for p in self.values()])[:-1]
            values_ = np.split(new_values, cumsizes)
            for par, val in zip(self.values(), values_):
                par._update_value(val.reshape(par.shape, order="F"), **is_close_kwargs)

    __cache_decorator = invalidate_cache(size, lb, ub, value, value_as_dict, sym)

    @__cache_decorator
    def __setitem__(self, name: str, par: LearnableParameter[SymType]) -> None:
        assert name == par.name, f"Key '{name}' must match parameter name '{par.name}'."
        return super().__setitem__(name, par)

    @__cache_decorator
    def update(
        self,
        pars: Iterable[LearnableParameter[SymType]],
        *args: LearnableParameter[SymType],
    ) -> None:
        return super().update(map(lambda p: (p.name, p), chain(pars, args)))

    @__cache_decorator
    def setdefault(
        self,
        par: LearnableParameter[SymType],
    ) -> LearnableParameter[SymType]:
        return super().setdefault(par.name, par)

    __delitem__ = __cache_decorator(dict.__delitem__)
    pop = __cache_decorator(dict.pop)
    popitem = __cache_decorator(dict.popitem)
    clear = __cache_decorator(dict.clear)

    def copy(
        self, deep: bool = False, invalidate_caches: bool = True
    ) -> "LearnableParametersDict[SymType]":
        """Creates a shallow or deep copy of the dict of learnable parameters.

        Parameters
        ----------
        deep : bool, optional
            If ``True``, a deepcopy of the dict and its parameters is returned;
            otherwise, the copy is only shallow.
        invalidate_caches : bool, optional
            If `True`, methods decorated with :func:`csnlp.core.cache.invalidate_cache`
            are called to clear cached properties/lru caches in the copied instance.
            Otherwise, caches in the copy are not invalidated. By default, ``True``.
            Only relevant when ``deep=True``.

        Returns
        -------
        LearnableParametersDict[T]
            A copy of the dict of learnable parameters.
        """
        return (
            SupportsDeepcopyAndPickle.copy(self, invalidate_caches)
            if deep
            else LearnableParametersDict[SymType](self.values())
        )

    def stringify(
        self, summarize: bool = True, precision: int = 3, ddof: int = 0
    ) -> str:
        """Returns a string representing the dict of learnable parameters.

        Parameters
        ----------
        summarize : bool, optional
            If ``True`` (default), array parameters are summarized; otherwise, the
            entire array is printed.
        precision : int, optional
            The printing precision of floating point numbers.
        ddof : int, optional
            Degrees of freedom for computing standard deviations (see
            :func:`numpy.std`).

        Returns
        -------
        str
            A string representing the dict and its parameters.
        """

        def p2s(p: LearnableParameter) -> str:
            if p.size == 1:
                return f"{p.name}={p.value.item():.{precision}f}"
            if summarize:
                return f"{p.name}: {summarize_array(p.value, precision, ddof)}"
            return np.array2string(p.value, precision=precision)

        return "; ".join(p2s(p) for p in self.values())

    def __str__(self) -> str:
        return self.stringify()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {super().__repr__()}>"
