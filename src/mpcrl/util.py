from collections import UserDict
from collections.abc import Mapping
from typing import Iterable, Tuple, Union

import numpy as np


class RangeNormalization(UserDict):
    """Class for the normalization of quantities. It suffices to register the range of
    the variable, and then it can be easily (de-)normalized according to such range."""

    def can_normalize(self, name: str) -> bool:
        """Checks whether there exists a range under the given name.

        Parameters
        ----------
        name : str
            Name of the normalization range.

        Returns
        -------
        bool
            Whether range `name` exists for normalization.
        """
        return name in self.data

    def register(
        self,
        other: Union[Mapping, Iterable[Tuple[str, np.ndarray]]] = None,
        **kwargs: np.ndarray,
    ) -> None:  # sourcery skip: dict-assign-update-to-union
        """Registers new normalization ranges, but raises if duplicates occur.

        Parameters
        ----------
        other : Mapping or iterable, optional
            A mapping (e.g., dict) of normalization ranges, by default `None`.

        Raises
        ------
        KeyError
            Raises if a duplicate key is detected.
        """
        if other is not None:
            kwargs.update(other)
        for k, v in kwargs.items():
            if k in self.data:
                raise KeyError(f"'{k}' already registered for normalization.")
            self.data[k] = v

    def normalize(self, name: str, x: np.ndarray) -> np.ndarray:
        """Normalizes the value `x` according to the ranges of `name`.

        Parameters
        ----------
        name : str
            Ranges to be used.
        x : array_like
            Value to be normalized. If an array, then take care that the last
            dimension matches the dimension of the normalization ranges,
            otherwise the output will have a different shape.

        Returns
        -------
        array_like
            Normalized `x`.

        Raises
        ------~
        AssertionError
            Raises if normalized output's shape does not match input's shape.
        """
        r = self.data[name]
        if np.ndim(r) == 1:
            out = (x - r[0]) / (r[1] - r[0])
        else:
            out = (x - r[:, 0]) / (r[:, 1] - r[:, 0])
        assert np.shape(out) == np.shape(x), "Normalization altered input shape."
        return out

    def denormalize(self, name: str, x: np.ndarray) -> np.ndarray:
        """Denormalizes the value `x` according to the ranges of `name`.

        Parameters
        ----------
        name : str
            Ranges to be used.
        x : array_like
            Value to be denormalized.

        Returns
        -------
        array_like
            Denormalized `x`.

        Raises
        ------
        AssertionError
            Raises if denormalized output's shape does not match input's shape.
        """
        r = self.data[name]
        if np.ndim(r) == 1:
            out = (r[1] - r[0]) * x + r[0]
        else:
            out = (r[:, 1] - r[:, 0]) * x + r[:, 0]
        assert np.shape(out) == np.shape(x), "Denormalization altered input shape."
        return out

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {super().__repr__()}>"

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}: {super().__str__()}>"
