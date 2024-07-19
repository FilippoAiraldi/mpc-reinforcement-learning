"""A submodule with an utility class for assigning unique names to each instance of a
subclass. This is useful, for instance, for debugging and logging purposes when multiple
agents are trained in parallel."""

from collections.abc import Iterator as _Iterator
from itertools import count as _count
from typing import Optional


class Named:
    """Base class for objects with names. It assigns a unique name to each instance of a
    subclass by appending an ID (an incremental integer) to the class' name or the
    provided one.

    Parameters
    ----------
    name : str, optional
        Name of the object. If `None`, one is automatically created from a counter
        of the class' instancies.
    """

    __ids: dict[type, _Iterator[int]] = {}

    def __init__(self, name: Optional[str] = None) -> None:
        cls = self.__class__
        if cls in self.__ids:
            _id = self.__ids[cls]
        else:
            _id = _count(0)
            self.__ids[cls] = _id
        self.name = name or f"{cls.__name__}{next(_id)}"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
