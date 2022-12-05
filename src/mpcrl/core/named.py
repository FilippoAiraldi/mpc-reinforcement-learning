from itertools import count
from typing import Dict, Iterator, Optional, Type


class Named:
    """Base class for objects with names."""

    __ids: Dict[Type, Iterator[int]] = {}

    def __init__(self, name: Optional[str] = None) -> None:
        """Instantiates the class with the given `name` attribute.

        Parameters
        ----------
        name : str, optional
            Name of the object. If `None`, one is automatically created from a counter
            of the class' instancies.
        """
        cls = self.__class__
        if cls in self.__ids:
            _id = self.__ids[cls]
        else:
            _id = count(0)
            self.__ids[cls] = _id
        self.name = name or f"{cls.__name__}{next(_id)}"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<{self.name}: {self.__class__.__name__}>"
