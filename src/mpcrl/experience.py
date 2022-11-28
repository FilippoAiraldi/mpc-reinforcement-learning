from typing import Deque, Iterable, Optional, TypeVar

from csnlp.util.funcs import np_random

T = TypeVar("T")


class ExperienceReplay(Deque[T]):
    """Container class for RL traning to save and sample experience transitions. The
    class inherits from `collections.deque`, adding a couple of simple functionalities
    to it for sampling transitions at random from past observed data."""

    def __init__(
        self,
        iterable: Iterable[T] = (),
        maxlen: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Instantiate the container for experience replay memory.

        Parameters
        ----------
        iterable : Iterable of T, optional
            Initial items to be inserted in the container. By default, empty.
        maxlen : int, optional
            Maximum length/capacity of the memory. If `None`, the deque has no maximum
            size, which is the default behaviour.
        seed : int, optional
            Seed for the random number generator. By default, `None`.
        """
        super().__init__(iterable, maxlen=maxlen)
        self.np_random, _ = np_random(seed)
