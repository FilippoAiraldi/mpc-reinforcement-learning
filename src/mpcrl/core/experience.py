from itertools import chain
from typing import Deque, Generator, Iterable, Optional, TypeVar, Union

from csnlp.nlp.funcs import np_random

T = TypeVar("T")


class ExperienceReplay(Deque[T]):
    """Deque-based class for RL traning to save and sample experience transitions. The
    class inherits from `collections.deque`, adding a couple of simple functionalities
    to it for sampling transitions at random from past observed data."""

    __slots__ = "np_random"

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

    def sample(
        self, n: Union[int, float], last_n: Union[int, float] = 0
    ) -> Generator[T, None, None]:
        """
        Samples the experience memory and yields the sampled items.

        Parameters
        n : int or float
            Size of the sample to draw from memory, either as integer or percentage of
            the maximum capacity (requires `maxlen != None`).
        last_n : int or float, optional
            Size or percentage of the sample (not of the total memory length) dedicated
            to including the last items added to the memory. By default, `last_n = 0`.

        Returns
        -------
        sample : Iterable of T
            An iterable sample is yielded.

        Raises
        ------
        TypeError
            Raises if `n` is float (a percentage of the maximum length), but `maxlen` is
            `None`.
        """
        L = len(self)
        if isinstance(n, float):
            n = int(self.maxlen * n)  # type: ignore
        n = min(max(n, 0), L)
        if isinstance(last_n, float):
            last_n = int(n * last_n)
        last_n = min(max(last_n, 0), n)

        # get last n indices and the sampled indices from the remaining
        last = range(L - last_n, L)
        sampled = self.np_random.choice(range(L - last_n), n - last_n, replace=False)
        yield from (self[i] for i in chain(last, sampled))
