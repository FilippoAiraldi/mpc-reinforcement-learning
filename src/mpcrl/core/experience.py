from itertools import chain
from typing import Deque, Iterable, Iterator, Optional, TypeVar, Union

from mpcrl.util.random import np_random

ExpType = TypeVar("ExpType")


class ExperienceReplay(Deque[ExpType]):
    """Deque-based class for RL traning to save and sample experience transitions. The
    class inherits from `collections.deque`, adding a couple of simple functionalities
    to it for sampling transitions at random from past observed data."""

    __slots__ = ("np_random", "sample_size", "include_last")

    def __init__(
        self,
        iterable: Iterable[ExpType] = (),
        maxlen: Optional[int] = None,
        sample_size: Union[int, float] = 1,
        include_last: Union[int, float] = 0,
        seed: Optional[int] = None,
    ) -> None:
        """Instantiate the container for experience replay memory.

        Parameters
        ----------
        iterable : Iterable of T, optional
            Initial items to be inserted in the container. By default, empty.
        maxlen : int, optional
            Maximum length/capacity of the memory. If `None`, the deque has no maximum
            size, which is the default behaviour.
        sample_size : int or float, optional
            Size (or percentage of replay `maxlen`) of the experience replay items to
            draw when performing an update. By default, one item per sampling is drawn.
        include_last : int or float, optional
            Size (or percentage of sample size) dedicated to including the latest
            experience transitions. By default, 0, i.e., no last item is included.
        seed : int, optional
            Seed for the random number generator. By default, `None`.
        """
        super().__init__(iterable, maxlen=maxlen)
        self.sample_size = sample_size
        self.include_last = include_last
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Resets the sampling RNG."""
        self.np_random = np_random(seed)

    def sample(self) -> Iterator[ExpType]:
        """Samples the experience memory and yields the sampled items.

        Returns
        -------
        sample : iterator of T
            An iterable sample is yielded.

        Raises
        ------
        TypeError
            Raises if `sample_size` is a float (a percentage of the maximum length), but
            `maxlen` is `None`, since it is impossible to compute the percentage of an
            unknown quantity.
        """
        L = len(self)
        n = self.sample_size
        last_n = self.include_last
        if isinstance(n, float):
            assert (
                self.maxlen is not None
            ), "Maxlen is `None`; cannot use sample percentages."
            n = int(self.maxlen * n)
        n = min(max(n, 0), L)
        if isinstance(last_n, float):
            last_n = int(n * last_n)
        last_n = min(max(last_n, 0), n)

        # get last n indices and the sampled indices from the remaining
        last = range(L - last_n, L)
        sampled = self.np_random.choice(range(L - last_n), n - last_n, False)
        yield from (self[i] for i in chain(last, sampled))
