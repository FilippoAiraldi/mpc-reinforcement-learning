from collections.abc import Iterable, Iterator
from itertools import chain
from typing import Deque, Optional, TypeVar, Union

import numpy as np

from ..util.seeding import RngType

ExpType = TypeVar("ExpType")


class ExperienceReplay(Deque[ExpType]):
    """Deque-based class for RL traning to save and sample experience transitions. The
    class inherits from `collections.deque`, adding a couple of simple functionalities
    to it for sampling transitions at random from past observed data."""

    def __init__(
        self,
        iterable: Iterable[ExpType] = (),
        maxlen: Optional[int] = None,
        sample_size: Union[int, float] = 1,
        include_latest: Union[int, float] = 0,
        seed: RngType = None,
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
        include_latest : int or float, optional
            Size (or percentage of `sample_size`) dedicated to including the latest
            experience transitions. By default, 0, i.e., no last item is included.
        seed : None, int, array_like[ints], SeedSequence, BitGenerator, Generator
            Seed for the random number generator. By default, `None`.
        """
        super().__init__(iterable, maxlen=maxlen)
        self.sample_size = sample_size
        self.include_latest = include_latest
        self.reset(seed)

    def reset(self, seed: RngType = None) -> None:
        """Resets the sampling RNG."""
        self.np_random = np.random.default_rng(seed)

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
        last_n = self.include_latest
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
        yield from (self[i] for i in chain(sampled, last))
