"""Naively, Reinforcement Learning algorithms can dish out an update of the MPC
parametrization at every time step, by leveraging only the current information. However,
as in Deep RL, it makes sense to enable the agent to store past experiences and
re-use them, or at least use them in a batched fashion, to improve the stability and
convergence learning process. :class:`ExperienceReplay` allows a learning agent to store
and sample, when performing an update, past experiences. See
:ref:`user_guide_experience` for a more detailed explanation."""

from collections import deque
from collections.abc import Iterable, Iterator
from itertools import chain
from typing import Optional, TypeVar, Union

import numpy as np

from ..util.seeding import RngType

ExpType = TypeVar("ExpType")


class ExperienceReplay(deque[ExpType]):
    """Class for Reinforcement Learning agents' traning to save and sample experience
    transitions.

    The class inherits from :class:`deque`, adding a couple of simple functionalities to
    it for sampling transitions at random from past observed data (see :meth:`reset` and
    :meth:`sample`).

    Parameters
    ----------
    iterable : Iterable of ExpType, optional
        Initial items to be inserted in the container. By default, empty.
    maxlen : int, optional
        Maximum length/capacity of the memory. If ``None``, the deque has no maximum
        size, which is the default behaviour.
    sample_size : int or float, optional
        Size (as integer, or float percentage of ``maxlen``) of the experience replay
        items to draw when performing an update. By default, one item per sampling is
        drawn. If a float percentage, ``maxlen`` must be provided.
    include_latest : int or float, optional
        Size (as integer, or float percentage of ``sample_size``) dedicated to including
        the latest experience items. By default, ``0``, i.e., no last item is included.
    seed : None, int, array_like of ints, SeedSequence, BitGenerator, Generator
        Seed for the :class:`numpy.random.Generator` used for sampling. By default,
        ``None``.

    Raises
    ------
    TypeError
        Raises if ``sample_size`` is a float (a percentage of the maximum length), but
        ``maxlen`` is ``None``, since it is impossible to compute the percentage of an
        unknown quantity.
    """

    def __init__(
        self,
        iterable: Iterable[ExpType] = (),
        maxlen: Optional[int] = None,
        sample_size: Union[int, float] = 1,
        include_latest: Union[int, float] = 0,
        seed: RngType = None,
    ) -> None:
        if isinstance(sample_size, float) and maxlen is None:
            raise TypeError(
                "Cannot compute the percentage of an unknown quantity (maxlen is None)."
            )
        super().__init__(iterable, maxlen=maxlen)
        self.sample_size = sample_size
        self.include_latest = include_latest
        self.reset(seed)

    def reset(self, seed: RngType = None) -> None:
        """Resets the seed of the :class:`numpy.random.Generator` used for sampling."""
        self.np_random = np.random.default_rng(seed)

    def sample(self) -> Iterator[ExpType]:
        """Samples the experience memory and yields the sampled items.

        Returns
        -------
        sample : iterator of ExpType
            An iterable sample is yielded.
        """
        L = len(self)
        n = self.sample_size
        last_n = self.include_latest
        if isinstance(n, float):
            n = int(self.maxlen * n)
        n = min(max(n, 0), L)
        if isinstance(last_n, float):
            last_n = int(n * last_n)
        last_n = min(max(last_n, 0), n)

        # get last n indices and the sampled indices from the remaining
        last = range(L - last_n, L)
        sampled = self.np_random.choice(range(L - last_n), n - last_n, False)
        yield from (self[i] for i in chain(sampled, last))
