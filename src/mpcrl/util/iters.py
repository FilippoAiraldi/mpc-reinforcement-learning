"""A submodule with utility functions for creating infinite iterators."""

import itertools
from collections.abc import Iterator as _Iterator


def bool_cycle(frequency: int, starts_with: bool = False) -> _Iterator[bool]:
    """Creates an infinite iterator which cycles via boolean values, where ``True``
    appears with the given frequency ``frequency``.

    Parameters
    ----------
    frequency : int
        A positive int specifing the frequency at which ``True`` appears.
    starts_with : bool, optional
        Whether the first value should be ``True`` or ``False``. By default ``False``.

    Returns
    -------
    Iterator of bool
        An iterator with the given frequency of ``True``.
    """
    if frequency <= 1:
        return itertools.repeat(True)

    iterator = itertools.cycle(
        itertools.chain(itertools.repeat(False, frequency - 1), (True,))
    )
    if starts_with:
        iterator = itertools.chain((True,), iterator)
    return iterator
