from collections.abc import Iterator
from itertools import chain, cycle, repeat


def bool_cycle(frequency: int, starts_with: bool = False) -> Iterator[bool]:
    """Creates an infinite iterator which cycles via boolean values, where `True`
    appears with the given frequency.

    Parameters
    ----------
    frequency : int
        A positive int specifing the frequency at which `True` appears.
    starts_with : bool, optional
        Whether the first value should be `True` or `False`. By default `False`.

    Returns
    ------
    Iterator of bool
        An iterator with the given frequency of `True`.
    """
    iterator = cycle(chain(repeat(False, frequency - 1), (True,)))
    if starts_with:
        iterator = chain((True,), iterator)
    return iterator
