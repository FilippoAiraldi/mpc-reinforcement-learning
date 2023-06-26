from itertools import chain, cycle, repeat
from typing import Iterator


def bool_cycle(frequency: int) -> Iterator[bool]:
    """Creates an iterator which cycles via boolean values, where `True` appears with
    the given frequency.

    Parameters
    ----------
    frequency : int
        A positive int specifing the frequency at which `True` appears.

    Returns
    ------
    Iterator[bool]
        An iterator with the given frequency of `True`.
    """
    return cycle(chain(repeat(False, frequency - 1), (True,)))
