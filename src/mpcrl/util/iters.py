from itertools import chain, count, cycle, repeat
from typing import Generator, Iterable, Iterator, Optional, Union


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


def generate_seeds(
    starting_seed: Union[None, int, Iterable[int]]
) -> Generator[Optional[int], None, None]:
    """Given a starting seed, converts it into a generator of seeds, where each seed is
    sequenatial, or `None`.

    Parameters
    ----------
    starting_seed : int or iterable of ints or None
        The starting seed, which can be an int (produces sequential seeds), an iterable
        of ints (produces the same iter), or None (produces always None)

    Yields
    ------
    int or None
        A generator that either returns None or int (possibly, sequential or taken from
        the starting seed).
    """
    if starting_seed is None:
        yield from repeat(None)
    elif isinstance(starting_seed, int):
        yield from count(starting_seed)
    else:
        yield from starting_seed
