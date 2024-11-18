"""A submodule with utility functions for creating infinite iterators."""

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
    return cycle(chain(repeat(False, frequency - 1), (True,)))


def generate_seeds(
    starting_seed: Union[None, int, Iterable[int]], step: int = 1
) -> Generator[Optional[int], None, None]:
    """Given a starting seed, converts it into a generator of seeds, where each seed is
    sequenatial, or `None`.

    Parameters
    ----------
    starting_seed : int or iterable of ints or None
        The starting seed, which can be an int (produces sequential seeds), an iterable
        of ints (produces the same iter), or `None` (produces always `None`).
    step : int, optional
        The step to use when generating sequential seeds, by default `1`.

    Yields
    ------
    int or None
        A generator that either returns `None` or `int` (possibly, sequential or taken
        from the starting seed).
    """
    if starting_seed is None:
        yield from repeat(None)
    elif isinstance(starting_seed, int):
        yield from count(starting_seed, step)
    else:
        yield from starting_seed
