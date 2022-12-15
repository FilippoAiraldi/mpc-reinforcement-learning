from itertools import count, repeat
from typing import Generator, Iterable, Optional, Tuple, Union

import numpy as np


def np_random(seed: Optional[int] = None) -> Tuple[np.random.Generator, int]:
    """Generates a random number generator from the seed and returns the Generator and
    seed.

    Full credit to OpenAI implementation at
    https://github.com/openai/gym/blob/master/gym/utils/seeding.py.


    Parameters
    ----------
    seed : int, optional
        The seed used to create the generator.

    Returns
    -------
    Tuple[Generator, int]
        The generator and resulting seed.

    Raises
    ------
    ValueError
        Seed must be a non-negative integer or omitted.
    """
    if seed is not None and not (isinstance(seed, int) and seed >= 0):
        raise ValueError(f"Seed must be a non-negative integer or omitted, not {seed}.")
    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    return rng, np_seed  # type: ignore


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
