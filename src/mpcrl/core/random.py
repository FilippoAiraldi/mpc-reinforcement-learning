from typing import Optional, Tuple

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
