from collections.abc import Sequence
from typing import Union

import numpy as np
from typing_extensions import TypeAlias

RngType: TypeAlias = Union[
    None,
    int,
    Sequence[int],
    np.random.SeedSequence,
    np.random.BitGenerator,
    np.random.Generator,
]


MAX_SEED = np.iinfo(np.uint32).max + 1


def mk_seed(rng: np.random.Generator) -> int:
    """Generates a random seed.

    Parameters
    ----------
    rng : np.random.Generator
        RNG generator

    Returns
    -------
    int
        A random integer in the range [0, 2**32)
    """
    return int(rng.integers(MAX_SEED))
