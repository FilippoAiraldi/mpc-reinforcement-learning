"""A submodule with utility functions and typing for seeding random number
generators. In particular, these have been conceived with the goal of being as
compatible with the :mod:`gymnasium` framework as possible."""

import sys
from collections.abc import Sequence as _Sequence
from typing import Union

import numpy as np

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

RngType: TypeAlias = Union[
    None,
    int,
    _Sequence[int],
    np.random.SeedSequence,
    np.random.BitGenerator,
    np.random.Generator,
]


MAX_SEED = np.iinfo(np.uint32).max + 1


def mk_seed(rng: np.random.Generator) -> int:
    """Generates a random seed compatible with :func:`gymnasium.Env.reset`.

    Parameters
    ----------
    rng : :class:`numpy.random.Generator`
        RNG generator.

    Returns
    -------
    seed
        A random integer in the range [0, 2**32).
    """
    return int(rng.integers(MAX_SEED))
