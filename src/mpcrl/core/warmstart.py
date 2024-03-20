from collections.abc import Generator
from itertools import chain
from typing import Literal, Optional

import numpy as np
from csnlp.multistart import RandomStartPoints, StructuredStartPoints
from numpy import typing as npt

from ..util.seeding import RngType


class WarmStartStrategy:
    """Class containing all the information to guide the warm start strategy for the
    MPC's NLP in order to speed up computations (by selecting appropriate initial
    conditions) and to support multiple initial conditions."""

    def __init__(
        self,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        structured_points: Optional[StructuredStartPoints] = None,
        random_points: Optional[RandomStartPoints] = None,
        seed: RngType = None,
    ) -> None:
        """Instantiates a warm start strategy for solving the MPC's NLP.

        Parameters
        ----------
        warmstart: "last" or "last-successful", optional
            The warmstart strategy for the MPC's NLP. If `last-successful`, the last
            successful solution is used to warm start the solver for the next iteration.
            If `last`, the last solution is used, regardless of success or failure.
        structured_points : StructuredStartPoint, optional
            Class containing info on how to generate structured starting points for the
            NLP. Note that its `multistarts` attribute will be overwritten with the
            correct value of requested structured initial conditions.
            If `None`, no structured point is generated.
        random_points : RandomStartPoints, optional
            Class containing info on how to generate random starting points for the NLP.
            Note that its `multistarts` attribute will be overwritten with the
            correct value of requested random initial conditions, and its `biases` will
            be updated with the last successful solution.
            If `None`, no random point is generated.
        seed : None, int, array of ints, SeedSequence, BitGenerator, Generator
            Seed for the random number generator. By default, `None`.
        """
        self.store_always = warmstart == "last"
        self.structured_points = structured_points
        self.random_points = random_points
        self.reset(seed)

    @property
    def can_generate(self) -> bool:
        """Returns whether the warm start strategy can generate initial conditions."""
        return self.structured_points is not None or self.random_points is not None

    def reset(self, seed: RngType = None) -> None:
        """Resets the sampling RNG."""
        self.np_random = np.random.default_rng(seed)

    def generate(
        self,
        n_struct: int,
        n_rand: int,
        biases: Optional[dict[str, npt.ArrayLike]] = None,
    ) -> Generator[dict[str, npt.ArrayLike], None, None]:
        """Generates initial conditions for the MPC's NLP.

        Parameters
        ----------
        n_struct : int
            The number of structured initial conditions to generate.
        n_rand : int
            The number of random initial conditions to generate.
        biases : dict of (str, array_like), optional
            Optional biases to add to the generated random points under the same name
            (not used for structured points).

        Yields
        ------
        Generator of dict of (str, array_like)
            Yields the initial conditions for the MPC's NLP.
        """
        to_be_chained = []

        if self.structured_points is not None:
            self.structured_points.multistarts = n_struct
            to_be_chained.append(self.structured_points)

        if self.random_points is not None:
            self.random_points.multistarts = n_rand
            self.random_points.np_random = self.np_random
            if biases is not None:
                self.random_points.biases.update(biases)
            to_be_chained.append(self.random_points)

        return chain.from_iterable(to_be_chained)

    def __repr__(self) -> str:
        nr = len(self.random_points) if self.random_points is not None else 0
        ns = len(self.structured_points) if self.structured_points is not None else 0
        return (
            f"{self.__class__.__name__}(store_always={self.store_always},"
            f"random_points={nr},structured_points={ns})"
        )

    def __str__(self) -> str:
        return self.__repr__()
