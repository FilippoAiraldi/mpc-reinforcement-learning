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
        update_biases_for_random_points: bool = True,
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
            NLP. If `None`, no structured point is generated.
        random_points : RandomStartPoints, optional
            Class containing info on how to generate random starting points for the NLP.
            Its `biases` field will be automatically updated with the last successful
            solution, unless disabled with `update_biases_for_random_points=False`). If
            `None`, no random point is generated.
        update_biases_for_random_points : bool, optional
            If `True`, the random points are biased around the bias-values updated with
            the latest successful MPC solution. If `False`, the biases in
            `random_points` are not updated.
        seed : None, int, array of ints, SeedSequence, BitGenerator, Generator
            Seed for the random number generator. By default, `None`.
        """
        self.store_always = warmstart == "last"
        self.structured_points = structured_points
        self.random_points = random_points
        self.update_biases_for_random_points = update_biases_for_random_points
        self.reset(seed)

    @property
    def n_points(self) -> int:
        """Returns the number of both random and structured starting points."""
        return (0 if self.random_points is None else self.random_points.multistarts) + (
            0 if self.structured_points is None else self.structured_points.multistarts
        )

    def reset(self, seed: RngType = None) -> None:
        """Resets the sampling RNG."""
        if self.random_points is not None:
            self.random_points.np_random = np.random.default_rng(seed)

    def generate(
        self, biases: Optional[dict[str, npt.ArrayLike]] = None
    ) -> Generator[dict[str, npt.ArrayLike], None, None]:
        """Generates initial conditions for the MPC's NLP.

        Parameters
        ----------
        biases : dict of (str, array_like), optional
            Optional biases that can be used to update the random points' original
            biases. If `None` or `update_biases_for_random_points=False`, the original
            biases are kept constant. These do not affect the generation of structure
            points in any way.

        Yields
        ------
        Generator of dict of (str, array_like)
            Yields the initial conditions for the MPC's NLP.
        """
        to_be_chained = []

        if self.structured_points is not None:
            to_be_chained.append(self.structured_points)

        if self.random_points is not None:
            if self.update_biases_for_random_points and biases is not None:
                self.random_points.biases.update(biases)
            to_be_chained.append(self.random_points)

        return chain.from_iterable(to_be_chained)

    def __repr__(self) -> str:
        nr = 0 if self.random_points is None else self.random_points.multistarts
        ns = 0 if self.structured_points is None else self.structured_points.multistarts
        return (
            f"{self.__class__.__name__}(store_always={self.store_always},"
            f"random_points={nr},structured_points={ns})"
        )

    def __str__(self) -> str:
        return self.__repr__()
