"""More often than not, the agent is given a nonlinear MPC problem to solve at each
time step. It is well-known that NLP problems are proned to convergence to local minima,
and a simple yet effective way to counter this is to solve the same problem multiple
times with different initial conditions. This is known as a warmstart strategy.

:class:`WarmStartStrategy` allows the user to specify a warmstart strategy for the NLP
solver of the MPC. In particular, the user can select which past solution to use as the
initial guess for the next iteration, and whether to generate structured or random
guesses around that solution to further explore the solution space and possibly converge
to a better solution. Moreover, the user can choose how this strategy should keep up
with new solutions.

This class makes heavy use of :mod:`csnlp.multistart`, so be sure to check the
documentation of that module for more basic information. See also
:ref:`user_guide_warmstarting`."""

from collections.abc import Generator
from itertools import chain
from typing import Literal, Optional

import numpy as np
from csnlp.multistart import RandomStartPoints, StructuredStartPoints
from numpy import typing as npt

from ..util.seeding import RngType


def _merge_init_conditions_dicts(
    init_cond: dict[str, npt.ArrayLike],
    prev_sol: dict[str, npt.ArrayLike],
) -> dict[str, npt.ArrayLike]:
    """Internal utility to merge two dictionaries of initial conditions, where the
    former comes from multistarting, and the latter from the previous solution."""
    out = prev_sol.copy()
    out.update(init_cond)
    return out


class WarmStartStrategy:
    """Class containing all the information to guide the warmstart strategy for the
    MPC's NLP in order to speed up computations (by selecting appropriate initial
    conditions) and to support multiple initial conditions.

    Parameters
    ----------
    warmstart: "last" or "last-successful", optional
        The warmstart strategy for the MPC's NLP solver. If ``"last-successful"``, the
        last **successful** solution is automatically used to warmstart the solver for
        the next iteration. If ``"last"``, the last solution is automatically used,
        regardless of success or failure.
    structured_points : :class:`csnlp.multistart.StructuredStartPoint`, optional
        Class containing info on how to generate structured starting points for the
        NLP solver. If ``None``, no structured point is generated.
    random_points : :class:`csnlp.multistart.RandomStartPoints`, optional
        Class containing info on how to generate random starting points for the NLP
        solver. Its :attr:`csnlp.multistart.RandomStartPoints.biases` field will be
        automatically updated with the last available solution (based on ``warmstart``),
        unless disabled with ``update_biases_for_random_points=False``). If ``None``,
        no random point is generated.
    update_biases_for_random_points : bool, optional
        If ``True``, the random points are biased around the values from the last
        available solution to the MPC optimization (based on ``warmstart``). If
        ``False``, the biases in ``random_points`` are not updated.
    seed : None, int, array of ints, SeedSequence, BitGenerator, Generator
        Seed for the random number generator. By default, ``None``.
    """

    def __init__(
        self,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        structured_points: Optional[StructuredStartPoints] = None,
        random_points: Optional[RandomStartPoints] = None,
        update_biases_for_random_points: bool = True,
        seed: RngType = None,
    ) -> None:
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
        """Resets the :class:`numpy.random.Generator` for sampling random points, if any
        were specified."""
        if self.random_points is not None:
            self.random_points.np_random = np.random.default_rng(seed)

    def generate(
        self, previous_sol: Optional[dict[str, npt.ArrayLike]] = None
    ) -> Generator[dict[str, npt.ArrayLike], None, None]:
        """Generates some initial conditions/guesses for the primal optimization
        variables of the MPC's NLP problem.

        Parameters
        ----------
        previous_sol : dict of (str, array_like), optional
            Optional dict that contains the previous solution's values, if available. If
            passed, it is used

            - to update the random points' original, unless
              ``update_biases_for_random_points=False``, at which point the original
              biases are kept constant (this does not affect the generation of structure
              points in any way)
            - to fill in the rest of the initial conditions that are not included in the
              multistart strategy (neither structured nor random).

        Yields
        ------
        Generator of dict of (str, array_like)
            Yields the initial conditions for the MPC's NLP.
        """
        to_be_chained = []
        given_prev_sol = previous_sol is not None

        if self.structured_points is not None:
            to_be_chained.append(self.structured_points)

        if self.random_points is not None:
            if self.update_biases_for_random_points and given_prev_sol:
                self.random_points.biases.update(previous_sol)
            to_be_chained.append(self.random_points)

        generator = chain.from_iterable(to_be_chained)
        if given_prev_sol:
            generator = map(
                lambda ic: _merge_init_conditions_dicts(ic, previous_sol), generator
            )
        return generator

    def __repr__(self) -> str:
        nr = 0 if self.random_points is None else self.random_points.multistarts
        ns = 0 if self.structured_points is None else self.structured_points.multistarts
        return (
            f"{self.__class__.__name__}(store_always={self.store_always},"
            f"random_points={nr},structured_points={ns})"
        )

    def __str__(self) -> str:
        return self.__repr__()
