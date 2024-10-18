"""A submodule with utility functions for geometry operations."""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay

from .seeding import RngType


class ConvexPolytopeUniformSampler:
    """Draws samples uniformly at random from a convex polytopic region.

    Under the hood, the given polytope is triangulated using the Delaunay triangulation
    (:class:`scipy.spatial.Delaunay`). To sample uniformly, for each sample a simplex is
    selected with probability proportional to its volume, and a point is drawn uniformly
    from that simplex via a Dirichlet distribution.

    Parameters
    ----------
    vertices : array-like of shape (n, d)
        The vertices of the polytope, where ``n`` is the number of vertices and ``d``
        is the dimension of the space.
    incremental : bool, optional
        Allows adding new points incrementally to the Delaunay triangulation. This takes
        up some additional resources.
    seed : int, sequence of int, seed or :class:`numpy.random.Generator`, optional
        The seed or random number generator to use for sampling.
    """

    def __init__(
        self,
        vertices: npt.ArrayLike,
        incremental: bool = False,
        seed: Optional[np.random.Generator] = None,
    ) -> None:
        self.reset(seed)
        self._triangulation = Delaunay(vertices, incremental=incremental)
        self._compute_simplex_volumes_ratios()

    def reset(self, seed: RngType = None) -> npt.NDArray[np.floating]:
        """Resets the RNG engine.

        Parameters
        ----------
        seeed : int, sequence of int, seed or :class:`numpy.random.Generator`, optional
            The seed or random number generator to use for sampling.

        Returns
        -------
        array of shape (size1, size2, ..., d)
            The samples drawn from the simplex.

        Raises
        ------
        ValueError
            Raised if the vertices are not ``d+1`` in number.
        """
        self._np_random = np.random.default_rng(seed)

    def add_points(self, points: npt.ArrayLike, restart: bool = False) -> None:
        """Processes a set of additional new points. See also
        :meth:`scipy.spatial.Delaunay.add_points`.

        Parameters
        ----------
        vertices : array-like  of shape (n, d)
            New points to add, where ``n`` is the number of new points and ``d`` is the
            dimension of the space.
        restart : bool, optional
            Whether to restart processing from scratch, rather than adding points
            incrementally.
        """
        self._triangulation.add_points(points, restart)
        self._compute_simplex_volumes_ratios()

    def close(self) -> None:
        """Finishes the incremental processing."""
        self._triangulation.close()

    def sample(self, size: Union[int, tuple[int, ...]]) -> npt.NDArray[np.floating]:
        """Sample uniformly from the polytope defined by its vertices.

        Parameters
        ----------
        size : int or tuple of ints
            The size of the sample array to draw.

        Returns
        -------
        array of shape (size1, size2, ..., d)
            The samples drawn from the polytope.
        """
        simplices = self._triangulation.points[self._triangulation.simplices]
        n_simplices, d_plus_1 = simplices.shape[:2]
        simplex_idx = self._np_random.choice(n_simplices, size, p=self._ratios)
        selected_simplices = simplices[simplex_idx]
        weights = self._np_random.dirichlet(np.ones(d_plus_1), size)
        return np.matmul(weights[..., None, :], selected_simplices).squeeze(-2)

    def _compute_simplex_volumes_ratios(self) -> None:
        """Computes the ratios of the volumes of the simplices in the triangulation."""
        simplices = self._triangulation.points[self._triangulation.simplices]
        if simplices.shape[0] == 1:
            self._ratios = np.ones(1, dtype=float)
        else:
            ratios = np.abs(np.linalg.det(simplices[:, 1:] - simplices[:, :1]))
            self._ratios = ratios / ratios.sum()
