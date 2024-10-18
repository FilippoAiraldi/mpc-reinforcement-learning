"""A submodule with utility functions for geometry operations."""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull as _ConvexHull
from scipy.spatial import Delaunay as _Delaunay

from .seeding import RngType


class ConvexPolytopeUniformSampler:
    """Draws samples uniformly at random from a convex polytopic region (samples from
    its interior and surface are possible to obtain).

    Under the hood, the convex hull and triangulation of the given polytope are
    performed (see :class:`scipy.spatial.ConvexHull` and
    :class:`scipy.spatial.Delaunay`). To sample uniformly from the interior, for each
    sample a triangulation simplex is selected with probability proportional to its
    volume, and a point is drawn uniformly from that simplex via a Dirichlet
    distribution. Likewise, to sample from its surface, a facet is instead selected at
    random with probability proportional to its surface.

    Parameters
    ----------
    vertices : array-like of shape (n, d)
        The vertices of the polytope, where ``n`` is the number of vertices and ``d``
        is the dimension of the space.
    incremental : bool, optional
        Allows adding new points incrementally to the convex hull and triangulation.
        This takes up some additional resources.
    disable_triangulation : bool, optional
        Whether to disable the triangulation. This is useful when only surface samples
        are desired. By default, ``False``.
    disable_convex_hull : bool, optional
        Whether to disable the convex hull computation. This is useful when only
        interior samples are desired. By default, ``False``.
    seed : int, sequence of int, seed or :class:`numpy.random.Generator`, optional
        The seed or random number generator to use for sampling.
    """

    def __init__(
        self,
        vertices: npt.ArrayLike,
        incremental: bool = False,
        disable_triangulation: bool = False,
        disable_convex_hull: bool = False,
        seed: Optional[np.random.Generator] = None,
    ) -> None:
        self.seed(seed)
        self._tri_enabled = not disable_triangulation
        self._qhull_enabled = not disable_convex_hull
        if self._qhull_enabled:
            self._qhull = _ConvexHull(vertices, incremental=incremental)
            self._compute_facet_areas_ratios()
        if self._tri_enabled:
            self._triangulation = _Delaunay(vertices, incremental=incremental)
            self._compute_simplex_volumes_ratios()

    def seed(self, seed: RngType = None) -> npt.NDArray[np.floating]:
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
        if self._qhull_enabled:
            self._qhull.add_points(points, restart)
            self._compute_facet_areas_ratios()
        if self._tri_enabled:
            self._triangulation.add_points(points, restart)
            self._compute_simplex_volumes_ratios()

    def close(self) -> None:
        """Finishes the incremental processing."""
        if self._qhull_enabled:
            self._qhull.close()
        if self._tri_enabled:
            self._triangulation.close()

    def sample_from_interior(
        self, size: Union[int, tuple[int, ...]] = ()
    ) -> npt.NDArray[np.floating]:
        """Sample uniformly from the interior of the polytope defined by the given
        vertices.

        Parameters
        ----------
        size : int or tuple of ints, optional
            The size of the sample array to draw. By default, one sample is drawn.

        Returns
        -------
        array of shape (size1, size2, ..., d)
            The samples drawn from the polytope.

        Raises
        ------
        RuntimeError
            Raised if the triangulation is disabled.
        """
        if not self._tri_enabled:
            raise RuntimeError("Triangulation is disabled.")
        simplices = self._triangulation.points[self._triangulation.simplices]
        n_simplices, d_plus_1 = simplices.shape[:2]
        simplex_idx = self._np_random.choice(n_simplices, size, p=self._vol_ratios)
        selected_simplices = simplices[simplex_idx]
        weights = self._np_random.dirichlet(np.ones(d_plus_1), size)
        return np.matmul(weights[..., None, :], selected_simplices).squeeze(-2)

    def sample_from_surface(
        self, size: Union[int, tuple[int, ...]] = ()
    ) -> npt.NDArray[np.floating]:
        """Sample uniformly from the surface of the polytope defined by the given
        vertices.

        Parameters
        ----------
        size : int or tuple of ints, optional
            The size of the sample array to draw. By default, one sample is drawn.

        Returns
        -------
        array of shape (size1, size2, ..., d)
            The samples drawn from the polytope.

        Raises
        ------
        RuntimeError
            Raised if the convex hull is disabled.
        """
        if not self._qhull_enabled:
            raise RuntimeError("Convex hull is disabled.")
        facets = self._qhull.points[self._qhull.simplices]
        n_facets, d = facets.shape[:2]
        facet_idx = self._np_random.choice(n_facets, size, p=self._area_ratios)
        selected_facets = facets[facet_idx]
        weights = self._np_random.dirichlet(np.ones(d), size)
        return np.matmul(weights[..., None, :], selected_facets).squeeze(-2)

    def _compute_simplex_volumes_ratios(self) -> None:
        """Computes the ratios of the volumes of the simplices in the triangulation."""
        simplices = self._triangulation.points[self._triangulation.simplices]
        if simplices.shape[0] == 1:
            self._vol_ratios = np.ones(1, dtype=float)
        else:
            vol = np.abs(np.linalg.det(simplices[:, 1:] - simplices[:, :1]))
            self._vol_ratios = vol / vol.sum()

    def _compute_facet_areas_ratios(self) -> None:
        """Computes the ratios of the areas of the facets in the convex hull."""
        facets = self._qhull.points[self._qhull.simplices]
        if facets.shape[0] == 1:
            self._area_ratios = np.ones(1, dtype=float)
        else:
            failed = False
            diff = facets[:, 1:] - facets[:, :1]
            gram = diff @ diff.transpose(0, 2, 1)
            try:
                vol = np.sqrt(np.linalg.det(gram))
                area_ratios = vol / vol.sum()
            except np.linalg.LinAlgError:
                failed = True

            if failed or not np.isfinite(area_ratios).all():
                # alternative method to compute area by projecting and adjusting for the
                # projection angle - https://math.stackexchange.com/a/2098632
                # 1) project facets to d-1 simplex in the last axis
                simplices = facets[..., :-1]
                proj_vol = np.abs(np.linalg.det(simplices[:, 1:] - simplices[:, :1]))
                # 2) compute the projected normals
                normals = self._qhull.equations[:, :-1]  # already normalized
                proj_normals = normals[:, :-1]
                # 3) adjust the projected volume for the angle of the normals
                sine = np.linalg.norm(proj_normals, axis=1)
                vol = proj_vol / np.cos(np.arcsin(sine))
                area_ratios = vol / vol.sum()

            self._area_ratios = area_ratios
