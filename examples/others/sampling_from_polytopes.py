r"""
.. _examples_other_sampling_from_polytopes:

Sampling from a convex polytopes
================================

This example demonstrates how to use the
:class:`mpcrl.util.geometry.ConvexPolytopeUniformSampler` to sample uniformly from the
interior and surface of a convex polytope in a N-dimensional space.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mpcrl.util.geometry import ConvexPolytopeUniformSampler

# %%
# Creating the polytope
# ---------------------
# Let's start by defining a set of vertices that define the polytope. At creation, it
# needs not be convex, but :class:`mpcrl.util.geometry.ConvexPolytopeUniformSampler`
# will assume its vertices form a convex polytope. This implies we can draw the vertices
# at random without any worry about the convexity of the resulting polytope.

np_random = np.random.default_rng(42)

for ndim in (2, 3, 7):
    nvertices = 10
    VERTICES = np_random.gumbel(size=(nvertices, ndim))

    # %%
    # Sampling from the polytope
    # --------------------------
    # Once the vertices have been defined, we can specify the number of samples to draw,
    # instantiate the samplers and call the
    # :meth:`mpcrl.util.geometry.ConvexPolytopeUniformSampler.sample` method

    n_samples = (100, 2)
    sampler = ConvexPolytopeUniformSampler(VERTICES, seed=np_random)
    interior_samples = sampler.sample_from_interior(n_samples)
    surface_samples = sampler.sample_from_surface(n_samples)

    # %%
    # We can check the validity of the samples by verifying that they lie within the
    # polytope or on its surface. Remember that a convex polytope can be defined as a
    # collection of inequalities, i.e., a point :math:`x` is inside the polytope if
    # :math:`A x + b \leq 0`, where :math:`A` is a matrix and :math:`b` is a vector, or lies
    # on its surface if :math:`a_i^\top x + b_i = 0` for one of facets :math:`i`.

    A = sampler._qhull.equations[:, :-1]
    b = sampler._qhull.equations[:, -1, None]

    # for the interior, for each sample ``j`` just check that the maximum value of the
    # inequalities ``A x_j + b`` is less than ``0``. Then, take the max across all
    # samples. So one call to max is enough.
    print(f"Checks in {ndim}-d")
    print("Interior samples validity:", (A @ interior_samples[..., None] + b).max())

    # for the surface, for each sample ``j`` compute the value of the facet equations
    # ``A x_j + b``. Then, take the absolute miniumum across the facets, as we suspect
    # the sample to lie on it. Finally, take the maximum across all samples.
    print(
        "Surface samples validity:",
        np.abs(A @ surface_samples[..., None] + b).min(2).max(),
    )

    # %%
    # Plotting the results
    # --------------------
    # Finally, we can plot the vertices and the samples to visualize the polytope and the
    # samples drawn from it. They should appear uniformly distributed within the polytope.
    # If not, try increasing the number of samples.

    if ndim == 2:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        x, y = sampler._triangulation.points.T
        ax.scatter(*VERTICES.T, c="C0", s=100, alpha=0.3)
        ax.triplot(x, y, sampler._triangulation.simplices, color="C0", alpha=0.3)
        ax.scatter(*interior_samples.T, c="C1", s=1)
        ax.scatter(*surface_samples.T, c="C2", s=1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    elif ndim == 3:
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(*VERTICES.T, c="C0", s=100)
        verts = [VERTICES[simplex] for simplex in sampler._triangulation.simplices]
        poly = Poly3DCollection(verts, alpha=0.1, edgecolor="C0")
        ax.add_collection3d(poly)
        ax.scatter(*interior_samples.T, c="C1", s=1)
        ax.scatter(*surface_samples.T, c="C2", s=1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_aspect("equal")

plt.show()
