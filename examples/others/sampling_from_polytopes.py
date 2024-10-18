r"""
.. _examples_other_sampling_from_polytopes:

Sampling from a convex polytopes
================================

This example demonstrates how to use the
:class:`mpcrl.util.geometry.ConvexPolytopeUniformSampler` to sample uniformly from a
convex polytope in a N-dimensional space.
"""

import matplotlib.pyplot as plt
import numpy as np

from mpcrl.util.geometry import ConvexPolytopeUniformSampler

# %%
# Creating the polytope
# ---------------------
# Let's start by defining a set of vertices that define the polytope. At creatiion, it
# needs not be convex, but the ConvexPolytopeUniformSampler will assume its vertices
# form a convex polytope. In other words, we can draw the vertices at random.

np_random = np.random.default_rng(69)
ndim = 3
nvertices = 10
VERTICES = np_random.normal(size=(nvertices, ndim))

# %%
# Sampling from the polytope
# --------------------------
# Once the vertices have been defined, we can specify the number of samples to draw,
# instantiate the sampler and call the
# :meth:`mpcrl.util.geometry.ConvexPolytopeUniformSampler.sample` method.

n_samples = (100, 10)
sampler = ConvexPolytopeUniformSampler(VERTICES, seed=np_random)
samples = sampler.sample(n_samples)

# %%
# Plotting the results
# --------------------
# Finally, we can plot the vertices and the samples to visualize the polytope and the
# samples drawn from it. They should appear uniformly distributed within the polytope.
# If not, try increasing the number of samples.

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(111, projection="3d")
ax.scatter(*samples.T, c="C1", s=1)
ax.scatter(*VERTICES.T, c="C0", s=100)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_aspect("equal")
plt.show()
