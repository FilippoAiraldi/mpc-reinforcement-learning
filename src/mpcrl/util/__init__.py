r"""A module with utility functions and classes around and for creating control
problems, agents and their components, and typing.

Overview
========

It contains the following submodules:

- :mod:`mpcrl.util.control`: a collection of basic utility functions for control
  applications
- :mod:`mpcrl.util.iters`: a submodule with utility functions for creating infinite
  iterators
- :mod:`mpcrl.util.math`: a collection of functions for mathematical operations and
  utilities
- :mod:`mpcrl.util.named`: a submodule with an utility class for assigning unique names
  to each instance of a subclass
- :mod:`mpcrl.util.seeding`: a submodule with utility functions and typing for seeding
  random number generators.

Submodules
==========

.. autosummary::
   :toctree: generated
   :template: module.rst

   control
   iters
   math
   named
   seeding
"""

__all__ = ["control", "iters", "math", "named", "seeding"]

from . import control, iters, math, named, seeding
