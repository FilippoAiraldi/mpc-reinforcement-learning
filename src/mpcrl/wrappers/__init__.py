"""A module with wrappers for agents and environments.

In :mod:`gymnasium`, wrappers are a common way to modify the behavior of environments to
specific needs; see :class:`gymnasium.Wrapper`. In this module, we provide new wrappers
for environments, as well as define our own :class:`mpcrl.wrappers_agents.Wrapper` class
to wrap also (learning and non-) agents.

Submodules
==========

.. autosummary::
   :toctree: generated
   :template: module.rst
   :caption: Submodules

   agents
   envs
"""

__all__ = ["wrappers_agents", "wrappers_envs"]

from . import agents as wrappers_agents
from . import envs as wrappers_envs
