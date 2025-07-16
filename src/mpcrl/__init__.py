r"""**M**\odel **P**\redictive **C**\ontrol-based **R**\einforcement **L**\earning
(**mpcrl**, for short) is a library for training model-based Reinforcement Learning (RL)
:cite:`sutton_reinforcement_2018` agents with Model Predictive Control (MPC) as function
approximation :cite:`rawlings_model_2017`.

==================== =======================================================================
**Documentation**        https://mpc-reinforcement-learning.readthedocs.io/en/stable/

**Download**             https://pypi.python.org/pypi/mpcrl/

**Source code**          https://github.com/FilippoAiraldi/mpc-reinforcement-learning/

**Report issues**        https://github.com/FilippoAiraldi/mpc-reinforcement-learning/issues
==================== =======================================================================
"""

import sys
import warnings

if sys.version_info < (3, 10):
    warnings.warn(
        "Support for Python 3.9 is deprecated and will not be tested in future "
        "releases. Consider upgrading to Python 3.10 or higher for better support.",
        DeprecationWarning,
    )

__version__ = "1.3.3"

__all__ = [
    "Agent",
    "ExperienceReplay",
    "GlobOptLearningAgent",
    "LearnableParameter",
    "LearnableParametersDict",
    "LearningAgent",
    "LstdDpgAgent",
    "LstdQLearningAgent",
    "MpcSolverError",
    "MpcSolverWarning",
    "RlLearningAgent",
    "UpdateError",
    "UpdateStrategy",
    "UpdateWarning",
    "WarmStartStrategy",
    "exploration",
    "optim",
    "schedulers",
    "wrappers_agents",
    "wrappers_envs",
]

from . import optim
from .agents.common.agent import Agent
from .agents.common.globopt_learning_agent import GlobOptLearningAgent
from .agents.common.learning_agent import LearningAgent
from .agents.common.rl_learning_agent import RlLearningAgent
from .agents.lstd_dpg import LstdDpgAgent
from .agents.lstd_q_learning import LstdQLearningAgent
from .core import exploration, schedulers
from .core.errors import MpcSolverError, MpcSolverWarning, UpdateError, UpdateWarning
from .core.experience import ExperienceReplay
from .core.parameters import LearnableParameter, LearnableParametersDict
from .core.update import UpdateStrategy
from .core.warmstart import WarmStartStrategy
from .wrappers import wrappers_agents, wrappers_envs
