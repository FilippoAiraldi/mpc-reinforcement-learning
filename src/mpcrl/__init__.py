__version__ = "1.2.0rc3"

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
from .wrappers import agents as wrappers_agents
from .wrappers import envs as wrappers_envs
