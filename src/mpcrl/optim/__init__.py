"""This submodule contains the optimizers that are used to update the parameters of the
agent's MPC scheme. These are mainly gradient-based, i.e., they exploit Jacobian (and
possibly Hessian) information to update the parameters. However, as we will see below,
the submodule also allows for gradient-free optimization techniques (to be combined with
:class:`mpcrl.GlobOptLearningAgent`). See also :ref:`user_guide_optim`."""

__all__ = [
    "Adam",
    "GradientBasedOptimizer",
    "GradientDescent",
    "GradientFreeOptimizer",
    "GD",
    "NewtonMethod",
    "NM",
    "RMSprop",
]

from .adam import Adam
from .gradient_based_optimizer import GradientBasedOptimizer
from .gradient_descent import GradientDescent
from .gradient_free_optimizer import GradientFreeOptimizer
from .newton_method import NewtonMethod
from .rmsprop import RMSprop

GD = GradientDescent
NM = NewtonMethod
