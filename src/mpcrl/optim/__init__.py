__all__ = ["Adam", "GradientDescent", "GD", "NetwonMethod", "NM", "RMSprop"]

from .adam import Adam
from .gradient_descent import GradientDescent
from .newton_method import NetwonMethod
from .rmsprop import RMSprop

GD = GradientDescent
NM = NetwonMethod
