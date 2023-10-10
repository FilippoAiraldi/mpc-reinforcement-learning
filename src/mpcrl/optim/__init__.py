__all__ = ["Adam", "GradientDescent", "GD", "NetwonMethod", "NM"]

from .adam import Adam
from .gradient_descent import GradientDescent
from .newton_method import NetwonMethod

GD = GradientDescent
NM = NetwonMethod
