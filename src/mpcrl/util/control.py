"""A collection of basic utility functions for control applications. In particular, it
contains a function for solving the discrete-time LQR problem and a function for
integrating a continuous-time dynamics using the Runge-Kutta 4 method.

Heavy inspiration was drawn from `MPCtools <https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/mpctools/util.py>`_.
"""

from typing import Callable, Optional
from typing import TypeVar as _Typevar

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_discrete_are as _solve_discrete_are

T = _Typevar("T")


def dlqr(
    A: npt.NDArray[np.floating],
    B: npt.NDArray[np.floating],
    Q: npt.NDArray[np.floating],
    R: npt.NDArray[np.floating],
    M: Optional[npt.NDArray[np.floating]] = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Computes the solution to the discrete-time LQR problem.

    The LQR problem is to solve the following optimization problem

    .. math::
        \min_{u} \sum_{t=0}^{\infty} x_t^\top Q x_t + u_t^\top R u_t + 2 x_t^\top M u_t

    for the linear time-invariant discrete-time system

    .. math:: x_{t+1} = A x_t + B u_t.

    The (famous) solution takes the form of a state feedback law

    .. math:: u_t = -K x_t

    with a quadratic cost-to-go function

    .. math:: V(x_t) = x_t^\top P x_t.

    The function returns the optimal state feedback matrix :math:`K` and the quadratic
    terminal cost-to-go matrix :math:`P`. If not provided, ``M`` is assumed to be zero.

    Parameters
    ----------
    A : array
        State matrix.
    B : array
        Control input matrix.
    Q : array
        State weighting matrix.
    R : array
        Control input weighting matrix.
    M : array, optional
        Mixed state-input weighting matrix, by default None.

    Returns
    -------
    tuple of two arrays
        Returns the optimal state feedback matrix :math:`K` and the quadratic terminal
        cost-to-go matrix :math:`P`.
    """
    if M is not None:
        RinvMT = np.linalg.solve(R, M.T)
        Atilde = A - B.dot(RinvMT)
        Qtilde = Q - M.dot(RinvMT)
    else:
        Atilde = A
        Qtilde = Q
        M = np.zeros(B.shape)
    P = _solve_discrete_are(Atilde, B, Qtilde, R)
    K = np.linalg.solve(B.T.dot(P).dot(B) + R, B.T.dot(P).dot(A) + M.T)
    return K, P


def rk4(f: Callable[[T], T], x0: T, dt: float = 1, M: int = 1) -> T:
    r"""Computes the Runge-Kutta 4 integration of the given function ``f`` with initial
    state ``x0``.

    In mathematical terms, given a continuous-time dynamics function
    :math:`\dot{x} = f(x)`, this method returns a discretized version of the dynamics
    :math:`x_{k+1} = f_d(x_k)` using the Runge-Kutta 4 method.

    Parameters
    ----------
    f : Callable[[casadi or array], casadi or array]
        A function that takes a state as input and returns the derivative of the state,
        i.e., continuous-time dynamics.
    x0 : casadi or array
        The initial state. Must be compatible as an argument to ``f``.
    dt : float, optional
        The discretization timestep, by default ``1``.
    M : int, optional
        How many RK4 steps to take in one ``dt`` interval, by default ``1``.

    Returns
    -------
    new state : casadi or array
        The new state after ``dt`` time, according to the discretization.
    """
    dt /= M
    x = x0
    for _ in range(M):
        k1 = f(x)
        k2 = f(x + k1 * dt / 2)
        k3 = f(x + k2 * dt / 2)
        k4 = f(x + k3 * dt)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    return x
