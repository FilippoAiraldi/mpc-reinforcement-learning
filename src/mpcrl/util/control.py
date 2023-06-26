from typing import Callable, Optional, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_discrete_are

T = TypeVar("T")


def dlqr(
    A: npt.NDArray[np.floating],
    B: npt.NDArray[np.floating],
    Q: npt.NDArray[np.floating],
    R: npt.NDArray[np.floating],
    M: Optional[npt.NDArray[np.floating]] = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Get the discrete-time LQR for the given system. Stage costs are
    ```
        x'Qx + 2*x'Mu + u'Ru
    ```
    with `M = 0`, if not provided.

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
        Returns the optimal state feedback matrix `K` and the quadratic terminal
        cost-to-go matrix `P`.

    Note
    ----
    Inspired by
    https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/mpctools/util.py.
    """
    if M is not None:
        RinvMT = np.linalg.solve(R, M.T)
        Atilde = A - B.dot(RinvMT)
        Qtilde = Q - M.dot(RinvMT)
    else:
        Atilde = A
        Qtilde = Q
        M = np.zeros(B.shape)
    P = solve_discrete_are(Atilde, B, Qtilde, R)
    K = np.linalg.solve(B.T.dot(P).dot(B) + R, B.T.dot(P).dot(A) + M.T)
    return K, P


def rk4(f: Callable[[T], T], x0: T, dt: float = 1, M: int = 1) -> T:
    """Computes the Runge-Kutta 4 integration of the given function `f` with initial
    state x0.

    Parameters
    ----------
    f : Callable[[casadi or array], casadi or array]
        A function that takes a state as input and returns the derivative of the state,
        i.e., continuous-time dynamics.
    x0 : casadi or array
        The initial state. Must be compatible as argument to `f`.
    dt : float, optional
        The discretization timestep, by default `1`.
    M : int, optional
        How many RK4 steps to take in one `dt` interval, by default `1`.

    Returns
    -------
    new state : casadi or array
        The new state after `dt` time, according to the discretization.

    Note
    ----
    Inspired by
    https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/mpctools/util.py.
    """
    dt /= M
    x = x0
    for _ in range(M):
        k1 = f(x)
        k2 = f(x + k1 * dt / 2)  # type: ignore
        k3 = f(x + k2 * dt / 2)  # type: ignore
        k4 = f(x + k3 * dt)  # type: ignore
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6  # type: ignore
    return x
