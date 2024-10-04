"""A collection of basic utility functions for control applications. In particular, it
contains a function for solving the discrete-time LQR problem and a function for
integrating a continuous-time dynamics using the Runge-Kutta 4 method.

Heavy inspiration was drawn from `MPCtools <https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/mpctools/util.py>`_.
"""

from collections.abc import Iterable as _Iterable
from typing import Callable, Optional
from typing import TypeVar as _TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_continuous_are as _solve_continuous_are
from scipy.linalg import solve_discrete_are as _solve_discrete_are

T = _TypeVar("T")
SymType = _TypeVar("SymType", cs.SX, cs.MX)


def lqr(
    A: npt.NDArray[np.floating],
    B: npt.NDArray[np.floating],
    Q: npt.NDArray[np.floating],
    R: npt.NDArray[np.floating],
    M: Optional[npt.NDArray[np.floating]] = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Computes the solution to the continuous-time LQR problem.

    The LQR problem is to solve the following optimization problem

    .. math::
        \min_{u} \int_{0}^{\infty} x^\top Q x + u^\top R u + 2 x^\top M u

    for the linear time-invariant continuous-time system

    .. math:: \dot{x} = A x + B u.

    The (famous) solution takes the form of a state feedback law

    .. math:: u = -K x

    with a quadratic cost-to-go function

    .. math:: V(x) = x^\top P x.

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
        Mixed state-input weighting matrix, by default ``None``.

    Returns
    -------
    tuple of two arrays
        Returns the optimal state feedback matrix :math:`K` and the quadratic terminal
        cost-to-go matrix :math:`P`.
    """
    P = _solve_continuous_are(A, B, Q, R, s=M)
    rhs = B.T.dot(P) if M is None else B.T.dot(P) + M.T
    K = np.linalg.solve(R, rhs)
    return K, P


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
        Mixed state-input weighting matrix, by default ``None``.

    Returns
    -------
    tuple of two arrays
        Returns the optimal state feedback matrix :math:`K` and the quadratic terminal
        cost-to-go matrix :math:`P`.
    """
    P = _solve_discrete_are(A, B, Q, R, s=M)
    rhs = B.T.dot(P).dot(A) if M is None else B.T.dot(P).dot(A) + M.T
    K = np.linalg.solve(B.T.dot(P).dot(B) + R, rhs)
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


def lie_derivative(
    ex: SymType, arg: SymType, field: SymType, order: int = 1
) -> SymType:
    """Computes the Lie derivative of the expression ``ex`` with respect to the argument
    ``arg`` along the field ``field``.

    Parameters
    ----------
    ex : casadi SX or MX
        Expression to compute the Lie derivative of.
    arg : casadi SX or MX
        Argument with respect to which to compute the Lie derivative.
    field : casadi SX or MX
        Field along which to compute the Lie derivative.
    order : int, optional
        Order (>= 1) of the Lie derivative, by default ``1``.

    Returns
    -------
    casadi SX or MX
        The Lie derivative of the expression ``ex`` with respect to the argument ``arg``
        along the field ``field``.
    """
    deriv = cs.dot(cs.gradient(ex, arg), field)
    if order <= 1:
        return deriv
    return lie_derivative(deriv, arg, field, order - 1)


def cbf(
    h: Callable[[SymType], SymType],
    x: SymType,
    u: SymType,
    dynamics: Callable[[SymType, SymType], SymType],
    alphas: _Iterable[Callable[[SymType], SymType]],
) -> tuple[cs.Function, int]:
    r"""Continuous-time Control Barrier Function (CBF) for the given constraint ``h``
    and system with dynamics ``dynamics``. This method constructs a CBF for the
    constraint :math:`h(x) \geq 0` using the given system's dynamics
    :math:`\dot{x} = f(x, u)`. Here, :math:`\dot{x}` is the time derivative of the state
    after applying control input :math:`u`, and :math:`f` is the dynamics function
    ``dynamics``.

    The method can also compute a High-Order (HO) CBF by passing more than one class
    :math:`\mathcal{K}` functions ``alphas``.

    As per [1]_, the HO-CBF :math:`\phi_m` of degree :math:`m` is recursively found as

    .. math::
        \phi_m(x) = \dot{\phi}_{m-1}(x) + \alpha_m(\phi_{m-1}(x))

    and should be imposed as the constraint :math:`\phi_m(x, t) \geq 0`.

    Parameters
    ----------
    h : callable
        The constraint function for which to build the CBF. It must be of the signature
        :math:`x \rightarrow h(x)`.
    x : casadi SX or MX
        The state variable :math:`x`.
    u : casadi SX or MX
        The control input variable :math:`u`.
    dynamics : callable
        The dynamics function :math:`f` with signature :math:`x,u \rightarrow f(x, u)`.
    alphas : iterable of callables
        An iterable of class :math:`\mathcal{K}` functions :math:`\alpha_m` for
        the HO-DCBF. The length of the iterable determines the degree of the HO-CBF is
        computed.

    Returns
    -------
    casadi Function and int
        Returns the HO-CBF function :math:`\phi_m` as a function with signature
        :math:`x,u \rightarrow \phi_m(x, u)`, as well as the degree ``m`` of the
        HO-CBF.

    References
    ----------
    .. [1] Xiao, W. and Belta, C., 2021. High-order control barrier functions. IEEE
       Transactions on Automatic Control, 67(7), pp. 3655-3662.

    Examples
    --------
    >>> import casadi as cs
    >>> A = cs.SX.sym("A", 2, 2)
    >>> B = cs.SX.sym("B", 2, 1)
    >>> x = cs.SX.sym("x", A.shape[0], 1)
    >>> u = cs.SX.sym("u", B.shape[1], 1)
    >>> dynamics = lambda x, u: A @ x + B @ u
    >>> M = cs.SX.sym("M")
    >>> c = cs.SX.sym("c")
    >>> gamma = cs.SX.sym("gamma")
    >>> alphas = [lambda z: gamma * z]
    >>> h = lambda x: M - c * x[0]  # >= 0
    >>> cbf, _ = cbf(h, x, u, dynamics, alphas)
    >>> print(cbf(x, u))
    """
    opts = {"cse": True, "allow_free": True}
    x_dot = dynamics(x, u)
    h_eval = phi_eval = h(x)
    phi = cs.Function("phi_0", (x, u), (h_eval,), ("x", "u"), ("phi_0",), opts)
    for degree, alpha in enumerate(alphas, start=1):
        name = f"phi_{degree}"
        phi = cs.Function(
            name,
            (x, u),
            (lie_derivative(phi_eval, x, x_dot) + alpha(phi_eval),),
            ("x", "u"),
            (name,),
            opts,
        )
        phi_eval = phi(x, u)
        # cs.depends_on(phi_eval, u)
        # phi.which_depends("u", [name], 2, True)[0]
    return phi, degree


def dcbf(
    h: Callable[[SymType], SymType],
    x: SymType,
    u: SymType,
    dynamics: Callable[[SymType, SymType], SymType],
    alphas: _Iterable[Callable[[SymType], SymType]],
) -> tuple[cs.Function, int]:
    r"""Discrete-time Control Barrier Function (DCBF) for the given constraint ``h`` and
    system with dynamics ``dynamics``. This method constructs a DCBF for the constraint
    :math:`h(x) \geq 0` using the given system's dynamics :math:`x_{+} = f(x, u)`. Here,
    :math:`x_{+}` is the next state after applying control input :math:`u`, and
    :math:`f` is the dynamics function ``dynamics``.

    The method can also compute a High-Order (HO) DCBF by passing more than one class
    :math:`\mathcal{K}` functions ``alphas``.

    As per [1]_, the HO-DCBF :math:`\phi_m` of degree :math:`m` is recursively found as

    .. math::
        \phi_m(x_k) = \phi_{m-1}(x_{k+1}) - \phi_{m-1}(x_k) + \alpha_m(\phi_{m-1}(x_k))

    and should be imposed as the constraint :math:`\phi_m(x_k) \geq 0`.

    Parameters
    ----------
    h : callable
        The constraint function for which to build the DCBF. It must be of the signature
        :math:`x \rightarrow h(x)`.
    x : casadi SX or MX
        The state variable :math:`x`.
    u : casadi SX or MX
        The control input variable :math:`u`.
    dynamics : callable
        The dynamics function :math:`f` with signature :math:`x,u \rightarrow f(x, u)`.
    alphas : iterable of callables
        An iterable of class :math:`\mathcal{K}` functions :math:`\alpha_m` for
        the HO-DCBF. The length of the iterable determines the degree of the HO-DCBF is
        computed.

    Returns
    -------
    casadi Function and int
        Returns the HO-DCBF function :math:`\phi_m` as a function with signature
        :math:`x,u \rightarrow \phi_m(x, u)`, as well as the degree ``m`` of the
        HO-DCBF.

    References
    ----------
    .. [1] Xiong, Y. and Zhai, D.H. and Tavakoli, M. and Xia, Y., 2022. Discrete-time
       control barrier function: High-order case and adaptive case. IEEE Transactions on
       Cybernetics, 53(5), pp. 3231-3239.

    Examples
    --------
    >>> import casadi as cs
    >>> A = cs.SX.sym("A", 2, 2)
    >>> B = cs.SX.sym("B", 2, 1)
    >>> x = cs.SX.sym("x", A.shape[0], 1)
    >>> u = cs.SX.sym("u", B.shape[1], 1)
    >>> dynamics = lambda x, u: A @ x + B @ u
    >>> M = cs.SX.sym("M")
    >>> c = cs.SX.sym("c")
    >>> gamma = cs.SX.sym("gamma")
    >>> alphas = [lambda z: gamma * z]
    >>> h = lambda x: M - c * x[0]  # >= 0
    >>> cbf, _ = dcbf(h, x, u, dynamics, alphas)
    >>> print(cbf(x, u))
    """
    opts = {"cse": True, "allow_free": True}
    x_next = dynamics(x, u)
    h_eval = phi_eval = h(x)
    phi = cs.Function("phi_0", (x, u), (h_eval,), ("x", "u"), ("phi_0",), opts)
    for degree, alpha in enumerate(alphas, start=1):
        name = f"phi_{degree}"
        phi = cs.Function(
            name,
            (x, u),
            (phi(x_next, u) - phi_eval + alpha(phi_eval),),
            ("x", "u"),
            (name,),
            opts,
        )
        phi_eval = phi(x, u)
        # cs.depends_on(phi_eval, u)
        # phi.which_depends("u", [name], 2, True)[0]
    return phi, degree
