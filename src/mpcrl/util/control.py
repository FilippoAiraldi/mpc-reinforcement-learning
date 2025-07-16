"""A collection of basic utility functions for control applications. In particular, it
contains functions for solving the LQR problems in continuous- and discrete-time,
discretization methods such as Runge-Kutta 4, and functions to build Control Barrier
Functions.

Some inspiration was drawn from `MPCtools <https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/mpctools/util.py>`_.
"""

from collections.abc import Iterable as _Iterable
from typing import Callable, Optional

import casadi as cs
import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_continuous_are as _solve_continuous_are
from scipy.linalg import solve_discrete_are as _solve_discrete_are

from .math import SymOrNumType, SymType
from .math import dual_norm as _dual_norm
from .math import lie_derivative as _lie_derivative


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
    P = _solve_continuous_are(A, B, Q, R, None, M)
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
    P = _solve_discrete_are(A, B, Q, R, None, M)
    rhs = B.T.dot(P).dot(A) if M is None else B.T.dot(P).dot(A) + M.T
    K = np.linalg.solve(B.T.dot(P).dot(B) + R, rhs)
    return K, P


def rk4(
    f: Callable[[SymOrNumType], SymOrNumType],
    x0: SymOrNumType,
    dt: float = 1,
    M: int = 1,
) -> SymOrNumType:
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


def cbf(
    h: Callable[[SymType], SymType],
    x: SymType,
    u: SymType,
    dynamics: Callable[[SymType, SymType], SymType],
    alphas: _Iterable[Callable[[SymType], SymType]],
) -> SymType:
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

    and should be imposed as the constraint :math:`\phi_m(x) \geq 0`.

    Parameters
    ----------
    h : callable
        The constraint function for which to build the CBF. It must be of the signature
        :math:`x \rightarrow h(x)`.
    x : casadi SX or MX
        The state vector variable :math:`x`.
    u : casadi SX or MX
        The control input vector variable :math:`u`.
    dynamics : callable
        The dynamics function :math:`f` with signature :math:`x,u \rightarrow f(x, u)`.
    alphas : iterable of callables
        An iterable of class :math:`\mathcal{K}` functions :math:`\alpha_m` for
        the HO-CBF. The length of the iterable determines the degree of the HO-CBF.

    Returns
    -------
    casadi SX or MX
        Returns the HO-CBF function :math:`\phi_m` as a symbolic variable that is
        function of the provided ``x`` and ``u``.

    References
    ----------
    .. [1] Wei Xiao and Calin Belta. High-order control barrier functions. *IEEE
           Transactions on Automatic Control*, 67(7), 3655-3662, 2021.

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
    >>> cbf = cbf(h, x, u, dynamics, alphas)
    >>> print(cbf)
    """
    x_dot = dynamics(x, u)
    phi = h(x)
    for alpha in alphas:
        phi = _lie_derivative(phi, x, x_dot) + alpha(phi)
    # cs.depends_on(phi, u)
    # phi.which_depends("u", [name], 2, True)[0]
    return phi


def dcbf(
    h: Callable[[SymType], SymType],
    x: SymType,
    u: SymType,
    dynamics: Callable[[SymType, SymType], SymType],
    alphas: _Iterable[Callable[[SymType], SymType]],
) -> cs.Function:
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
        The state vector variable :math:`x`.
    u : casadi SX or MX
        The control input vector variable :math:`u`.
    dynamics : callable
        The dynamics function :math:`f` with signature :math:`x,u \rightarrow f(x, u)`.
    alphas : iterable of callables
        An iterable of class :math:`\mathcal{K}` functions :math:`\alpha_m` for
        the HO-DCBF. The length of the iterable determines the degree of the HO-DCBF.

    Returns
    -------
    casadi SX or MX
        Returns the HO-DCBF function :math:`\phi_m` as a symbolic variable that is
        function of the provided ``x`` and ``u``.

    References
    ----------
    .. [1] Yuhang Xiong, Di-Hua Zhai, Mahdi Tavakoli, Yuanqing Xia. Discrete-time
       control barrier function: High-order case and adaptive case. *IEEE Transactions
       on Cybernetics*, 53(5), 3231-3239, 2022.

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
    >>> cbf = dcbf(h, x, u, dynamics, alphas)
    >>> print(cbf)
    """
    x_next = dynamics(x, u)
    phi = h(x)
    for alpha in alphas:
        phi_next = cs.substitute(phi, x, x_next)
        phi = phi_next - phi + alpha(phi)
    return phi


def iccbf(
    h: Callable[[SymType], SymType],
    x: SymType,
    u: SymType,
    dynamics_f_and_g: Callable[[SymType], tuple[SymType, SymType]],
    alphas: _Iterable[Callable[[SymType], SymType]],
    norm: float,
    bound: float = 1.0,
) -> cs.Function:
    r"""Continuous-time Input Constrained Control Barrier Function (ICCBF) for the given
    constraint ``h`` and system with dynamics ``dynamics_f_and_g`` subject to
    norm-bounded control action. This method constructs an ICCBF for the constraint
    :math:`h(x) \geq 0` using the given system's control-affine dynamics
    :math:`\dot{x} = f(x) + g(x) u`. Here, :math:`\dot{x}` is the time derivative of the
    state after applying control input :math:`u`, and :math:`f` and :math:`g` are
    returned by the dynamics function ``dynamics_f_and_g``. The ICCBF takes into account
    the norm-bounded control input :math:`|| u ||_p \leq b`, where the norm power
    :math:`p` and bound :math:`b` are specified as ``norm`` and ``bound``.

    The method can also compute a High-Order (HO) ICCBF by passing more than one class
    :math:`\mathcal{K}` functions ``alphas``.

    As per [1]_, the HO-ICCBF :math:`\phi_m` of degree :math:`m` is recursively found as

    .. math::
        \phi_m(x) = L_f \phi_{m-1}(x)
                    + \inf_{u \in \mathcal{U}} \left\{ L_g \phi_{m-1}(x) u \right\}
                    + \alpha_m(\phi_{m-1}(x))

    and should be imposed as the constraint :math:`\phi_m(x) \geq 0`.

    Parameters
    ----------
    h : callable
        The constraint function for which to build the ICCBF. It must be of the
        signature :math:`x \rightarrow h(x)`.
    x : casadi SX or MX
        The state vector variable :math:`x`.
    u : casadi SX or MX
        The control input vector variable :math:`u`.
    dynamics_f_and_g : callable
        A callable computing, for the given state, the dynamics components :math:`f(x)`
        and :math:`g(x)`, i.e., the signature is :math:`x \rightarrow f(x), g(x)`.
    alphas : iterable of callables
        An iterable of class :math:`\mathcal{K}` functions :math:`\alpha_m` for
        the HO-ICCBF. The length of the iterable determines the degree of the HO-ICCBF.
    norm : float
        The norm power :math:`p` for the control input norm constraint
        :math:`||u||_p \le b`, with :math:`1 \le p \le \infty`.
    bound : float, optional
        The bound :math:`b` for the control input norm constraint :math:`||u||_p \le b`,
        by default ``1.0``. Must be larger than zero.

    Returns
    -------
    casadi SX or MX
        Returns the HO-ICCBF function :math:`\phi_m` as a symbolic variable that is
        function of the provided ``x`` and ``u``.

    References
    ----------
    .. [1] Devansh R. Agrawal and Dimitra Panagou. Safe control synthesis via input
           constrained control barrier functions. In *60th IEEE Conference on Decision
           and Control (CDC)*, 6113-6118, 2021.

    Examples
    --------
    >>> import casadi as cs
    >>> A = cs.SX.sym("A", 2, 2)
    >>> B = cs.SX.sym("B", 2, 1)
    >>> x = cs.SX.sym("x", A.shape[0], 1)
    >>> u = cs.SX.sym("u", B.shape[1], 1)
    >>> dynamics_f_and_g = lambda x: (A @ x, B)
    >>> M = cs.SX.sym("M")
    >>> c = cs.SX.sym("c")
    >>> gamma = cs.SX.sym("gamma")
    >>> alphas = [lambda z: gamma * z]
    >>> h = lambda x: M - c * x[0]  # >= 0
    >>> cbf = iccbf(h, x, u, dynamics_f_and_g, alphas, norm=2, bound=0.5)
    >>> print(cbf)
    """
    # find the dual norm of the given norm
    y = cs.SX.sym("y", u.shape[0], 1)
    dual_norm_func = cs.Function("dual_norm", (y,), (_dual_norm(y, norm),))

    # continue from here as usual
    f, g = dynamics_f_and_g(x)
    phi = h(x)
    for alpha in alphas:
        Lf_phi = _lie_derivative(phi, x, f)
        Lg_phi = _lie_derivative(phi, x, g)
        Lg_phi_u_sup = bound * dual_norm_func(Lg_phi)
        alpha_phi = alpha(phi)
        phi = Lf_phi - Lg_phi_u_sup + alpha_phi
    return Lf_phi + Lg_phi * u + alpha_phi  # skip infimum in last iteration
