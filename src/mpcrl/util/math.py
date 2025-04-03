"""A collection of functions for mathematical operations and utilities. In particular,
these functions support the creation of monomial basis functions for approximating the
value function, and the modifications of Hessian matrices to positive-definite ones."""

from itertools import combinations as _combinations
from typing import Optional
from typing import TypeVar as _TypeVar
from typing import Union

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp.util.math import prod as _prod
from scipy.special import comb as _comb

SymType = _TypeVar("SymType", cs.SX, cs.MX)
SymOrNumType = _TypeVar("SymOrNumType", cs.SX, cs.MX, cs.DM, np.ndarray)


def summarize_array(
    a: npt.NDArray[np.floating],
    precision: int = 3,
    ddof: int = 0,
    axis: Optional[int] = None,
) -> str:
    """Summarizes the stats of a given array, i.e., the shape, the minimum and maximum,
    and the mean and standard deviation.

    Parameters
    ----------
    a : array
        The numerical array to summarize.
    precision : int, optional
        The decimal precision, by default ``3``.
    ddof : int, optional
        Degrees of freedom used to compute the standard deviation, by default ``0``.
    axis : int, optional
        The axis along which to summarize the array, by default ``None``.

    Returns
    -------
    str
        The summarizing string.
    """
    s = a.shape
    mean = a.mean(axis=axis)
    std = a.std(ddof=ddof, axis=axis)
    min = a.min(axis=axis)
    max = a.max(axis=axis)
    return (
        f"s={s}, x∈[{min:.{precision}f}, {max:.{precision}f}], "
        f"μ={mean:.{precision}f}, σ={std:.{precision}f}"
    )


def cholesky_added_multiple_identity(
    A: npt.NDArray[np.floating], beta: float = 1e-3, maxiter: int = 1000
) -> npt.NDArray[np.floating]:
    r"""Lower Cholesky factorization with added multiple of the identity matrix to ensure
    positive-definitiveness from Algorithm 3.3 in :cite:`nocedal_numerical_2006`.

    The basic idea is to add a multiple of the identity matrix to the original matrix
    unitl the factorization is successful, i.e., find :math:`\tau \ge 0` such that
    :math:`L L^\top = A + \tau I` is successful.

    Parameters
    ----------
    A : array of double
        The 2D matrix to compute the cholesky factorization of.
    beta : float, optional
        Initial tolerance of the algorithm, by default ``1e-3``.
    maxiter : int, optional
        Maximum iterations of the algorithm, by default ``1000``.

    Returns
    -------
    array of double
        The lower cholesky factorization of the modified ``A`` (with the addition of
        identity matrices to ensure that it is positive-definite).

    Raises
    ------
    ValueError
        If the factorization is unsuccessful for the maximum number of iterations.
    """
    a_min = np.diag(A).min()
    tau = 0 if a_min > 0 else -a_min + beta
    identity = np.eye(A.shape[0])
    for _ in range(maxiter):
        try:
            return np.linalg.cholesky(A + tau * identity)
        except np.linalg.LinAlgError:
            tau = max(1.05 * tau, beta)
    raise ValueError("Maximum iterations reached.")


def nchoosek(n: Union[int, npt.ArrayLike], k: int) -> Union[int, np.ndarray]:
    """Emulates
    `MATLAB's nchoosek <https://www.mathworks.com/help/matlab/ref/nchoosek.html>`_, and
    returns the binomial coefficient, i.e.,  the number of combinations of ``n`` items
    taken ``k`` at a time. If ``n`` is an array, then it is flatten and all possible
    combinations of its elements are returned.

    Parameters
    ----------
    n : int or array_like
        Number of elements or array of elements to choose from.
    k : int
        Number of elements to choose.

    Returns
    -------
    int or array
        Depending on the type of input ``n``, the output is either the total number of
        combinations or the combinations in a matrix.
    """
    return (
        _comb(n, k, exact=True)
        if isinstance(n, int)
        else np.vstack(list(_combinations(np.asarray(n).reshape(-1), k)))
    )


def monomial_powers(n: int, k: int) -> npt.NDArray[np.int_]:
    r"""Computes the powers of all ``n``-dimensional monomials of degree ``k``. In
    mathematical terms, consider the monomial
    :math:`\prod_{i=1}^n x_i^{p_i}`. Then, this functions returns all possible
    combinations of powers :math:`(p_1, p_2, \ldots, p_n)` such that their sum is equal
    to ``k``, i.e., :math:`\sum_{i=1}^n p_i = k`.

    Parameters
    ----------
    n : int
        The number of monomial elements.
    k : int
        The degree of each monomial.

    Returns
    -------
    2d array of ints
        A 2D array containing in each row the power of each index in order to obtain the
        desired monomial of power ``k``. The shape of the array is :math:`(c, n)`, where
        :math:`c = \frac{(k + n - 1)!}{k!(n - 1)!}`.

    Examples
    --------
    >>> from mpcrl.util.math import monomial_powers
    >>> monomial_powers(3, 2)
    array([[2, 0, 0],
           [1, 1, 0],
           [1, 0, 1],
           [0, 2, 0],
           [0, 1, 1],
           [0, 0, 2]])
    """
    # see https://en.wikipedia.org/wiki/Homogeneous_polynomial#Properties and
    # https://math.stackexchange.com/a/36251
    m = nchoosek(k + n - 1, n - 1)
    dividers = np.column_stack(
        (
            np.zeros((m, 1), int),
            np.vstack(nchoosek(np.arange(1, k + n), n - 1)),
            np.full((m, 1), k + n, int),
        )
    )
    return np.flipud(np.diff(dividers, axis=1) - 1).astype(int)


def monomials_basis_function(n: int, mindegree: int, maxdegree: int) -> cs.Function:
    r"""Creates a :class:`casadi.Function` made of monomials as bases in ``n`` variables
    and degrees from ``mindegree`` to ``maxdegree``.

    In mathematical terms, given an input vector :math:`s \in \mathbb{R^n}` and the
    minimum and maximum degrees :math:`m` and :math:`M`, consider the degree
    :math:`d_i=m+i` with :math:`i=0,\ldots,M-m`. Associated to this degree, there exist
    :math:`c_i = \frac{(d_i + n - 1)!}{d_i!(n - 1)!}` monomials of the form
    :math:`m_{i,k}(s) = \prod_{j=1}^n s_j^{p_{j,k}}, \ k = 1,\ldots,c_i` satisfying the
    condition :math:`\sum_{j=1}^n p_{j,k} = d_i` (see :func:`monomial_powers`).

    This function returns a basis vector
    :math:`\Phi : \mathbb{R}^n \rightarrow \mathbb{R}^{\sum_{i=0}^{M-m}{c_i}}` that
    stacks all of the aforementioned monomials:

    .. math::

        \Phi(s) = \Bigr[
            m_{0,1}(s), \ldots, m_{0,c_0}(s), m_{1,1}(s), \ldots, m_{1,c_1}(s), \ldots, m_{M-m,1}(s), \ldots, m_{M-m,c_{M-m}}(s)
        \Bigl]^\top

    Parameters
    ----------
    n : int
        Dimension of the input vector.
    mindegree : int
        Minimum degree of monomials (included).
    maxdegree : int
        Maximum degree of monomials (included).

    Returns
    -------
    :class:`casadi.Function`
        A casadi function of the form :math:`\Phi(s)`, where :math:`s \in \mathbb{R^n}`
        is the input and :math:`\Phi` the basis vector.

    Examples
    --------
    >>> import casadi as cs
    >>> from mpcrl.util.math import monomial_powers
    >>> Phi = monomials_basis_function(3, 1, 2)
    >>> s = cs.SX.sym("s", Phi.size1_in(0), 1)
    >>> Phi(s)
    SX([s_0, s_1, s_2, sq(s_0), (s_0*s_1), (s_0*s_2), sq(s_1), (s_1*s_2), sq(s_2)])
    """
    s = cs.SX.sym("s", n, 1)  # prod is faster with SX, and during runtime
    y = cs.vertcat(
        *(
            _prod(s**p)
            for k in range(mindegree, maxdegree + 1)
            for p in monomial_powers(n, k)
        )
    )
    return cs.Function("Phi", (s,), (y,), ("s",), ("Phi(s)",), {"cse": True})


def clip(x: SymOrNumType, lower: SymOrNumType, upper: SymOrNumType) -> SymOrNumType:
    """Clips variable ``x`` to the range defined by ``lower`` and ``upper``.

    Parameters
    ----------
    x : casadi SX or MX or array-like
        The variable to clip.
    lower : casadi SX or MX or array-like
        The lower bound of the clipping.
    upper : casadi SX or MX or array-like
        The upper bound of the clipping.

    Returns
    -------
    casadi SX or MX or array-like
        The clipped variable.
    """
    return cs.fmax(lower, cs.fmin(upper, x))


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
    deriv = cs.mtimes(cs.jacobian(ex, arg), field)
    if order <= 1:
        return deriv
    return lie_derivative(deriv, arg, field, order - 1)


def dual_norm(x: SymOrNumType, ord: float) -> SymOrNumType:
    r"""Computes the dual norm of a given vector ``x`` with respect to the given order
    ``ord``. The dual norm of :math:`x` is defined as

    .. math:: || x ||_q = \sup_{|| y ||_p \leq 1} y^\top x,

    where :math:`p` is specified by ``ord`` and :math:`q` is its conjugate power, i.e.,
    :math:`1/p + 1/q = 1`.

    Parameters
    ----------
    x : casadi SX, MX, DM, or array-like
        The input scalar or vector. If a scalar, the absolute value is returned.
    ord : float
        The order of the norm. Must be larger than or equal to 1.

    Returns
    -------
    casadi SX, MX, or DM
        The dual norm of the input vector. Depending on the input type, the output is
        either symbolic or numeric.
    """
    if getattr(x, "shape", ()) in ((), (1,), (1, 1)):
        return cs.fabs(x)
    if ord == 1:
        return cs.norm_inf(x)
    if ord == 2:
        return cs.norm_2(x)
    if np.isposinf(ord).item():
        return cs.norm_1(x)
    dual_norm = ord / (ord - 1)
    return cs.power(cs.sum1(cs.power(cs.fabs(x), dual_norm)), 1 / dual_norm)
