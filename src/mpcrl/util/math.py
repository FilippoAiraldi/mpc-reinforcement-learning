from itertools import combinations
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_discrete_are
from scipy.special import comb


def summarize_array(
    a: npt.NDArray[np.double], precision: int = 3, ddof: int = 0
) -> str:
    """Summarizes the stats of a given array.

    Parameters
    ----------
    a : array
        The numerical array to summarize.
    precision : int, optional
        The decimal precision, by default 3.
    ddof : int, optional
        Degrees of freedom used to compute the standard deviation, by default 0.

    Returns
    -------
    str
        The summarizing string.
    """
    n = a.size
    mean = a.mean()
    std = a.std(ddof=ddof)
    min = a.min()
    max = a.max()
    return (
        f"n={n} x∈[{min:.{precision}f}, {max:.{precision}f}] "
        f"μ={mean:.{precision}f} σ={std:.{precision}f}"
    )


def cholesky_added_multiple_identities(
    A: npt.NDArray[np.double], beta: float = 1e-3, maxiter: int = 1000
) -> npt.NDArray[np.double]:
    """Lower Cholesky factorization with added multiple of the identity matrix to ensure
    positive-definitiveness [1, Algorithm 3.3].

    Parameters
    ----------
    A : array of double
        The matrix to compute the cholesky factorization of.
    beta : float, optional
        Initial tolerance of the algorithm, by default 1e-3.
    maxiter : int, optional
        Maximum iterations of the algorithm, by default 1000.

    Returns
    -------
    array of double
        The lower cholesky factorization of the modified A (with the addition of
        identity matrices to ensure that it is positive-definite).

    Raises
    ------
    ValueError
        If the factorization is unsuccessful for the maximum number of iterations.

    References
    ----------
    [1] J. Nocedal, S. Wright. 2006. Numerical Optimization. Springer.
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


def dlqr(
    A: npt.NDArray[np.double],
    B: npt.NDArray[np.double],
    Q: npt.NDArray[np.double],
    R: npt.NDArray[np.double],
    M: Optional[npt.NDArray[np.double]] = None,
) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
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


def nchoosek(n: Union[int, npt.ArrayLike], k: int) -> Union[int, np.ndarray]:
    """Emulates the `nchoosek` function from Matlab. Returns the binomial coefficient,
    i.e.,  the number of combinations of `n` items taken `k` at a time. If `n` is an
    array, then it is flatten and all possible combinations of its elements are
    returned.

    Parameters
    ----------
    n : int or array_like
        Number of elements or array of elements to choose from.
    k : int
        Number of elements to choose.

    Returns
    -------
    int or array
        Depending on the type of input `n`, the output is either the total number of
        combinations or the combinations in a matrix.
    """
    return (
        comb(n, k, exact=True)
        if isinstance(n, int)
        else np.row_stack(list(combinations(np.asarray(n).flatten(), k)))
    )


def monomial_powers(d: int, k: int) -> npt.NDArray[np.int64]:
    """Computes the powers of all `d`-dimensional monomials of degree `k`.

    Parameters
    ----------
    d : int
        The number of monomial elements.
    k : int
        The degree of each monomial.

    Returns
    -------
    array of ints
        An array containing in each row the power of each index in order to obtain the
        desired monomial of power `k`.
    """
    m = nchoosek(k + d - 1, d - 1)
    dividers = np.column_stack(
        (
            np.zeros((m, 1), dtype=int),
            np.row_stack(  # type: ignore[call-overload]
                nchoosek(np.arange(1, k + d), d - 1)
            ),
            np.full((m, 1), k + d, dtype=int),
        )
    )
    return np.flipud(np.diff(dividers, axis=1) - 1).astype(int)
