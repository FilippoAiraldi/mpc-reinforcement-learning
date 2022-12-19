import numpy as np
import numpy.typing as npt


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
