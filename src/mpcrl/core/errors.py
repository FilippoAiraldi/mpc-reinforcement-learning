from warnings import warn


class MpcSolverError(RuntimeError):
    """Exception class for raising errors when MPC solvers fails."""


class MpcSolverWarning(RuntimeWarning):
    """Warning class for raising errors when MPC solvers fails."""


class UpdateError(RuntimeError):
    """Exception class for raising errors when RL agent's updates fail."""


class UpdateWarning(RuntimeWarning):
    """Warning class for raising errors when RL agent's updates fail."""


def raise_or_warn_on_update_failure(msg: str, raises: bool) -> None:
    """Raises errors or warnings with a message for update failures.

    Parameters
    ----------
    msg : str
        The exception or warning message.
    raises : bool
        If `True`, raises an exception; otherwise, throws a warning.

    Raises
    ------
    UpdateError
        Raises `UpdateError` if `raises=True`; otherwise raises
        `UpdateWarning`.
    """
    if raises:
        raise UpdateError(msg)
    else:
        warn(msg, UpdateWarning)


def raise_or_warn_on_mpc_failure(msg: str, raises: bool) -> None:
    """Raises errors or warnings with a message for MPC failures.

    Parameters
    ----------
    msg : str
        The exception or warning message.
    raises : bool
        If `True`, raises an exception; otherwise, throws a warning.

    Raises
    ------
    UpdateError
        Raises `MpcSolverError` if `raises=True`; otherwise raises
        `MpcSolverWarning`.
    """
    if raises:
        raise MpcSolverError(msg)
    else:
        warn(msg, MpcSolverWarning)
