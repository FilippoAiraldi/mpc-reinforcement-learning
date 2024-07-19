"""Error and warning classes for signaling failure in the MPC solver and in the
updates, as well as convenience functions for raising these errors and warnings."""

from warnings import warn as _warn


class MpcSolverError(RuntimeError):
    """Exception class for raising errors when an MPC solver fails."""


class MpcSolverWarning(RuntimeWarning):
    """Warning class for raising errors when an MPC solver fails."""


class UpdateError(RuntimeError):
    """Exception class for raising errors when a learning agent's updates fails."""


class UpdateWarning(RuntimeWarning):
    """Warning class for raising errors when a learning agent's updates fails."""


def raise_or_warn_on_mpc_failure(msg: str, raises: bool) -> None:
    """Raises an error or warning with the given message due to an MPC solver's failure.

    Parameters
    ----------
    msg : str
        The exception or warning message.
    raises : bool
        If ``True``, raises an exception; otherwise, throws a warning.

    Raises
    ------
    UpdateError
        Raises :class:`MpcSolverError` if ``raises=True``; otherwise raises
        :class:`MpcSolverWarning`.
    """
    if raises:
        raise MpcSolverError(msg)
    else:
        _warn(msg, MpcSolverWarning)


def raise_or_warn_on_update_failure(msg: str, raises: bool) -> None:
    """Raises an error or warning with the given message due to an update failure.

    Parameters
    ----------
    msg : str
        The exception or warning message.
    raises : bool
        If ``True``, raises an exception; otherwise, throws a warning.

    Raises
    ------
    UpdateError
        Raises :class:`UpdateError` if ``raises=True``; otherwise raises
        :class:`UpdateWarning`.
    """
    if raises:
        raise UpdateError(msg)
    else:
        _warn(msg, UpdateWarning)
