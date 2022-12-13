class MpcSolverError(RuntimeError):
    """Exception class for raising errors when MPC solvers fails."""


class MpcSolverWarning(RuntimeWarning):
    """Warning class for raising errors when MPC solvers fails."""

    @classmethod
    def from_exception(cls, ex: MpcSolverError) -> "MpcSolverWarning":
        """Creates a warning from the corresponding error."""
        return cls(*ex.args)


class UpdateError(RuntimeError):
    """Exception class for raising errors when RL agent's updates fail."""


class UpdateWarning(RuntimeWarning):
    """Warning class for raising errors when RL agent's updates fail."""

    @classmethod
    def from_exception(cls, ex: UpdateError) -> "UpdateWarning":
        """Creates a warning from the corresponding error."""
        return cls(*ex.args)
