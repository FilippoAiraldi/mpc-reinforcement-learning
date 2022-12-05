class MpcSolverError(RuntimeError):
    """Exception class for raising errors when MPC solvers fails."""


class UpdateError(RuntimeError):
    """Exception class for raising errors when RL agent's updates fail."""
