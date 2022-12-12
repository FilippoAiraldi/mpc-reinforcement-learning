from typing import Any


class Scheduler:
    """Schedulers are helpful classes to udpate or decay different quantities such as
    learning rates and/or exploration probability. This is a base class that actually
    does not update the value and keeps it constant."""

    __slots__ = ("value",)

    def __init__(self, init_value: Any) -> None:
        """Builds the scheduler.

        Parameters
        ----------
        init_value
            Initial value that will be updated by this scheduler.
        """
        super().__init__()
        self.value = init_value

    def step(self) -> None:
        """Updates the value of the scheduler by one step."""
        return  # does literally nothing


class ExponentialScheduler(Scheduler):
    """Exponentiallly decays the value of the scheduler by `factor` every step, i.e.,
    after k steps the value `value_k = init_value * factor**k`."""

    __slots__ = ("factor",)

    def __init__(self, init_value: Any, factor: float) -> None:
        """Builds the exponential scheduler.

        Parameters
        ----------
        init_value
            Initial value that will be updated by this scheduler.
        factor : float
            The exponential factor to decay the initial value with.
        """
        super().__init__(init_value)
        self.factor = factor

    def step(self) -> None:
        self.value *= self.factor


class LinearScheduler(Scheduler):
    """Linearly updates the initial value of the scheduler towards a final one to be
    reached in N total steps, i.e., after k steps, the value is
    `value_k = init_value + (final_value - init_value) * k / total_steps `
    """

    __slots__ = ("increment",)

    def __init__(self, init_value: Any, final_value: Any, total_steps: int) -> None:
        """Builds the exponential scheduler.

        Parameters
        ----------
        init_value
            Initial value that will be updated by this scheduler.
        final_value
            Final value that will be reached by the scheduler after `total_steps`.
        total_steps : int
            Total number of steps to linearly interpolate between `init_value` and
            `final_value`.
        """
        super().__init__(init_value)
        self.increment = (final_value - init_value) / total_steps

    def step(self) -> None:
        self.value += self.increment
