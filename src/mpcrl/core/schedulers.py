from itertools import accumulate, repeat
from operator import mul
from typing import Generic, TypeVar

import numpy as np

ScType = TypeVar("ScType")  # supports basic algebraic operations


class Scheduler(Generic[ScType]):
    """Schedulers are helpful classes to udpate or decay different quantities such as
    learning rates and/or exploration probability. This is a base class that actually
    does not update the value and keeps it constant."""

    __slots__ = ("value",)

    def __init__(self, init_value: ScType) -> None:
        """Builds the scheduler.

        Parameters
        ----------
        init_value : supports-algebraic-operations
            Initial value that will be updated by this scheduler.
        """
        super().__init__()
        self.value = init_value

    def step(self) -> None:
        """Updates the value of the scheduler by one step.

        Raises
        ------
        StopIteration
            Raises if the final iteration of the scheduler (if it has any) has been
            reached and `step` is called again.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x0={self.value})"

    def __str__(self) -> str:
        return self.__repr__()


class ExponentialScheduler(Scheduler):
    """Exponentiallly decays the value of the scheduler by `factor` every step, i.e.,
    after k steps the value `value_k = init_value * factor**k`."""

    __slots__ = ("factor",)

    def __init__(self, init_value: ScType, factor: ScType) -> None:
        """Builds the exponential scheduler.

        Parameters
        ----------
        init_value : supports-algebraic-operations
            Initial value that will be updated by this scheduler.
        factor : Tsc
            The exponential factor to decay the initial value with.
        """
        super().__init__(init_value)
        self.factor = factor

    def step(self) -> None:
        self.value *= self.factor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x0={self.value},factor={self.factor})"


class LinearScheduler(Scheduler):
    """Linearly updates the initial value of the scheduler towards a final one to be
    reached in N total steps, i.e., after k steps, the value is
    `value_k = init_value + (final_value - init_value) * k / total_steps`.
    """

    __slots__ = ("generator", "init_value", "final_value", "total_steps")

    def __init__(
        self, init_value: ScType, final_value: ScType, total_steps: int
    ) -> None:
        """Builds the exponential scheduler.

        Parameters
        ----------
        init_value : supports-algebraic-operations
            Initial value that will be updated by this scheduler.
        final_value : supports-algebraic-operations
            Final value that will be reached by the scheduler after `total_steps`.
        total_steps : int
            Total number of steps to linearly interpolate between `init_value` and
            `final_value`.
        """
        super().__init__(init_value)
        increment = (final_value - init_value) / total_steps  # type: ignore
        self.init_value = init_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.generator = accumulate(repeat(increment, total_steps), initial=init_value)
        next(self.generator)

    def step(self) -> None:
        self.value = next(self.generator)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(x={self.value},x0={self.init_value},"
            f"xf={self.final_value},N={self.total_steps})"
        )


class LogLinearScheduler(ExponentialScheduler):
    """Updates the initial value of the scheduler towards a final one to be
    reached in N total steps in a linear fashion between the exponents of the two, i.e.,
    after k steps, the value is
    `value_k = init_value * exp(ln(final_value / init_value) / total_steps)**k`.
    """

    __slots__ = ("generator", "init_value", "final_value", "total_steps")

    def __init__(
        self, init_value: ScType, final_value: ScType, total_steps: int
    ) -> None:
        factor = np.exp(np.log(final_value / init_value) / total_steps)  # type: ignore
        super().__init__(init_value, factor)
        self.init_value = init_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.generator = accumulate(
            repeat(factor, total_steps), func=mul, initial=init_value
        )
        next(self.generator)

    def step(self) -> None:
        self.value = next(self.generator)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(x={self.value},x0={self.init_value},"
            f"xf={self.final_value},N={self.total_steps})"
        )
