from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import accumulate, repeat
from operator import mul
from typing import Generic, TypeVar, Union

import numpy as np

ScType = TypeVar("ScType")
ScType.__doc__ = "A type that supports basic algebraic operations."


class Scheduler(ABC, Generic[ScType]):
    """Schedulers are helpful classes to update or decay different quantities, such as
    learning rates and/or exploration probability.

    Note
    ----
    If the scheduler has a final iteration, it is expected to raise a `StopIteration`
    exception when the last iteration is reached.
    """

    def __init__(self, init_value: ScType) -> None:
        """Builds the scheduler.

        Parameters
        ----------
        init_value : supports-algebraic-operations
            Initial value that will be updated by this scheduler.
        """
        super().__init__()
        self.value = init_value

    @abstractmethod
    def step(self) -> None:
        """Updates the value of the scheduler by one step.

        Raises
        ------
        StopIteration
            Raises if the final iteration of the scheduler (if any) has been reached and
            `step` was called again.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x={self.value})"

    def __str__(self) -> str:
        return self.__repr__()


class NoScheduling(Scheduler[ScType]):
    """Scheduler that actually performs no scheduling and holds the initial value
    constant."""

    def step(self) -> None:
        return


class ExponentialScheduler(Scheduler[ScType]):
    """Exponentiallly decays the value of the scheduler by `factor` every step, i.e.,
    after k steps the value `value_k = init_value * factor**k`."""

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
        self.value *= self.factor  # type: ignore[operator]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x={self.value},factor={self.factor})"


class LinearScheduler(Scheduler[ScType]):
    """Linearly updates the initial value of the scheduler towards a final one to be
    reached in N total steps, i.e., after k steps, the value is
    `value_k = init_value + (final_value - init_value) * k / total_steps`.
    """

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
        increment = (final_value - init_value) / total_steps  # type: ignore[operator]
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


class LogLinearScheduler(ExponentialScheduler[ScType]):
    """Updates the initial value of the scheduler towards a final one to be
    reached in N total steps in a linear fashion between the exponents of the two, i.e.,
    after k steps, the value is
    `value_k = init_value * exp(ln(final_value / init_value) / total_steps)**k`.
    """

    def __init__(
        self, init_value: ScType, final_value: ScType, total_steps: int
    ) -> None:
        factor = np.exp(
            np.log(final_value / init_value) / total_steps  # type: ignore[operator]
        )
        super().__init__(init_value, factor)
        self.init_value = init_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.generator = accumulate(
            repeat(factor, total_steps), mul, initial=init_value
        )
        next(self.generator)

    def step(self) -> None:
        self.value = next(self.generator)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(x={self.value},x0={self.init_value},"
            f"xf={self.final_value},N={self.total_steps})"
        )


class Chain(Scheduler[ScType]):
    """Chains multiple schedulers together."""

    def __init__(
        self,
        schedulers: Iterable[Union[Scheduler[ScType], tuple[Scheduler[ScType], int]]],
    ) -> None:
        """Builds the chain of schedulers.

        Parameters
        ----------
        schedulers : iterable of schedulers or (scheduler, int)
            An iterable of schedulers to chain together. If a tuple is passed, the first
            element is expected to be a scheduler, and the second one an integer
            indicating the number of steps to run that scheduler for. This is useful in
            case the scheduler has no fixed number of steps, such as `LinearScheduler`.
        """
        self.schedulers = iter(schedulers)
        self._next_scheduler()
        super().__init__(self._current_scheduler.value)

    def step(self) -> None:
        if self._current_steps is not None:
            self._current_scheduler.step()
            self._current_steps -= 1
            if self._current_steps == 0:
                self._next_scheduler()
        else:
            try:
                self._current_scheduler.step()
            except StopIteration:
                self._next_scheduler()
        self.value = self._current_scheduler.value

    def _next_scheduler(self) -> None:
        """Fetches the next scheduler in the chain. If the chain is over, raises
        `StopIteration`, which is in line with the behaviour of other schedulers."""
        scheduler = next(self.schedulers)
        if isinstance(scheduler, tuple):
            self._current_scheduler, self._current_steps = scheduler
        else:
            self._current_scheduler = scheduler
            self._current_steps = None

    def __repr__(self) -> str:
        s = self._current_scheduler
        return f"{self.__class__.__name__}(x={self.value},current={s})"
