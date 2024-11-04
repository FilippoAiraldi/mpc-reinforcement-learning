"""A submodule providing both base classes for schedulers and some concrete
implementations for the most common cases, such as linearly and exponentially decaying
schedulers."""

import abc
from collections.abc import Iterable as _Iterable
from itertools import accumulate as _acc
from itertools import repeat as _rep
from operator import mul as _mul
from typing import Generic as _Generic
from typing import TypeVar as _TypeVar
from typing import Union

import numpy as np
import numpy.typing as npt

ScType = _TypeVar("ScType", int, float, npt.NDArray[np.int_], npt.NDArray[np.floating])


class Scheduler(abc.ABC, _Generic[ScType]):
    """Base abstract class for schedulers.

    Schedulers are helpful classes to update or decay different quantities, such as
    learning rates or exploration probability.

    Parameters
    ----------
    init_value : supports-algebraic-operations
        Initial value that will be updated by this scheduler.

    Notes
    -----
    If the scheduler has a final iteration, it is expected to raise a
    :class:`StopIteration` when the last iteration is reached.
    """

    def __init__(self, init_value: ScType) -> None:
        super().__init__()
        self.value = init_value

    @abc.abstractmethod
    def step(self) -> None:
        """Updates the value of the scheduler by one step.

        Raises
        ------
        StopIteration
            Raises if the final iteration of the scheduler (if any) has been reached and
            :meth:`step` was called again.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x={self.value})"

    def __str__(self) -> str:
        return self.__repr__()


class NoScheduling(Scheduler[ScType]):
    """Scheduler that actually performs no scheduling and holds the initial value
    indefinitely constant."""

    def step(self) -> None:
        return


class ExponentialScheduler(Scheduler[ScType]):
    r"""Exponentially decay scheduler.

    It exponentially decays the value of the scheduler by ``factor`` every step, i.e.,
    mathematically, at the :math:`k`-th step, the value :math:`v_k` is

    .. math:: v_k = v_0 f^k,

    where :math:`v_0` is the initial value and :math:`f` is the ``factor``.

    Parameters
    ----------
    init_value : supports-algebraic-operations
        Initial value that will be updated by this scheduler.
    factor : supports-algebraic-operations
        The exponential factor to decay the initial value with.
    """

    def __init__(self, init_value: ScType, factor: ScType) -> None:
        super().__init__(init_value)
        self.factor = factor

    def step(self) -> None:
        self.value *= self.factor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x={self.value},factor={self.factor})"


class LinearScheduler(Scheduler[ScType]):
    r"""Linear scheduling from initial to final value in a fixed number of steps.

    This scheduler linearly updates the initial value towards a final one to be reached
    in :math:`N` total steps, i.e., mathematically, at the :math:`k`-th step, the value
    :math:`v_k` is

    .. math:: v_k = v_0 + \left(v_N - v_0\right) \frac{k}{N},

    where :math:`v_0` is the initial value and :math:`v_0` the final.

    Parameters
    ----------
    init_value : supports-algebraic-operations
        Initial value that will be updated by this scheduler.
    final_value : supports-algebraic-operations
        Final value that will be reached by the scheduler after ``total_steps``.
    total_steps : int
        Total number of steps to linearly interpolate between ``init_value`` and
        ``final_value``.
    """

    def __init__(
        self, init_value: ScType, final_value: ScType, total_steps: int
    ) -> None:
        super().__init__(init_value)
        increment = (final_value - init_value) / total_steps
        self.init_value = init_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.generator = _acc(_rep(increment, total_steps), initial=init_value)
        next(self.generator)

    def step(self) -> None:
        self.value = next(self.generator)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(x={self.value},x0={self.init_value},"
            f"xf={self.final_value},N={self.total_steps})"
        )


class LogLinearScheduler(ExponentialScheduler[ScType]):
    r"""Scheduler that updates the scheduled quantity from the initial to the final
    value in a logarithmic fashion.

    In other words, this scheduler updates the initial value of the scheduler towards
    the final one (to be reached in :math:`N` total steps) in a linear fashion between
    the exponents of the two. Mathematically, at the :math:`k`-th step, the value
    :math:`v_k` is

    .. math:: v_k = v_0 \exp\left(
            \text{ln} \left( \frac{v_N}{v_0} \right) \frac{k}{N}
        \right),

    where :math:`v_0` is the initial value, and :math:`v_N` the final.

    Parameters
    ----------
    init_value : supports-algebraic-operations
        Initial value that will be updated by this scheduler.
    final_value : supports-algebraic-operations
        Final value that will be reached by the scheduler after ``total_steps``.
    total_steps : int
        Total number of steps to log-linearly interpolate between ``init_value`` and
        ``final_value``.
    """

    def __init__(
        self, init_value: ScType, final_value: ScType, total_steps: int
    ) -> None:
        factor = np.exp(np.log(final_value / init_value) / total_steps)
        super().__init__(init_value, factor)
        self.init_value = init_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.generator = _acc(_rep(factor, total_steps), _mul, initial=init_value)
        next(self.generator)

    def step(self) -> None:
        self.value = next(self.generator)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(x={self.value},x0={self.init_value},"
            f"xf={self.final_value},N={self.total_steps})"
        )


class Chain(Scheduler[ScType]):
    """Chains multiple schedulers together.

    Parameters
    ----------
    schedulers : iterable of schedulers or of (scheduler, int)
        An iterable of schedulers to be chained together. If an iterable of tuples is
        passed instead, the first element is expected to be an instance of scheduler,
        and the second one an integer indicating the number of steps to run that
        scheduler for. This is useful in case the scheduler has no fixed number of
        steps, such as :class:`ExponentialScheduler`.
    """

    def __init__(
        self,
        schedulers: _Iterable[Union[Scheduler[ScType], tuple[Scheduler[ScType], int]]],
    ) -> None:
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
        :class:`StopIteration`, which is in line with the behaviour of other
        schedulers."""
        scheduler = next(self.schedulers)
        if isinstance(scheduler, tuple):
            self._current_scheduler, self._current_steps = scheduler
        else:
            self._current_scheduler = scheduler
            self._current_steps = None

    def __repr__(self) -> str:
        s = self._current_scheduler
        return f"{self.__class__.__name__}(x={self.value},current={s})"
