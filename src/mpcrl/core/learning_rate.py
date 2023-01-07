from typing import Generic, Literal, TypeVar, Union

import numpy as np
import numpy.typing as npt

from mpcrl.core.schedulers import NoScheduling, Scheduler
from mpcrl.util.math import summarize_array

LrType = TypeVar("LrType", npt.NDArray[np.double], float)


class LearningRate(Generic[LrType]):
    """Learning rate class for scheduling and decaying its value during training."""

    __slots__ = ("scheduler", "hook")

    def __init__(
        self,
        init_value: Union[LrType, Scheduler[LrType]],
        hook: Literal["on_update", "on_episode_end", "on_env_step"] = "on_update",
    ) -> None:
        """Initializes the learning rate.

        Parameters
        ----------
        init_value : float, array or scheduler of any of the two.
            The initial value of the learning rate, in general, a small number. A
            scheduler can be passed so that the learning rate is decayed according to
            its stepping strategy. The learning rate can be a single float, or an array
            of rates for each parameter.
        hook : {'on_update', 'on_episode_end', 'on_env_step'}, optional
            Specifies to which callback to hook, i.e., when to step the learning rate's
            scheduler to decay its value (see `step` method also). The options are:

                - `on_update` steps the exploration after each agent's update
                - `on_episode_end` steps the exploration after each episode's end
                - `on_env_step` steps the exploration after each env's step.

            By default, 'on_update' is selected.
        """
        self.scheduler: Scheduler[LrType] = (
            init_value
            if isinstance(init_value, Scheduler)
            else NoScheduling[LrType](init_value)
        )
        self.hook = hook

    @property
    def value(self) -> LrType:
        """Gets the current value of the learning rate."""
        return self.scheduler.value

    def step(self) -> None:
        """Steps/decays the learning rate according to its scheduler."""
        self.scheduler.step()

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        v = self.value
        s = (
            f"{v:.3f}"
            if isinstance(v, (float, int))
            else summarize_array(v)  # type: ignore[arg-type]
        )
        return f"{self.__class__.__name__}(lr={s},hook={self.hook})"
