from typing import Generic, Literal, TypeVar, Union

import numpy as np
import numpy.typing as npt

from mpcrl.core.schedulers import NoScheduling, Scheduler
from mpcrl.util.math import summarize_array

LrType = TypeVar("LrType", npt.NDArray[np.double], float)


class LearningRate(Generic[LrType]):
    """Learning rate class for scheduling and decaying its value during training."""

    __slots__ = ("value_scheduler", "stepping_strategy")

    def __init__(
        self,
        init_value: Union[LrType, Scheduler[LrType]],
        stepping_strategy: Literal[
            "on_update", "on_episode_start", "on_env_step"
        ] = "on_update",
    ) -> None:
        """Initializes the learning rate.

        Parameters
        ----------
        init_value : float, array or scheduler of any of the two.
            The initial value of the learning rate, in general, a small number. A
            scheduler can be passed so that the learning rate is decayed according to
            its stepping strategy. The learning rate can be a single float, or an array
            of rates for each parameter.
        stepping_strategy : {'on_update', 'on_episode_start', 'on_env_step'}, optional
            Specifies when to step the learning rate's scheduler to decay its value (see
            `step` method also). The options are:

                - `on_update` steps the exploration after each agent's update
                - `on_episode_start` steps the exploration after each episode's start
                - `on_env_step` steps the exploration after each env's step

            By default, 'on_update' is selected.
        """
        self.value_scheduler: Scheduler[LrType] = (
            init_value
            if isinstance(init_value, Scheduler)
            else NoScheduling[LrType](init_value)
        )
        self.stepping_strategy = stepping_strategy

    @property
    def value(self) -> LrType:
        """Gets the current value of the learning rate."""
        return self.value_scheduler.value

    def step(self) -> None:
        """Steps/decays the learning rate according to its scheduler."""
        self.value_scheduler.step()

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        v = self.value
        s = (
            f"{v:.3f}"
            if isinstance(v, (float, int))
            else summarize_array(v)  # type: ignore[arg-type]
        )
        return f"{self.__class__.__name__}(lr={s},step={self.stepping_strategy})"
