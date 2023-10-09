from typing import Generic, Literal, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt

from mpcrl.core.schedulers import NoScheduling, Scheduler

LrType = TypeVar("LrType", npt.NDArray[np.floating], float)


class LearningRate(Generic[LrType]):
    """Learning rate class for scheduling and decaying its value during training."""

    def __init__(
        self,
        value: Union[LrType, Scheduler[LrType]],
        hook: Literal["on_update", "on_episode_end", "on_timestep_end"] = "on_update",
    ) -> None:
        """Initializes the learning rate.

        Parameters
        ----------
        init_value : float, array or scheduler of any of the two.
            The initial value of the learning rate, in general, a small number. A
            scheduler can be passed so that the learning rate is decayed according to
            its stepping strategy. The learning rate can be a single float, or an array
            of rates for each parameter.
        hook : {'on_update', 'on_episode_end', 'on_timestep_end'}, optional
            Specifies to which callback to hook, i.e., when to step the learning rate's
            scheduler to decay its value (see `step` method also). The options are:
             - `on_update` steps the exploration after each agent's update
             - `on_episode_end` steps the exploration after each episode's end
             - `on_timestep_end` steps the exploration after each env's timestep.

            By default, 'on_update' is selected.
        """
        self.scheduler: Scheduler[LrType] = (
            value if isinstance(value, Scheduler) else NoScheduling[LrType](value)
        )
        self._hook = hook

    @property
    def value(self) -> LrType:
        """Gets the current value of the learning rate."""
        return self.scheduler.value

    @property
    def hook(self) -> Optional[str]:
        """Specifies to which callback to hook, i.e., when to step the learning rate's
        scheduler to decay its value (see `step` method also). Can be `None` in case no
        hook is needed."""
        # return hook only if the learning rate scheduler requires to be stepped
        return None if isinstance(self.scheduler, NoScheduling) else self._hook

    def step(self, *_, **__) -> None:
        """Steps/decays the learning rate according to its scheduler."""
        self.scheduler.step()

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        hook = self.hook
        hookstr = "None" if hook is None else f"'{hook}'"
        return f"{self.__class__.__name__}(lr={self.scheduler},hook={hookstr})"
