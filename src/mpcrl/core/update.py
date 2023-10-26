from collections.abc import Iterator
from typing import Literal

from ..util.iters import bool_cycle, chain, repeat


class UpdateStrategy:
    """A class for the update strategy."""

    def __init__(
        self,
        frequency: int,
        hook: Literal["on_episode_end", "on_timestep_end"] = "on_timestep_end",
        skip_first: int = 0,
    ) -> None:
        """Initializes the update strategy.

        Parameters
        ----------
        frequency : int
            Frequency at which, each time the hook is called, an update should be
            carried out.
        skip_first : int, optional
            Skips the first `skip_first` updates. By default 0, so no update is skipped.
            This is useful when, e.g., the agent has to wait for the experience buffer
            to be filled before starting to update.
        hook : {'on_episode_end', 'on_timestep_end'}, optional
            Specifies to which callback to hook, i.e., when to check if an update is due
            according to the given frequency. The options are:
             - `on_episode_end` checks for an update after each episode's end
             - `on_timestep_end` checks for an update after each env's timestep.

            By default, 'on_timestep_end' is selected.
        """
        self.frequency = frequency
        self.hook = hook
        self._update_cycle = chain(
            repeat(False, skip_first * frequency), bool_cycle(frequency)
        )

    def can_update(self) -> bool:
        """Returns whether an update must be carried out at the current instant
        according to the strategy.

        Notes
        -----
        This methods steps internal iterators to check whether an update is due, so
        calling this method again will return a different value.

        Returns
        -------
        bool
            `True` if the agent should update according to this strategy; otherwise,
            `False`.
        """
        return next(self._update_cycle)

    def __iter__(self) -> Iterator[bool]:
        """With `__next__`, makes this class act like an iterator."""
        return self._update_cycle

    def __next__(self) -> bool:
        """With `__iter__`, makes this class act like an iterator."""
        return next(self._update_cycle)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(frequency={self.frequency},hook={self.hook})"

    def __str__(self) -> str:
        return self.__repr__()
