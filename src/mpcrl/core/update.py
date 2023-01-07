from typing import Iterator, Literal

from mpcrl.util.iters import bool_cycle


class UpdateStrategy:
    """A class for the update strategy."""

    __slots__ = ("frequency", "hook", "update_cycle")

    def __init__(
        self,
        frequency: int,
        hook: Literal["on_episode_end", "on_env_step"] = "on_env_step",
    ) -> None:
        """Initializes the update strategy.

        Parameters
        ----------
        frequency : int
            Frequency at which, each time the hook is called, an update should be
            carried out.
        hook : {'on_update', 'on_episode_end', 'on_env_step'}, optional
            Specifies to which callback to hook, i.e., when to check if an update is due
            according to the given frequency. The options are:

                - `on_episode_end` checks for an update after each episode's end
                - `on_env_step` checks for an update after each env's step.

            By default, 'on_env_step' is selected.
        """
        self.frequency = frequency
        self.hook = hook
        self.update_cycle = bool_cycle(frequency)

    def can_update(self) -> bool:
        """Returns whether an update must be carried out at the current instant
        according to the strategy.

        Notes
        -----
        This methods steps an internal iterator to check whether an update is due, so
        calling this method again will return a different value.

        Returns
        -------
        bool
            `True` if the agent should update according to this strategy; otherwise,
            `False`.
        """
        return next(self.update_cycle)

    def __iter__(self) -> Iterator[bool]:
        """With `__next__`, makes this class act like an iterator."""
        return self.update_cycle

    def __next__(self) -> bool:
        """With `__iter__`, makes this class act like an iterator."""
        return next(self.update_cycle)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(frequency={self.frequency},hook={self.hook})"

    def __str__(self) -> str:
        return self.__repr__()
