"""The update strategy is likely to be one of the most important aspects that a designer
has to consider when training a Reinforcement Learning agent. When instantiating an
agent, via :class:`UpdateStrategy`, the user can specify when and with which frequency
to update the agent's MPC parametrization (e.g., at the end of every training episode,
or every ``N`` time steps), as well as the number of updates to skip at the beginning
(in case we need to wait for experience buffers to properly fill first). See
:ref:`user_guide_updating` for a more thorough explanation."""

from collections.abc import Iterator
from itertools import chain, repeat
from typing import Literal

from ..util.iters import bool_cycle


class UpdateStrategy:
    """A class holding information on the update strategy to be used by the learning
    algorithm.

    Parameters
    ----------
    frequency : int
        Frequency at which, each time the hook is called, an update should be
        carried out.
    skip_first : int, optional
        Skips the first ``skip_first`` updates. By default ``0``, so no update is
        skipped. This is useful when, e.g., the agent has to wait for the experience
        buffer to be filled before starting to update.
    hook : {"on_episode_end", "on_timestep_end"}, optional
        Specifies to which callback to hook, i.e., when to check if an update is due
        according to the given frequency. The options are:

        - ``"on_episode_end"`` checks if an update is due  after each episode ends
        - ``"on_timestep_end"`` checks for an update after each simulation's time step.

        By default, ``"on_timestep_end"`` is selected.
    """

    def __init__(
        self,
        frequency: int,
        hook: Literal["on_episode_end", "on_timestep_end"] = "on_timestep_end",
        skip_first: int = 0,
    ) -> None:
        self.frequency = frequency
        self.hook = hook
        self._update_cycle = chain(
            repeat(False, skip_first * frequency), bool_cycle(frequency)
        )

    def can_update(self) -> bool:
        """Returns whether an update must be carried out now, at the current instant,
        according to the specified strategy.

        Notes
        -----
        This methods steps the internal iterators to check whether an update is due with
        :func:`next`. This means that calling this method has a side effect on the state
        of these iterators, and calling immediately again can result in a different
        outcome.

        Returns
        -------
        bool
            ``True`` if the agent should update according to this strategy; otherwise,
            ``False``.
        """
        return next(self._update_cycle)

    def __iter__(self) -> Iterator[bool]:
        return self._update_cycle

    def __next__(self) -> bool:
        return next(self._update_cycle)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(frequency={self.frequency},hook={self.hook})"

    def __str__(self) -> str:
        return self.__repr__()
