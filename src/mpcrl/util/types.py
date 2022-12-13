from typing import Optional, Protocol, Tuple, TypeVar

import numpy as np

Tobs = TypeVar("Tobs", bound=np.generic)
Tact = TypeVar("Tact", bound=np.generic)


class GymEnvLike(Protocol[Tobs, Tact]):  # type: ignore
    """Class that exposes an API similar to OpenAI's Gym environments, with methods for
    - resetting the env
    - stepping the env.
    """

    def step(self, action: Tact) -> Tuple[Tobs, float, bool, bool, dict]:
        """Run one timestep of the environment's dynamics."""

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Tobs, dict]:
        """Resets the environment to an initial state and returns the initial
        observation."""
