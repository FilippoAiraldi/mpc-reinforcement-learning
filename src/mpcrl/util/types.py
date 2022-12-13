from typing import Optional, Protocol, Tuple, TypeVar

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class GymEnvLike(Protocol[ObsType, ActType]):  # type: ignore
    """Class that exposes an API similar to OpenAI's Gym environments, with methods for
    - resetting the env
    - stepping the env.
    """

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Run one timestep of the environment's dynamics."""

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        """Resets the environment to an initial state and returns the initial
        observation."""
