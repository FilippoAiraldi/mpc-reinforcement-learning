from typing import Any, Dict, Optional, Protocol, SupportsFloat, Tuple, TypeVar

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class GymEnvLike(Protocol[ObsType, ActType]):  # type: ignore
    """Class that exposes an API similar to OpenAI's Gym environments, with methods for
    - resetting the env
    - stepping the env.
    """

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Run one timestep of the environment's dynamics."""

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Resets the environment to an initial state and returns the initial
        observation."""
