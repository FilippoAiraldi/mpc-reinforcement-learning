from typing import Any, Dict, Generic, Optional, Tuple, Type, Union

from csnlp.util.io import SupportsDeepcopyAndPickle

from mpcrl.agents.agent import Agent, SymType
from mpcrl.agents.learning_agent import ExpType, LearningAgent


class Wrapper(SupportsDeepcopyAndPickle, Generic[SymType]):
    """Wraps a learning agent to allow a modular transformation of its methods. This
    class is the base class for all wrappers. The subclass could override some methods
    to change the behavior of the original environment without touching the original
    code."""

    __slots__ = ("agent",)

    def __init__(self, agent: Agent[SymType]) -> None:
        """Wraps an agent's instance.

        Parameters
        ----------
        agent : Agent or subclass
            The agent to wrap.
        """
        super().__init__()
        self.agent = agent

    @property
    def unwrapped(self) -> Agent[SymType]:
        """'Returns the original agent of the wrapper."""
        return self.agent.unwrapped

    def is_wrapped(self, wrapper_type: Type["Wrapper[SymType]"]) -> bool:
        """Gets whether the agent instance is wrapped or not by the wrapper type.

        Parameters
        ----------
        wrapper_type : type of Wrapper
            Type of wrapper to check if the agent is wrapped with.

        Returns
        -------
        bool
            `True` if wrapped by an instance of `wrapper_type`; `False`, otherwise.
        """
        if isinstance(self, wrapper_type):
            return True
        return self.agent.is_wrapped(wrapper_type)

    def __getattr__(self, name: str) -> Any:
        """Reroutes attributes to the wrapped agent instance."""
        if name.startswith("_"):
            raise AttributeError(f"Accessing private attribute '{name}' is prohibited.")
        return getattr(self.agent, name)

    def __str__(self) -> str:
        """Returns the wrapped agent string."""
        return f"<{self.__class__.__name__}{self.agent.__str__()}>"

    def __repr__(self) -> str:
        """Returns the wrapped agent representation."""
        return f"<{self.__class__.__name__}{self.agent.__repr__()}>"


class LearningWrapper(Wrapper[SymType], Generic[SymType, ExpType]):
    """Identical to `Wrapper`, but for learning agents."""

    def __init__(self, agent: LearningAgent[SymType, ExpType]) -> None:
        super().__init__(agent)
        self.agent: LearningAgent[SymType, ExpType]
        self.establish_callback_hooks()

    @property
    def unwrapped(self) -> LearningAgent[SymType, ExpType]:
        return self.agent.unwrapped

    def establish_callback_hooks(self) -> None:
        """Similar to agents, this method must be used to perform the connections
        between callbacks and any invokable method (hook). If the wrapper has no hooks,
        then this method does nothing"""

    def hook_callback(self, *args, **kwargs) -> None:
        """See `LearningAgent.hook_callback`."""
        self.agent.hook_callback(*args, **kwargs)

    def __setstate__(
        self,
        state: Union[
            None, Dict[str, Any], Tuple[Optional[Dict[str, Any]], Dict[str, Any]]
        ],
    ) -> None:
        """When setting the state of the wrapper, it makes sure to avoid hooks with a
        previous copy of the agent, and re-establishes them again (otherwise, the new
        wrapper copy will call hooks to callbacks of the old agent)."""
        if isinstance(state, tuple) and len(state) == 2:
            state, slotstate = state
        else:
            slotstate = None
        if state is not None:
            # remove wrapped methods for callbacks due to this wrapper
            # (otherwise, new copies will still be calling the old one)."""
            for name in ("on_update", "on_episode_end", "on_env_step"):
                state.pop(name, None)  # type: ignore[union-attr]
            self.__dict__.update(state)  # type: ignore[arg-type]
        if slotstate is not None:
            for key, value in slotstate.items():
                setattr(self, key, value)
        # re-perform hooks
        self.establish_callback_hooks()
