from typing import Any, Generic, Type

from csnlp.util.io import SupportsDeepcopyAndPickle

from mpcrl.agents.agent import Agent, SymType
from mpcrl.agents.learning_agent import ExpType, LearningAgent
from mpcrl.core.callbacks import RemovesCallbackHooksInState


class Wrapper(SupportsDeepcopyAndPickle, RemovesCallbackHooksInState, Generic[SymType]):
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
        SupportsDeepcopyAndPickle.__init__(self)
        RemovesCallbackHooksInState.__init__(self)
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

    def hook_callback(self, *args, **kwargs) -> None:
        """See `LearningAgent.hook_callback`."""
        self.agent.hook_callback(*args, **kwargs)

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
        Wrapper.__init__(self, agent)
        self.agent: LearningAgent[SymType, ExpType]
        self.establish_callback_hooks()

    @property
    def unwrapped(self) -> LearningAgent[SymType, ExpType]:
        return self.agent.unwrapped
