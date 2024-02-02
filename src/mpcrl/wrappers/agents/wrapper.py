from typing import Any, Callable, Generic, Union

from csnlp.util.io import SupportsDeepcopyAndPickle

from ...agents.common.agent import Agent, SymType
from ...agents.common.learning_agent import ExpType, LearningAgent
from ...core.callbacks import CallbackMixin


class Wrapper(SupportsDeepcopyAndPickle, CallbackMixin, Generic[SymType]):
    """Wraps a learning agent to allow a modular transformation of its methods. This
    class is the base class for all wrappers. The subclass could override some methods
    to change the behavior of the original agent without touching the original code."""

    def __init__(self, agent: Agent[SymType]) -> None:
        """Wraps an agent's instance.

        Parameters
        ----------
        agent : Agent or subclass
            The agent to wrap.
        """
        SupportsDeepcopyAndPickle.__init__(self)
        CallbackMixin.__init__(self)
        del self._hooks  # only keep one dict of hooks, i.e., the agent's one
        self.agent = agent
        self._hooked_callbacks: dict[str, list[str]] = {}
        self.establish_callback_hooks()

    @property
    def unwrapped(self) -> Union[Agent[SymType], LearningAgent[SymType, ExpType]]:
        """'Returns the original agent of the wrapper."""
        return self.agent.unwrapped

    def is_wrapped(self, wrapper_type: type["Wrapper[SymType]"]) -> bool:
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

    def hook_callback(
        self, attachername: str, callbackname: str, func: Callable[..., None]
    ) -> None:
        """See `LearningAgent.hook_callback`."""
        # store the callback id for later removal via `detach_wrapper(s)`
        self._hooked_callbacks.setdefault(callbackname, []).append(attachername)
        self.unwrapped.hook_callback(attachername, callbackname, func)

    def detach_wrapper(
        self,
    ) -> Union[Agent[SymType], LearningAgent[SymType, ExpType], "Wrapper[SymType]"]:
        """Detaches the wrapper from the agent, returning the wrapped agent. De facto,
        this method detaches all the hooks attached by this wrapper.

        Returns
        -------
        Agent or Wrapper
            Returns the wrapped agent (or other wrapper) instance. This instance has no
            more active hooks attached by this wrapper.
        """
        hooks = self.unwrapped._hooks
        hooked_callbacks = self._hooked_callbacks

        # for each callback type, remove the hooks attached by this wrapper
        for callbackname, attachernames in hooked_callbacks.items():
            hook_group = hooks[callbackname]
            for attachername in attachernames:
                hook_group.pop(attachername)

            # if the callback has no more hooks, remove it
            if not hook_group:
                hooks.pop(callbackname)

        # clear hooked callbacks tracking
        hooked_callbacks.clear()
        return self.agent

    def detach_wrappers(self) -> Union[Agent[SymType], LearningAgent[SymType, ExpType]]:
        """Similar to `detach_wrapper`, but detaches all wrappers around the agent.

        Returns
        -------
        Agent
            Returns the wrapped agent instance. This instance has no more active hooks
            attached by all the wrappers around it.
        """
        agent_ = self
        while hasattr(agent_, "detach_wrapper") and callable(agent_.detach_wrapper):
            agent_ = agent_.detach_wrapper()
        return agent_

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

    @property
    def unwrapped(self) -> LearningAgent[SymType, ExpType]:
        return self.agent.unwrapped
