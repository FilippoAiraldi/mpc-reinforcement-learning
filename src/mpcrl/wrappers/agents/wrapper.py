from typing import Any, Callable, Generic, Union

from csnlp.util.io import SupportsDeepcopyAndPickle

from ...agents.common.agent import Agent, SymType
from ...agents.common.learning_agent import ExpType, LearningAgent
from ...core.callbacks import CallbackMixin


class Wrapper(SupportsDeepcopyAndPickle, CallbackMixin, Generic[SymType]):
    """Wraps an instance of :class:`mpcrl.Agent` to allow a modular transformation of
    its behaviour. This class is the base class for all wrappers. The subclass could
    override some methods to change the behavior of the original agent without touching
    the original code.

    Parameters
    ----------
    agent : Agent or subclass
        The agent to wrap.
    """

    def __init__(self, agent: Agent[SymType]) -> None:
        SupportsDeepcopyAndPickle.__init__(self)
        CallbackMixin.__init__(self)
        del self._hooks  # keep only one dict of hooks, i.e., the agent's one
        self.agent = agent
        self._hooked_callbacks: dict[str, list[str]] = {}
        self._establish_callback_hooks()

    @property
    def unwrapped(self) -> Union[Agent[SymType], LearningAgent[SymType, ExpType]]:
        """Returns the original agent wrapped by this wrapper."""
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
            ``True`` if wrapped by an instance of ``wrapper_type``; ``False``,
            otherwise.
        """
        if isinstance(self, wrapper_type):
            return True
        return self.agent.is_wrapped(wrapper_type)

    def _hook_callback(
        self, attachername: str, callbackname: str, func: Callable[..., None]
    ) -> None:
        # store the callback id for later removal via `detach_wrapper(s)`
        self._hooked_callbacks.setdefault(callbackname, []).append(attachername)
        self.unwrapped._hook_callback(attachername, callbackname, func)

    def detach_wrapper(
        self, recursive: bool = False
    ) -> Union[Agent[SymType], LearningAgent[SymType, ExpType], "Wrapper[SymType]"]:
        """Detaches the wrapper from the agent, returning the unwrapped agent. De facto,
        this method detaches all the hooks attached by this wrapper.

        Parameters
        ----------
        recursive : bool, optional
            If ``True``, detaches all the wrappers around the agent recursively.

        Returns
        -------
        Agent or LearningAgent or Wrapper
            Returns the wrapped agent (or other wrapper) instance. This instance has no
            more active hooks attached by this wrapper. If ``recursive=True``, all the
            wrappers around the agent and their hooks are detached.

        Notes
        -----
        Detaching a wrapper is useful when you want to make sure that the wrapper's
        hooked callback cannot modify the behaviour or data of the agent, for example,
        after learning is done and you want to save and evaluate your learnt policy.
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
        return (
            self.agent.detach_wrapper(True)
            if recursive
            and hasattr(self.agent, "detach_wrapper")
            and callable(self.agent.detach_wrapper)
            else self.agent
        )

    def __getattr__(self, name: str) -> Any:
        """Reroutes attributes to the wrapped agent instance."""
        if name.startswith("_"):
            raise AttributeError(f"Accessing private attribute '{name}' is prohibited.")
        return getattr(self.agent, name)

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}{self.agent.__str__()}>"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}{self.agent.__repr__()}>"


class LearningWrapper(Wrapper[SymType], Generic[SymType, ExpType]):
    """A :class:`Wrapper` subclass dedicated to wrapping instances of
    :class:`mpcrl.LearningAgent`."""

    def __init__(self, agent: LearningAgent[SymType, ExpType]) -> None:
        Wrapper.__init__(self, agent)
        self.agent: LearningAgent[SymType, ExpType]

    @property
    def unwrapped(self) -> LearningAgent[SymType, ExpType]:
        return self.agent.unwrapped
