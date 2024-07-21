from abc import ABC
from typing import Any, Generic, Optional

from gymnasium import Env

from mpcrl.agents.common.agent import ActType, ObsType

from ...core.update import UpdateStrategy
from ...optim.gradient_free_optimizer import GradientFreeOptimizer
from .agent import SymType
from .learning_agent import LearningAgent


class GlobOptLearningAgent(LearningAgent[SymType, None], ABC, Generic[SymType]):
    """Class for learning agents that employ gradient-free Global Optimization
    strategies (e.g., Bayesian Optimization) to learn/improve the MPC policy.

    Contrary to :class:`RlLearningAgent`, this class does not require a discount factor,
    but requires an instance of a :class:`optim.GradientFreeOptimizer` that
    adheres to the ask-tell interface, i.e., it must implement the
    :func:`optim.GradientFreeOptimizer.ask` and
    :func:`optim.GradientFreeOptimizer.tell` methods.

    Parameters
    ----------
    optimizer : GradientFreeOptimizer
        An instance of :class:`optim.GradientFreeOptimizer` optimizer to ask for
        a suggested set of  parameters to try out, and later tell the value of the
        objective function for that suggested set of parameters.
    kwargs
        Additional arguments to be passed to :class:`LearningAgent`.

        Note: the following kwargs are not yet supported
         - ``"experience"``: usually, GO strategies do not require experience replay
         - ``"update_strategy"``: updates are fixed at the end of each episode.
    """

    def __init__(
        self, optimizer: GradientFreeOptimizer[SymType], **kwargs: Any
    ) -> None:
        for key in ("experience", "update_strategy"):
            if key in kwargs:
                raise ValueError(
                    f"{self.__class__.__name__} does not yet support `{key}` kwargs."
                )
        self.optimizer = optimizer
        super().__init__(update_strategy=UpdateStrategy(1, "on_episode_end"), **kwargs)
        self.optimizer.set_learnable_parameters(self._learnable_pars)

    def train(self, *args: Any, **kwargs: Any) -> Any:
        # prime the initial value of the learnable parameters by asking it to the
        # optimizer. This is the usual way to start Global Optimization strategies,
        # whereas in RL we usually start the user's given initial values.
        self.update()
        return super().train(*args, **kwargs)

    def train_one_episode(
        self,
        env: Env[ObsType, ActType],
        episode: int,
        init_state: ObsType,
        raises: bool = True,
    ) -> float:
        # simply evaluate the MPC on the env with the current set of parameters for one
        # episode, and then tell the optimizer the value of the objective function
        rewards = 0.0
        state = init_state
        truncated, terminated, timestep = False, False, 0

        while not (truncated or terminated):
            action, sol = self.state_value(state, False)
            if not sol.success:
                self.on_mpc_failure(episode, timestep, sol.status, raises)

            state, r, truncated, terminated, _ = env.step(action)
            self.on_env_step(env, episode, timestep)

            rewards += float(r)
            timestep += 1
            self.on_timestep_end(env, episode, timestep)

        values = (
            self._learnable_pars.value_as_dict
            if self.optimizer.prefers_dict
            else self._learnable_pars.value
        )
        self.optimizer.tell(values, rewards)
        return rewards

    def update(self) -> Optional[str]:
        # simply ask the optimizer for a new set of parameters, and then update the
        # current parameters with this new set
        theta_new, status = self.optimizer.ask()
        self._learnable_pars.update_values(theta_new)
        return status
