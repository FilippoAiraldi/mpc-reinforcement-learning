from typing import Any, Literal, Optional, TypeVar

import numpy as np
import numpy.typing as npt
from gymnasium import Env

from ...agents.common.learning_agent import ExpType, LearningAgent
from ...util.iters import bool_cycle
from ...util.seeding import RngType, mk_seed
from .wrapper import LearningWrapper, SymType

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Evaluate(LearningWrapper[SymType, ExpType]):
    """Wrapper for evaluating an agent during training.

    On the given hook and with the given frequency, this wrapper automatically evaluates
    the agent on the specified environment by calling the agent's method
    :meth:`mpcrl.Agent.evaluate`. The resulting evaluation returns are stored in the
    attribute :attr:`eval_returns`.

    Parameters
    ----------
    agent : LearningAgent
        The learning agent to be evaluated by the wrapper.
    eval_env : gymnasium.Env
        A gym environment to evaluate the agent in.
    hook : {"on_episode_end", "on_timestep_end", "on_update"}
        Hook to trigger the evaluation. The evaluation will be triggered every
        ``frequency`` invokations of the specified hook.
    frequency : int
        Frequency of the evaluation.
    n_eval_episodes : int, optional
        How many episodes to evaluate the agent for, by default ``1``.
    eval_immediately : bool, optional
        Whether to evaluate the agent immediately after the wrapper is created, by
        default ``False``.
    deterministic : bool, optional
        Whether the agent should act deterministically; by default, ``True``.
    seed : None, int, array_like of ints, SeedSequence, BitGenerator, Generator
        Agent's and each env's random seeds for the evaluation.
    raises : bool, optional
        If ``True``, when any of the MPC solver runs fails, or when an update fails, the
        corresponding error is raised; otherwise, only a warning is raised.
    env_reset_options : dict, optional
        Additional information to specify how the environment is reset at each evalution
        episode (optional, depending on the specific environment).
    fix_seed : bool, optional
        If ``True``, the seed is fixed and the same seed is used for all evaluations.
    """

    def __init__(
        self,
        agent: LearningAgent[SymType, ExpType],
        eval_env: Env[ObsType, ActType],
        hook: Literal["on_episode_end", "on_timestep_end", "on_update"],
        frequency: int,
        n_eval_episodes: int = 1,
        eval_immediately: bool = False,
        *,
        deterministic: bool = True,
        seed: RngType = None,
        raises: bool = True,
        env_reset_options: Optional[dict[str, Any]] = None,
        fix_seed: bool = False,
    ) -> None:
        self.eval_env = eval_env
        self._hook = hook
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        np_random = np.random.default_rng(seed)
        self._seed = mk_seed(np_random) if fix_seed else np_random
        self._raises = raises
        self._env_reset_options = env_reset_options
        self._keep_seed_fixed = fix_seed
        self._eval_cycle = bool_cycle(frequency)
        self.eval_returns: list[npt.NDArray[np.floating]] = []
        self._is_eval_in_progress = False
        super().__init__(agent)
        if eval_immediately:
            self._evaluate(force=True)

    def _evaluate(self, *_: Any, **kwargs: Any) -> None:
        # we always return if an evaluation is already in progress to avoid reentrancy
        if self._is_eval_in_progress:
            return
        # we return also if:
        #  - the agent is not training (we do not want this hook to fire on .evaluate)
        #  - or the cycle is not at the evaluation point
        # unless we have forced an evaluation in __init__ via `eval_immediately=True`
        forced = kwargs.get("force", False)
        unwrapped_agent = self.agent.unwrapped
        is_training = unwrapped_agent._is_training
        if not forced and (not is_training or not next(self._eval_cycle)):
            return

        self._is_eval_in_progress = True
        try:
            self.eval_returns.append(
                self.agent.evaluate(
                    self.eval_env,
                    self._n_eval_episodes,
                    self._deterministic,
                    self._seed,
                    self._raises,
                    self._env_reset_options,
                )
            )
        finally:
            self._is_eval_in_progress = False
            unwrapped_agent._is_training = is_training

    def _establish_callback_hooks(self) -> None:
        super()._establish_callback_hooks()
        self._hook_callback(repr(self), self._hook, self._evaluate)
