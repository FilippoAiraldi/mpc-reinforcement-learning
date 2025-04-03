import sys
from collections.abc import Collection, Iterable, Iterator
from itertools import chain
from typing import Any, Generic, Literal, Optional, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Solution, wrappers
from csnlp.core.cache import invalidate_caches_of
from csnlp.util.io import SupportsDeepcopyAndPickle
from csnlp.wrappers import Mpc
from gymnasium import Env
from gymnasium.spaces import Box

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from ...core.callbacks import AgentCallbackMixin
from ...core.exploration import ExplorationStrategy, NoExploration
from ...core.warmstart import WarmStartStrategy
from ...util.named import Named
from ...util.seeding import RngType, mk_seed

SymType = TypeVar("SymType", cs.SX, cs.MX)
ActType: TypeAlias = Union[npt.ArrayLike, dict[str, npt.ArrayLike]]
ObsType: TypeAlias = Union[npt.ArrayLike, dict[str, npt.ArrayLike]]


def _update_dicts(sinks: Iterable[dict], source: dict) -> Iterator[dict]:
    """Internal utility for updating dicts `sinks` with one `source`."""
    for sink in sinks:
        sink.update(source)
        yield sink


class Agent(Named, SupportsDeepcopyAndPickle, AgentCallbackMixin, Generic[SymType]):
    r"""Simple MPC-based agent with a fixed (i.e., non-learnable) MPC controller.

    In this agent, the MPC controller parametrized in :math:`\theta` is used as policy
    provider, as well as to provide the value function :math:`V_\theta(s)` and quality
    function :math:`Q_\theta(s,a)`, where :math:`s` and :math:`a` are the state of the
    environment and a generic action, respectively. Since it only supports a fixed
    parametrization, this class does not use any RL or other learning method to improve
    its MPC policy.

    Parameters
    ----------
    mpc : :class:`csnlp.wrappers.Mpc`
        The MPC controller used as policy provider by this agent. The instance is
        modified in place to create the approximations of the state function
        :math:`V_\theta(s)` and action value function :math:`Q_\theta(s,a)`, so it is
        recommended not to modify it further after initialization of the agent.
        Moreover, some parameter and constraint names will need to be created, so an
        error is thrown if these names are already in use in the mpc.
    fixed_parameters : dict of (str, array_like) or collection of, optional
        A dict (or collection of dict, in case of the ``mpc`` wrapping an underlying
        :class:`csnlp.multistart.MultistartNlp` instance) whose keys are the names of
        the MPC parameters and the values are their corresponding values. Use this to
        specify fixed parameters, that is, non-learnable. If ``None``, then no fixed
        parameter is assumed.
    exploration : :class:`core.exploration.ExplorationStrategy`, optional
        Exploration strategy for inducing exploration in the online MPC policy. By
        default ``None``, in which case :class:`core.exploration.NoExploration` is used.
    warmstart : "last" or "last-successful" or WarmStartStrategy, optional
        The warmstart strategy for the MPC's NLP. If ``"last-successful"``, the last
        successful solution is used to warmstart the solver for the next iteration. If
        ``"last"``, the last solution is used, regardless of success or failure.
        Furthermore, an instance of :class:`core.warmstart.WarmStartStrategy` can
        be passed to specify a strategy for generating multiple warmstart points for the
        MPC's NLP instance. This is useful to generate multiple initial conditions for
        highly non-convex, nonlinear problems. This feature can only be used with an
        MPC that has an underlying multistart NLP problem (see :mod:`csnlp.multistart`).
    use_last_action_on_fail : bool, optional
        In case the MPC solver fails
         - if ``False``, the action from the last solver's iteration is returned anyway
           (though suboptimal)
         - if ``True``, the action from the last successful call to the MPC is returned
           instead (if the MPC has been solved at least once successfully).

        By default, ``False``.
    remove_bounds_on_initial_action : bool, optional
        When ``True``, the upper and lower bounds on the initial action are removed in
        the action-value function approximator :math:`Q_\theta(s,a)` since the first
        action is constrained to be equal to the provided action :math:`a`. This is
        useful to avoid issues in the LICQ of the NLP. However, it can lead to numerical
        problems. By default, ``False``.
    name : str, optional
        Name of the agent. If ``None``, one is automatically created from a counter of
        the class' instancies.

    Raises
    ------
    ValueError
        Raises if
         - the given ``mpc`` has no control action as optimization variable
         - the reserved parameter and constraint names are already in use (see
           :attr:`cost_perturbation_parameter`, :attr:`init_action_parameter` and
           :attr:`init_action_constraint`)
         - a multistart ``mpc`` is given, but the warmstart strategy ``warmstart`` asks
           for an incompatible number of starting points to be generated
         - a warmstart strategy ``warmstart`` is given, but the ``mpc`` does not have an
           underlying multistart NLP problem, so it cannot handle multiple starting
           points (see :attr:`csnlp.Nlp.is_multi` and
           :attr:`csnlp.multistart.MultistartNlp.is_multi`).
    """

    cost_perturbation_method = "normal"
    r"""The name of the method from :class`numpy.random.Generator` to be used to
    generate perturbations of the cost function in the state value function
    :math:`V_\theta(s)`."""

    cost_perturbation_parameter = "cost_perturbation"
    r"""The name of the parameter to be added to the original ``mpc`` problem for
    perturbing the state value function :math:`V_\theta(s)`."""

    init_action_parameter = "a_init"
    r"""Name of the parameter to be added to the original ``mpc`` problem for
    constraining the first action to be equal to :math:`a` in the action value function
    :math:`Q_\theta(s,a)`."""

    init_action_constraint = init_action_parameter
    r"""Name of the equality constraint to be added to the original ``mpc`` problem for
    constraining the first action to be equal to :math:`a` in the action value function
    :math:`Q_\theta(s,a)`."""

    def __init__(
        self,
        mpc: Mpc[SymType],
        fixed_parameters: Union[
            None, dict[str, npt.ArrayLike], Collection[dict[str, npt.ArrayLike]]
        ] = None,
        exploration: Optional[ExplorationStrategy] = None,
        warmstart: Union[
            Literal["last", "last-successful"], WarmStartStrategy
        ] = "last-successful",
        use_last_action_on_fail: bool = False,
        remove_bounds_on_initial_action: bool = False,
        name: Optional[str] = None,
    ) -> None:
        if isinstance(warmstart, str):
            warmstart = WarmStartStrategy(warmstart)
        ws_points = warmstart.n_points
        if mpc.is_multi and ws_points != 0 and mpc.nlp.starts - ws_points not in (0, 1):
            raise ValueError(
                "A multistart MPC was given with {mpc.nlp.starts} multistarts, but the "
                f"given warmstart strategy asks for {ws_points} starting points. "
                "Expected either 0 warmstart points (i.e., it is disabled), or the same"
                " number as MPC's multistarts, or at most one less."
            )
        elif not mpc.is_multi and ws_points > 0:
            raise ValueError(
                "Got a warmstart strategy with more than 0 starting points, but the "
                "given does not have an underlying multistart NLP problem."
            )
        Named.__init__(self, name)
        SupportsDeepcopyAndPickle.__init__(self)
        AgentCallbackMixin.__init__(self)
        self._fixed_pars = fixed_parameters
        if exploration is None:
            exploration = NoExploration()
        self._exploration = exploration
        self._warmstart = warmstart
        self._last_action_on_fail = use_last_action_on_fail
        self._last_solution: Optional[Solution[SymType]] = None
        self._last_action: Optional[cs.DM] = None
        self._V, self._Q = self._setup_V_and_Q(mpc, remove_bounds_on_initial_action)
        self._post_setup_V_and_Q()

    @property
    def unwrapped(self) -> "Agent":
        """Gets the underlying wrapped instance of an agent. In this case, since the
        agent is not wrapped at all, returns itself."""
        return self

    def is_wrapped(self, *args: Any, **kwargs: Any) -> bool:
        """Gets whether the agent instance is wrapped or not by the wrapper type.

        Returns
        -------
        bool
            A flag indicating whether the agent is wrapped or not.
        """
        return False

    @property
    def V(self) -> Mpc[SymType]:
        r"""Gets the MPC function approximation of the state value function
        :math:`V_\theta(s)`."""
        return self._V

    @property
    def Q(self) -> Mpc[SymType]:
        r"""Gets the MPC function approximation of the action value function
        :math:`Q_\theta(s,a)`."""
        return self._Q

    @property
    def fixed_parameters(
        self,
    ) -> Union[None, dict[str, npt.ArrayLike], Collection[dict[str, npt.ArrayLike]]]:
        """Gets the fixed parameters of the MPC controller, i.e., the non-learnable
        ones.

        Returns
        -------
        ``None`` or dict of (str, array_like), or collection of
            The returned object can be either
             - ``None``, if the MPC controller has no fixed parameters
             - a dict whose keys are the names of the MPC parameters and the values are
               their corresponding values, when the MPC controller wraps an instance of
               :class:`csnlp.Nlp`, or it wraps an instance of
               :class:`csnlp.multistart.MultistartNlp` but the same set of parameters
               is meant to be used for all scenarios
             - a collection of such dictionaries, when the MPC controller wraps an
               instance of :class:`csnlp.multistart.MultistartNlp` and different
               parameters are meant to be used for each scenario.
        """
        return self._fixed_pars

    @property
    def exploration(self) -> ExplorationStrategy:
        r"""Gets the exploration strategy used within this agent to perturb the
        policy provided by the MPC controller via :math:`V_\theta(s)`."""
        return self._exploration

    @property
    def warmstart(self) -> WarmStartStrategy:
        """Gets the warmstart strategy used within this agent. This strategy is used to
        generate the initial guess for the solver to optimize the MPC's NLP."""
        return self._warmstart

    def reset(self, seed: RngType = None) -> None:
        """Resets the agent. This includes resetting the warmstart strategy, the
        exploration strategy, and the some internal variables of the agent.

        Parameters
        ----------
        seed : RngType, optional
            The seed to reset the  :class:`numpy.random.Generator` instances. By
            default, ``None``.
        """
        self._last_solution = None
        self._last_action = None
        self.warmstart.reset(seed)
        self.exploration.reset(seed)

    def _solve_mpc(
        self,
        mpc: Mpc[SymType],
        state: Union[npt.ArrayLike, dict[str, npt.ArrayLike]],
        action: Union[None, npt.ArrayLike, dict[str, npt.ArrayLike]] = None,
        perturbation: Optional[npt.ArrayLike] = None,
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        store_solution: bool = True,
    ) -> Solution[SymType]:
        r"""Solves the agent's specific MPC optimal control problem.

        Parameters
        ----------
        mpc : Mpc
            The MPC problem to solve, either :attr:`V` or :attr:`Q`.
        state : array_like or dict of (str, array_like)
            The initial state at which to evaluate the MPC policy, i.e., :math:`s` in
            :math:`V_\theta(s)` or :math:`Q_\theta(s,a)`. It can be either a 1D array
            representing the value of all initial states of the MPC, concatenated.
            Otherwise, a dict whose keys are the names of each state, and values are
            their numerical initial state values.
        action : array_like or dict of (str, array_like), optional
            Same for ``state``, but for the action, i.e., the initial action at which to
            evaluate the MPC action value function, i.e., :math:`a` in
            :math:`Q_\theta(s,a)`. Obviously, it is only pertinent if ``mpc`` is
            :attr:`Q`, while it should be ``None`` for :attr:`V`.
        perturbation : array_like, optional
            The **gradient-based** cost perturbation used to induce exploration in
            :math:`V_\theta(s)`. Should be ``None`` for :math:`Q_\theta(s,a)`, or in
            case of other types of exploration are used.
        vals0 : dict of (str, array_like) or iterable of, optional
            A dict (or an iterable of dict, in case of
            :class:`csnlp.multistart.MultistartNlp` is used), whose
            keys are the names of the MPC variables, and values are the numerical
            initial values of each variable. Use this argument to warmstart the MPC.
            If ``None``, and a previous solution (possibly, successful) is available,
            the MPC solver is automatically warmstarted. If an iterable is passed
            instead, the warmstarting strategy is bypassed.
        store_solution : bool, optional
            By default, the MPC solution is stored accordingly to the :attr:`warmstart`
            strategy. If set to ``False``, this flag allows to disable the behaviour for
            this particular solution.

        Returns
        -------
        Solution
            The solution of the MPC.
        """
        # convert state keys into initial state keys (i.e., with "_0")
        if isinstance(state, dict):
            x0_dict = {f"{k}_0": v for k, v in state.items()}
        else:
            mpcstates = mpc.initial_states
            if len(mpcstates) == 1:
                states = (state,)
            else:
                cumsizes = np.cumsum([s.shape[0] for s in mpcstates.values()][:-1])
                states = np.split(np.asarray(state), cumsizes)
            x0_dict = dict(zip(mpcstates.keys(), states))

        # convert action dict to vector if not None
        if action is None:
            u0_vec = None
        elif isinstance(action, dict):
            u0_vec = cs.vertcat(*(action[k] for k in mpc.actions.keys()))
        else:
            u0_vec = action

        # merge (initial) state, action and perturbation in unique dict
        additional_pars = x0_dict
        if u0_vec is not None:
            additional_pars[self.init_action_parameter] = u0_vec
        if self.cost_perturbation_parameter in mpc.parameters:
            additional_pars[self.cost_perturbation_parameter] = (
                0 if perturbation is None else perturbation
            )

        # create pars and vals0
        pars = self._get_parameters()
        if pars is None:
            pars = additional_pars
        elif isinstance(pars, dict):
            pars.update(additional_pars)
        else:  # iterable of dict
            pars = _update_dicts(pars, additional_pars)
        if vals0 is None and self._last_solution is not None:
            vals0 = self._last_solution.vals

        # use the warmstart strategy to generate multiple initial points for the NLP if
        # the NLP supports multi and `vals0` is not already an iterable of dict
        if mpc.is_multi and (vals0 is None or isinstance(vals0, dict)):
            more_vals0s = self._warmstart.generate(vals0)
            if mpc.nlp.starts > self._warmstart.n_points:
                # the difference between these two has been checked to be at most one,
                # meaning we can include `vals0` itself
                more_vals0s = chain((vals0,), more_vals0s)
            vals0 = more_vals0s

        # solve and store solution
        sol = mpc(pars, vals0)
        if store_solution and (self._warmstart.store_always or sol.success):
            self._last_solution = sol
        return sol

    def state_value(
        self,
        state: Union[npt.ArrayLike, dict[str, npt.ArrayLike]],
        deterministic: bool = False,
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        action_space: Optional[Box] = None,
        **kwargs,
    ) -> tuple[cs.DM, Solution[SymType]]:
        r"""Computes the MPC-based state value function approximation
        :math:`V_\theta(s)`.

        Parameters
        ----------
        state : array_like or dict of (str, array_like)
            The initial state at which to evaluate the MPC approximation of the state
            value function, i.e., :math:`s` in :math:`V_\theta(s)`. It can be either a
            1D array representing the value of all initial states of the MPC,
            concatenated. Otherwise, a dict whose keys are the names of each state, and
            values are their numerical initial state values.
        deterministic : bool, optional
            If ``False``, the MPC controller is perturbed according to the
            :attr:`exploration` strategy to induce some exploratory behaviour.
            Otherwise, no perturbation is performed. By default, ``False``.
        vals0 : dict of (str, array_like) or iterable of, optional
            A dict (or an iterable of dict, in case of
            :class:`csnlp.multistart.MultistartNlp` is used), whose
            keys are the names of the MPC variables, and values are the numerical
            initial values of each variable. Use this argument to warmstart the MPC.
            If ``None``, and a previous solution (possibly, successful) is available,
            the MPC solver is automatically warmstarted. If an iterable is passed
            instead, the warmstarting strategy is bypassed.
        action_space : gymnasium.spaces.Box, optional
            The action space of the environment the agent is being evaluated/trained on.
            If not ``None``, it is used in case an additive exploration perturbation is
            summed to the action in order to clip it back into the action space.

        Returns
        -------
        casadi.DM
            The first optimal action according to the solution of the state value
            function, possibly perturbed by exploration noise, i.e.,

            .. math:: u_0^\star = \arg\min_{u} V_\theta(s)

        Solution
            The solution of the MPC approximation :math:`V_\theta(s)` at the given
            state.
        """
        V = self._V
        exploration = self._exploration
        exploration_mode = exploration.mode
        na = V.na
        if deterministic or exploration_mode is None or not exploration.can_explore():
            pert = None
        else:
            pert = exploration.perturbation(self.cost_perturbation_method, size=(na, 1))
            assert np.shape(pert) == (na, 1), (
                f"Expected shape of perturbation to be ({na}, 1); got "
                f"{np.shape(pert)} instead."
            )

        grad_pert = pert if exploration_mode == "gradient-based" else None
        sol = self._solve_mpc(V, state, perturbation=grad_pert, vals0=vals0, **kwargs)
        action_opt = cs.vertcat(*(sol.vals[u][:, 0] for u in V.actions.keys()))

        if sol.success:
            self._last_action = action_opt
        elif self._last_action_on_fail and self._last_action is not None:
            action_opt = self._last_action

        if pert is not None and exploration_mode == "additive":
            action_opt_noisy = action_opt + pert
            if action_space is not None:
                lb = action_space.low.reshape(na, 1)
                ub = action_space.high.reshape(na, 1)
                action_opt_noisy = np.clip(action_opt_noisy, lb, ub)
                # if np.equal(action_opt_noisy, action_opt).all():
                #     action_opt_noisy = np.clip(action_opt - pert, lb, ub)
            action_opt = action_opt_noisy
        return action_opt, sol

    def action_value(
        self,
        state: Union[npt.ArrayLike, dict[str, npt.ArrayLike]],
        action: Union[npt.ArrayLike, dict[str, npt.ArrayLike]],
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        **kwargs,
    ) -> Solution[SymType]:
        r"""Computes the MPC-based action value function approximation
        :math:`Q_\theta(s,a)`.

        Parameters
        ----------
        state : array_like or dict of (str, array_like)
            The initial state at which to evaluate the action value function, i.e.,
            :math:`s` in :math:`Q_\theta(s,a)`. It can be either a 1D array representing
            the value of all initial states of the MPC, concatenated. Otherwise, a dict
            whose keys are the names of each state, and values are their numerical
            initial state values.
        action : array_like or dict of (str, array_like), optional
            Same for ``state``, but for the action, i.e., the initial action at which to
            evaluate the MPC action value function, i.e., :math:`a` in
            :math:`Q_\theta(s,a)`.
        vals0 : dict of (str, array_like) or iterable of, optional
            A dict (or an iterable of dict, in case of
            :class:`csnlp.multistart.MultistartNlp` is used), whose
            keys are the names of the MPC variables, and values are the numerical
            initial values of each variable. Use this argument to warmstart the MPC.
            If ``None``, and a previous solution (possibly, successful) is available,
            the MPC solver is automatically warmstarted. If an iterable is passed
            instead, the warmstarting strategy is bypassed.

        Returns
        -------
        Solution
            The solution of the MPC approximation :math:`Q_\theta(s,a)` at the given
            state and action pair.
        """
        return self._solve_mpc(self._Q, state, action, vals0=vals0, **kwargs)

    def evaluate(
        self,
        env: Env[ObsType, ActType],
        episodes: int,
        deterministic: bool = True,
        seed: RngType = None,
        raises: bool = True,
        env_reset_options: Optional[dict[str, Any]] = None,
    ) -> npt.NDArray[np.floating]:
        r"""Evaluates the agent in a given environment.

        Parameters
        ----------
        env : Env[ObsType, ActType]
            The gym environment where to evaluate the agent in.
        episodes : int
            Number of evaluation episodes.
        deterministic : bool, optional
            Whether the agent should act deterministically, i.e., applying no
            exploration to the policy provided by the MPC. By default, ``True``.
        seed : None, int, array_like of ints, SeedSequence, BitGenerator, Generator
            Seed for the agent's and env's random number generator. By default ``None``.
        raises : bool, optional
            If ``True``, when any of the MPC solver runs fails, or when an update fails,
            the corresponding error is raised; otherwise, only a warning is raised.
        env_reset_options : dict, optional
            Additional information to specify how the environment is reset at each
            evalution episode (optional, depending on the specific environment).

        Returns
        -------
        array of doubles
            The cumulative returns (one return per evaluation episode).

        Raises
        ------
        MpcSolverError or MpcSolverWarning
            Raises if the MPC optimization solver fails and ``raises=True``.

        Notes
        -----
        After solving :math:`V_\theta(s)` for the current env's state `s`, the action
        is passed to the environment as the concatenation of the first optimal action
        variables of the MPC (see `csnlp.Mpc.actions`).
        """
        rng = np.random.default_rng(seed)
        self.reset(rng)
        returns = np.zeros(episodes)
        self.on_validation_start(env)

        for episode in range(episodes):
            state, _ = env.reset(seed=mk_seed(rng), options=env_reset_options)
            truncated, terminated, timestep = False, False, 0
            self.on_episode_start(env, episode, state)

            while not (truncated or terminated):
                action, sol = self.state_value(state, deterministic)
                if not sol.success:
                    self.on_mpc_failure(episode, timestep, sol.status, raises)

                state, r, truncated, terminated, _ = env.step(action)
                self.on_env_step(env, episode, timestep)

                returns[episode] += r
                timestep += 1
                self.on_timestep_end(env, episode, timestep)

            self.on_episode_end(env, episode, returns[episode])

        self.on_validation_end(env, returns)
        return returns

    def _setup_V_and_Q(
        self, mpc: Mpc[SymType], remove_bounds_on_initial_action: bool
    ) -> tuple[Mpc[SymType], Mpc[SymType]]:
        """Internal utility to setup the function approximators for the value function
        ``V(s)`` and the quality function ``Q(s,a)``."""
        na = mpc.na
        if na <= 0:
            raise ValueError(f"Expected Mpc with na>0; got na={na} instead.")

        # create V and Q function approximations
        V, Q = mpc, mpc.copy()
        V.unwrapped.name += "_V"
        Q.unwrapped.name += "_Q"

        # for Q, add the additional constraint on the initial action to be equal to a0,
        # and remove the now useless upper/lower bounds on the initial action
        a0 = Q.nlp.parameter(self.init_action_parameter, (na, 1))
        u0 = cs.vcat(mpc.first_actions.values())
        Q.nlp.constraint(self.init_action_constraint, u0, "==", a0)
        if remove_bounds_on_initial_action:
            for name, a in mpc.first_actions.items():
                na_ = a.size1()
                Q.nlp.remove_variable_bounds(name, "both", ((r, 0) for r in range(na_)))

        # for V, add the cost perturbation parameter (only if gradient-based)
        if self._exploration.mode == "gradient-based":
            perturbation = V.nlp.parameter(self.cost_perturbation_parameter, (na, 1))
            f = V.nlp.f
            if mpc.is_wrapped(wrappers.NlpScaling):
                f = mpc.scale(f)
            V.nlp.minimize(f + cs.dot(perturbation, u0))

        # invalidate caches for V and Q since some modifications have been done
        for nlp in (V, Q):
            nlp_ = nlp
            while nlp_ is not nlp_.unwrapped:
                invalidate_caches_of(nlp_)
                nlp_ = nlp_.nlp
            invalidate_caches_of(nlp_.unwrapped)
        return V, Q

    def _post_setup_V_and_Q(self) -> None:
        """Internal utility that is run after the creation of ``V`` and ``Q``, allowing
        for further customization in inheriting classes."""

    def _get_parameters(
        self,
    ) -> Union[None, dict[str, npt.ArrayLike], Collection[dict[str, npt.ArrayLike]]]:
        """Internal utility to retrieve parameters of the MPC in order to solve it.
        :class:`Agent` has no learnable parameter, so only fixed parameters are
        returned."""
        return self._fixed_pars

    def __deepcopy__(self, memo: Optional[dict[int, list[Any]]] = None) -> "Agent":
        """Ensures that the copy has a new name."""
        y = super().__deepcopy__(memo)
        if hasattr(y, "name"):
            y.name += "_copy"
        return y
