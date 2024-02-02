from collections.abc import Collection, Iterable, Iterator
from typing import Any, Generic, Literal, Optional, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Solution, wrappers
from csnlp.core.cache import invalidate_caches_of
from csnlp.util.io import SupportsDeepcopyAndPickle
from csnlp.wrappers import Mpc
from gymnasium import Env
from typing_extensions import TypeAlias

from ...core.callbacks import AgentCallbackMixin
from ...core.exploration import ExplorationStrategy, NoExploration
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
    """Simple MPC-based agent with a fixed (i.e., non-learnable) MPC controller.

    In this agent, the MPC controller is used as policy provider, as well as to provide
    the value function `V(s)` and quality function `Q(s,a)`, where `s` and `a` are the
    state and action of the environment, respectively. However, this class does not use
    any RL method to improve its MPC policy."""

    cost_perturbation_method = "normal"
    cost_perturbation_parameter = "cost_perturbation"
    init_action_parameter = init_action_constraint = "a_init"

    def __init__(
        self,
        mpc: Mpc[SymType],
        fixed_parameters: Union[
            None, dict[str, npt.ArrayLike], Collection[dict[str, npt.ArrayLike]]
        ] = None,
        warmstart: Literal["last", "last-successful"] = "last-successful",
        use_last_action_on_fail: bool = False,
        remove_bounds_on_initial_action: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """Instantiates an agent with an MPC controller.

        Parameters
        ----------
        mpc : Mpc[casadi.SX or MX]
            The MPC controller used as policy provider by this agent. The instance is
            modified in place to create the approximations of the state function `V(s)`
            and action value function `Q(s,a)`, so it is recommended not to modify it
            further after initialization of the agent. Moreover, some parameter and
            constraint names will need to be created, so an error is thrown if these
            names are already in use in the mpc. These names are under the attributes
            `perturbation_parameter`, `action_parameter` and `action_constraint`.
        fixed_parameters : dict[str, array_like] or collection of, optional
            A dict (or collection of dict, in case of `csnlp.MultistartNlp`) whose keys
            are the names of the MPC parameters and the values are their corresponding
            values. Use this to specify fixed parameters, that is, non-learnable. If
            `None`, then no fixed parameter is assumed.
        warmstart: 'last' or 'last-successful', optional
            The warmstart strategy for the MPC's NLP. If 'last-successful', the last
            successful solution is used to warm start the solver for the next iteration.
            If 'last', the last solution is used, regardless of success or failure.
        use_last_action_on_fail : bool, optional
            When `True`, if the MPC solver fails in solving the state value function
            `V(s)`, the last successful action is returned. When `False`, the action
            from the last MPC iteration is returned instead. By default, `False`.
        remove_bounds_on_initial_action : bool, optional
            When `True`, the upper and lower bounds on the initial action are removed in
            the action-value function approximator Q(s,a) since the first action is
            constrained to be equal to the initial action. This is useful to avoid
            issues in the LICQ of the NLP. However, it can lead to numerical problems.
            By default, `False`.
        name : str, optional
            Name of the agent. If `None`, one is automatically created from a counter of
            the class' instancies.

        Raises
        ------
        ValueError
            Raises if the given mpc has no control action as optimization variable; or
            if the required parameter and constraint names are already specified in the
            mpc.
        """
        Named.__init__(self, name)
        SupportsDeepcopyAndPickle.__init__(self)
        AgentCallbackMixin.__init__(self)
        self._fixed_pars = fixed_parameters
        self._exploration: ExplorationStrategy = NoExploration()
        self._store_last_successful = warmstart == "last-successful"
        self._last_action_on_fail = use_last_action_on_fail
        self._last_solution: Optional[Solution[SymType]] = None
        self._last_action: Optional[cs.DM] = None
        self._V, self._Q = self._setup_V_and_Q(mpc, remove_bounds_on_initial_action)
        self._post_setup_V_and_Q()

    @property
    def unwrapped(self) -> "Agent":
        """Gets the underlying wrapped instance of an agent."""
        return self

    def is_wrapped(self, *args: Any, **kwargs: Any) -> bool:
        """Gets whether the agent instance is wrapped or not by the wrapper type."""
        return False

    @property
    def V(self) -> Mpc[SymType]:
        """Gets the MPC function approximation of the state value function `V(s)`."""
        return self._V

    @property
    def Q(self) -> Mpc[SymType]:
        """Gets the MPC function approximation of the action value function `Q(s,a)`."""
        return self._Q

    @property
    def fixed_parameters(
        self,
    ) -> Union[None, dict[str, npt.ArrayLike], Collection[dict[str, npt.ArrayLike]]]:
        """Gets the fixed parameters of the MPC controller (i.e., non-learnable). Can be
        an iterable in the case of `csnlp.MultistartNpl`, where multiple parameters can
        be specified, one for each scenario in the MPC scheme."""
        return self._fixed_pars

    @property
    def exploration(self) -> ExplorationStrategy:
        """Gets the exploration strategy used within this agent."""
        return self._exploration

    def reset(self, seed: RngType = None) -> None:
        """Resets the agent's internal variables and exploration's RNG."""
        self._last_solution = None
        self._last_action = None
        if hasattr(self.exploration, "reset"):
            self.exploration.reset(seed)

    def solve_mpc(
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
        """Solves the agent's specific MPC optimal control problem.

        Parameters
        ----------
        mpc : Mpc
            The MPC problem to solve, either `Agent.V` or `Agent.Q`.
        state : array_like or dict[str, array_like]
            A 1D array representing the value of all states of the MPC, concatenated.
            Otherwise, a dict whose keys are the names of each state, and values are
            their numerical values.
        action : array_like or dict[str, array_like], optional
            Same for `state`, for the action. Only valid if evaluating the action value
            function `Q(s,a)`. For this reason, it can be `None` for `V(s)`.
        perturbation : array_like, optional
            The cost perturbation used to induce exploration in `V(s)`. Can be `None`
            for `Q(s,a)`.
        vals0 : dict[str, array_like] or iterable of, optional
            A dict (or an iterable of dict, in case of `csnlp.MultistartNlp`), whose
            keys are the names of the MPC variables, and values are the numerical
            initial values of each variable. Use this to warm-start the MPC. If `None`,
            and a previous solution (possibly, successful) is available, the MPC solver
            is automatically warm-started.
        store_solution : bool, optional
            By default, the mpc solution is stored accordingly to the `warmstart`
            strategy. If set to `False`, this flag allows to disable the behaviour for
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
                states = np.split(state, cumsizes)
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

        # solve and store solution
        sol = mpc(pars, vals0)
        if store_solution and (not self._store_last_successful or sol.success):
            self._last_solution = sol
        return sol

    def state_value(
        self,
        state: Union[npt.ArrayLike, dict[str, npt.ArrayLike]],
        deterministic: bool = False,
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        **kwargs,
    ) -> tuple[cs.DM, Solution[SymType]]:
        """Computes the state value function `V(s)` approximated by the MPC.

        Parameters
        ----------
        state : array_like or dict[str, array_like]
            The state `s` at which to evaluate the value function `V(s)`. Can either be
            a dict of state names and values, or a single concatenated column vector of
            all the states.
        deterministic : bool, optional
            If `True`, the cost of the MPC is perturbed according to the exploration
            strategy to induce some exploratory behaviour. Otherwise, no perturbation is
            performed. By default, `deterministic=False`.
        vals0 : dict[str, array_like] or iterable of, optional
            A dict (or an iterable of dict, in case of `csnlp.MultistartNlp`), whose
            keys are the names of the MPC variables, and values are the numerical
            initial values of each variable. Use this to warm-start the MPC. If `None`,
            and a previous solution (possibly, successful) is available, the MPC solver
            is automatically warm-started.

        Returns
        -------
        casadi.DM
            The first optimal action according to the solution of `V(s)`.
        Solution
            The solution of the MPC approximating `V(s)` at the given state.
        """
        if deterministic or not self._exploration.can_explore():
            pert = None
        else:
            pert = self._exploration.perturbation(
                self.cost_perturbation_method,
                size=self.V.parameters[self.cost_perturbation_parameter].shape,
            )
        sol = self.solve_mpc(self._V, state, perturbation=pert, vals0=vals0, **kwargs)
        first_action = cs.vertcat(*(sol.vals[u][:, 0] for u in self._V.actions.keys()))

        if sol.success:
            self._last_action = first_action
        elif self._last_action_on_fail and self._last_action is not None:
            first_action = self._last_action

        return first_action, sol

    def action_value(
        self,
        state: Union[npt.ArrayLike, dict[str, npt.ArrayLike]],
        action: Union[npt.ArrayLike, dict[str, npt.ArrayLike]],
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        **kwargs,
    ) -> Solution[SymType]:
        """Computes the action value function `Q(s,a)` approximated by the MPC.

        Parameters
        ----------
        state : array_like or dict[str, array_like]
            The state `s` at which to evaluate the value function `Q(s,a)`. Can either
            be a dict of state names and values, or a single concatenated column vector
            of all the states.
        action : array_like or dict[str, array_like]
            The action `a` at which to evaluate the value function `Q(s,a)`. Can either
            be a dict of action names and values, or a single concatenated column vector
            of all the actions.
        vals0 : dict[str, array_like] or iterable of, optional
            A dict (or an iterable of dict, in case of `csnlp.MultistartNlp`), whose
            keys are the names of the MPC variables, and values are the numerical
            initial values of each variable. Use this to warm-start the MPC. If `None`,
            and a previous solution (possibly, successful) is available, the MPC solver
            is automatically warm-started.

        Returns
        -------
        Solution
            The solution of the MPC approximating `Q(s,a)` at given state and action.
        """
        return self.solve_mpc(self._Q, state, action, vals0=vals0, **kwargs)

    def evaluate(
        self,
        env: Env[ObsType, ActType],
        episodes: int,
        deterministic: bool = True,
        seed: RngType = None,
        raises: bool = True,
        env_reset_options: Optional[dict[str, Any]] = None,
    ) -> npt.NDArray[np.floating]:
        """Evaluates the agent in a given environment.

        Note: after solving `V(s)` for the current state `s`, the action is computed and
        passed to the environment as the concatenation of the first optimal action
        variables of the MPC (see `csnlp.Mpc.actions`).

        Parameters
        ----------
        env : Env[ObsType, ActType]
            A gym environment where to test the agent in.
        episodes : int
            Number of evaluation episodes.
        deterministic : bool, optional
            Whether the agent should act deterministically; by default, `True`.
        seed : None, int, array_like[ints], SeedSequence, BitGenerator, Generator
            Agent's and each env's RNG seed.
        raises : bool, optional
            If `True`, when any of the MPC solver runs fails, or when an update fails,
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
            Raises if the MPC optimization solver fails and `warns_on_exception=False`.
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
        V(s) and the quality function Q(s,a)."""
        na = mpc.na
        if na <= 0:
            raise ValueError(f"Expected Mpc with na>0; got na={na} instead.")

        # create V and Q function approximations
        V, Q = mpc, mpc.copy()
        V.unwrapped.name += "_V"
        Q.unwrapped.name += "_Q"
        a0 = Q.nlp.parameter(self.init_action_parameter, (na, 1))
        perturbation = V.nlp.parameter(self.cost_perturbation_parameter, (na, 1))

        u0 = cs.vcat(mpc.first_actions.values())
        f = V.nlp.f
        if mpc.is_wrapped(wrappers.NlpScaling):
            f = mpc.scale(f)

        V.nlp.minimize(f + cs.dot(perturbation, u0))
        Q.nlp.constraint(self.init_action_constraint, u0, "==", a0)

        # remove upper/lower bound on initial action in Q, since it is constrained as a0
        if remove_bounds_on_initial_action:
            for name, a in mpc.first_actions.items():
                na_ = a.size1()
                Q.nlp.remove_variable_bounds(name, "both", ((r, 0) for r in range(na_)))

        # invalidate caches for V and Q since some modifications have been done
        for nlp in (V, Q):
            nlp_ = nlp
            while nlp_ is not nlp_.unwrapped:
                invalidate_caches_of(nlp_)
                nlp_ = nlp_.nlp
            invalidate_caches_of(nlp_.unwrapped)
        return V, Q

    def _post_setup_V_and_Q(self) -> None:
        """Internal utility that is run after the creation of V and Q, allowing for
        further customization in inheriting classes."""
        # warn user of any constraints that linearly includes x0 and u0 (aside from
        # x(0)==s0 and u(0)==a0), which may thus lead to LICQ-failure
        # - u0 in Q should only appear in the 1st dynamics constraint and a0 con
        # - u0 in V should only appear in the 1st dynamics constraint and lbx/ubx
        # - x0 in V, Q should only appear in the 1st dynamics constraint and s0 con
        # for mpc in (self._V, self._Q):
        #     name = mpc.unwrapped.name[-1]
        #     u0 = cs.vcat(self._V.first_actions.values())
        #     x0 = cs.vvcat(self._V.first_states.values())
        #     con = cs.vertcat(mpc.g, mpc.h, mpc.h_lbx, mpc.h_ubx)

        #     if mpc.unwrapped.sym_type.__name__ == "SX":
        #         nnz_con_u0 = len(set(cs.jacobian_sparsity(con, u0).get_triplet()[0]))
        #         nnz_con_x0 = len(set(cs.jacobian_sparsity(con, x0).get_triplet()[0]))
        #     else:
        #         nnz_con_u0 = len(  # computes the same as above, but for MX
        #             set(
        #                 chain.from_iterable(
        #                     cs.jacobian(con, a)[:, : a.size1()]
        #                     .sparsity()
        #                     .get_triplet()[0]
        #                     for a in mpc.actions.values()
        #                 )
        #             )
        #         )
        #         nnz_con_x0 = len(
        #             set(
        #                 chain.from_iterable(
        #                     cs.jacobian(con, s)[:, : s.size1()]
        #                     .sparsity()
        #                     .get_triplet()[0]
        #                     for s in mpc.states.values()
        #                 )
        #             )
        #         )

        #     nnz_exp_u0 = mpc.ns + (mpc.na * 2 if name == "V" else mpc.na)
        #     if nnz_con_u0 > nnz_exp_u0:
        #         warnings.warn(
        #             f"detected {nnz_con_u0} (expected {nnz_exp_u0}) constraints on "
        #             f"initial actions in {name}; make sure that the initial action is"
        #             "not overconstrained (LICQ may be compromised).",
        #             RuntimeWarning,
        #         )
        #     nnz_exp_x0 = mpc.ns * 2
        #     if nnz_con_x0 > nnz_exp_x0:
        #         warnings.warn(
        #             f"detected {nnz_con_x0} (expected {nnz_exp_x0}) constraints on "
        #             f"initial states in {name}; make sure that the initial state is "
        #             "not overconstrained (LICQ may be compromised).",
        #             RuntimeWarning,
        #         )

    def _get_parameters(
        self,
    ) -> Union[None, dict[str, npt.ArrayLike], Collection[dict[str, npt.ArrayLike]]]:
        """Internal utility to retrieve parameters of the MPC in order to solve it.
        `Agent` has no learnable parameter, so only fixed parameters are returned."""
        return self._fixed_pars

    def __deepcopy__(self, memo: Optional[dict[int, list[Any]]] = None) -> "Agent":
        """Ensures that the copy has a new name."""
        y = super().__deepcopy__(memo)
        if hasattr(y, "name"):
            y.name += "_copy"
        return y
