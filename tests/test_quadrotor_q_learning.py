import unittest
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import (
    Any,
    Deque,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import casadi as cs
import numpy as np
from csnlp import Nlp, Solution
from csnlp.util.math import quad_form
from csnlp.wrappers import Mpc
from gymnasium.wrappers import TimeLimit
from scipy.linalg import cho_solve

from mpcrl import (
    ExperienceReplay,
    LearnableParameter,
    LearnableParametersDict,
    LstdQLearningAgent,
    MpcSolverError,
    UpdateError,
)
from mpcrl import exploration as E
from mpcrl import schedulers as S
from mpcrl.util.math import cholesky_added_multiple_identities
from mpcrl.wrappers import RecordUpdates

# ==================================================================================== #
# ---------------------------------- START OLD CODE ---------------------------------- #
# ==================================================================================== #


@dataclass
class QuadRotorEnvConfig:
    T: float = 0.1
    g: float = 9.81
    thrust_coeff: float = 1.4
    pitch_d: float = 10
    pitch_dd: float = 8
    pitch_gain: float = 10
    roll_d: float = 10
    roll_dd: float = 8
    roll_gain: float = 10
    winds: Dict[float, float] = field(default_factory=lambda: {1: 1.0, 2: 0.7, 3: 0.85})
    x0: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 3.5, 0, 0, 0, 0, 0, 0, 0])
    )
    xf: np.ndarray = field(
        default_factory=lambda: np.array([3, 3, 0.2, 0, 0, 0, 0, 0, 0, 0])
    )
    soft_constraints: bool = True
    x_bounds: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-0.5, 3.5],
                [-0.5, 3.5],
                [-0.175, 4],
                [-np.inf, np.inf],
                [-np.inf, np.inf],
                [-np.inf, np.inf],
                [np.deg2rad(-30), np.deg2rad(30)],
                [np.deg2rad(-30), np.deg2rad(30)],
                [-np.inf, np.inf],
                [-np.inf, np.inf],
            ]
        )
    )
    u_bounds: np.ndarray = field(
        default_factory=lambda: np.array(
            [[-np.pi, np.pi], [-np.pi, np.pi], [0, 2 * 9.81]]
        )
    )


class QuadRotorEnv:
    spec: Dict = None
    nx: int = 10
    nu: int = 3

    def __init__(self, config: Union[dict, QuadRotorEnvConfig] = None) -> None:
        config = init_config(config, QuadRotorEnvConfig)
        self.config = config

        # create dynamics matrices
        self._A, self._B, self._C, self._e = self.get_dynamics(
            g=config.g,
            thrust_coeff=config.thrust_coeff,
            pitch_d=config.pitch_d,
            pitch_dd=config.pitch_dd,
            pitch_gain=config.pitch_gain,
            roll_d=config.roll_d,
            roll_dd=config.roll_dd,
            roll_gain=config.roll_gain,
            winds=config.winds,
        )
        # weight for positional, control action usage and violation errors
        self._Wx = np.ones(self.nx)
        self._Wu = np.ones(self.nu)
        self._Wv = np.array([1e2, 1e2, 3e2, 3e2])

    @property
    def A(self) -> np.ndarray:
        return self._A.copy()

    @property
    def B(self) -> np.ndarray:
        return self._B.copy()

    @property
    def C(self) -> np.ndarray:
        return self._C.copy()

    @property
    def e(self) -> np.ndarray:
        return self._e.copy()

    @property
    def x(self) -> np.ndarray:
        return self._x.copy()

    @x.setter
    def x(self, val: np.ndarray) -> None:
        self._x = val.copy()

    def position_error(self, x: np.ndarray) -> float:
        return (np.square((x - self.config.xf)) * self._Wx).sum(axis=-1)

    def control_usage(self, u: np.ndarray) -> float:
        return (np.square(u) * self._Wu).sum(axis=-1)

    def constraint_violations(self, x: np.ndarray, u: np.ndarray) -> float:
        W = self._Wv
        return (
            W[0] * np.maximum(0, self.config.x_bounds[:, 0] - x).sum(axis=-1)
            + W[1] * np.maximum(0, x - self.config.x_bounds[:, 1]).sum(axis=-1)
            + W[2] * np.maximum(0, self.config.u_bounds[:, 0] - u).sum(axis=-1)
            + W[3] * np.maximum(0, u - self.config.u_bounds[:, 1]).sum(axis=-1)
        )

    def phi(self, alt: Union[float, np.ndarray]) -> np.ndarray:
        if isinstance(alt, np.ndarray):
            alt = alt.squeeze()
            assert alt.ndim == 1, "Altitudes must be a vector"

        return np.vstack([np.exp(-np.square(alt - h)) for h in self.config.winds])

    def reset(
        self,
        seed: int = None,
        x0: np.ndarray = None,
        xf: np.ndarray = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        seed_seq = np.random.SeedSequence(seed)
        self.np_random = np.random.Generator(np.random.PCG64(seed_seq))
        if x0 is None:
            x0 = self.config.x0
        if xf is None:
            xf = self.config.xf
        self.x = x0
        self.config.x0 = x0
        self.config.xf = xf
        self._n_within_termination = 0
        return self.x, {}

    def step(self, u: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        u = np.asarray(u).squeeze()  # in case a row or col was passed
        wind = (
            self._C
            @ self.phi(self.x[2])
            * self.np_random.uniform(
                low=[0, 0, -1, 0, 0, 0, -1, -1, 0, 0],
                high=[1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
            ).reshape(self.nx, 1)
        )
        self.x = (
            self._A @ self.x.reshape((-1, 1))
            + self._B @ u.reshape((-1, 1))
            + self._e
            + wind
        ).flatten()
        error = self.position_error(self.x)
        usage = self.control_usage(u)
        violations = self.constraint_violations(self.x, u)
        cost = float(error + usage + violations)
        return self.x, cost, False, False, {"error": error}

    def render(self):
        raise NotImplementedError("Render method unavailable.")

    def get_dynamics(
        self,
        g: Union[float, cs.SX],
        thrust_coeff: Union[float, cs.SX],
        pitch_d: Union[float, cs.SX],
        pitch_dd: Union[float, cs.SX],
        pitch_gain: Union[float, cs.SX],
        roll_d: Union[float, cs.SX],
        roll_dd: Union[float, cs.SX],
        roll_gain: Union[float, cs.SX],
        winds: Dict[float, float] = None,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[cs.SX, cs.SX, cs.SX],
    ]:
        T = self.config.T
        is_casadi = any(
            isinstance(o, (cs.SX, cs.MX, cs.DM))
            for o in [
                g,
                thrust_coeff,
                pitch_d,
                pitch_dd,
                pitch_gain,
                roll_d,
                roll_dd,
                roll_gain,
            ]
        )
        if is_casadi:
            diag = lambda o: cs.diag(cs.vertcat(*o))  # noqa: E731
            block = cs.blockcat
        else:
            diag = np.diag
            block = np.block
            assert winds is not None, "Winds are required to compute matrix C."
            nw = len(winds)
            wind_mag = np.array(list(winds.values()))
        A = T * block(
            [
                [np.zeros((3, 3)), np.eye(3), np.zeros((3, 4))],
                [np.zeros((2, 6)), np.eye(2) * g, np.zeros((2, 2))],
                [np.zeros((1, 10))],
                [np.zeros((2, 6)), -diag((pitch_d, roll_d)), np.eye(2)],
                [np.zeros((2, 6)), -diag((pitch_dd, roll_dd)), np.zeros((2, 2))],
            ]
        ) + np.eye(10)
        B = T * block(
            [
                [np.zeros((5, 3))],
                [0, 0, thrust_coeff],
                [np.zeros((2, 3))],
                [pitch_gain, 0, 0],
                [0, roll_gain, 0],
            ]
        )
        if not is_casadi:
            C = T * block(
                [
                    [wind_mag],
                    [wind_mag],
                    [wind_mag],
                    [np.zeros((3, nw))],
                    [wind_mag],
                    [wind_mag],
                    [np.zeros((2, nw))],
                ]
            )
        e = block([[np.zeros((5, 1))], [-T * g], [np.zeros((4, 1))]])
        return (A, B, e) if is_casadi else (A, B, C, e)


@dataclass(frozen=True)
class QuadRotorSolution:
    f: float
    vars: Dict[str, cs.SX]
    vals: Dict[str, np.ndarray]
    stats: Dict[str, Any]
    get_value: partial

    @property
    def status(self) -> str:
        return self.stats["return_status"]

    @property
    def success(self) -> bool:
        return self.stats["success"]

    def value(self, x: cs.SX) -> np.ndarray:
        return self.get_value(x)


class GenericMPC:
    def __init__(self, name: str = None) -> None:
        self.name = f"MPC{np.random.random()}" if name is None else name
        self.f: cs.SX = None  # objective
        self.vars: Dict[str, cs.SX] = {}
        self.pars: Dict[str, cs.SX] = {}
        self.cons: Dict[str, cs.SX] = {}
        self.p = cs.SX()
        self.x, self.lbx, self.ubx = cs.SX(), np.array([]), np.array([])
        self.lam_lbx, self.lam_ubx = cs.SX(), cs.SX()
        self.g, self.lbg, self.ubg = cs.SX(), np.array([]), np.array([])
        self.lam_g = cs.SX()
        self.h, self.lbh, self.ubh = cs.SX(), np.array([]), np.array([])
        self.lam_h = cs.SX()
        self.solver: cs.Function = None
        self.opts: Dict = None

    @property
    def ng(self) -> int:
        return self.g.shape[0]

    def add_par(self, name: str, *dims: int) -> cs.SX:
        assert name not in self.pars, f"Parameter {name} already exists."
        par = cs.SX.sym(name, *dims)
        self.pars[name] = par
        self.p = cs.vertcat(self.p, cs.vec(par))
        return par

    def add_var(
        self,
        name: str,
        *dims: int,
        lb: np.ndarray = -np.inf,
        ub: np.ndarray = np.inf,
    ) -> Tuple[cs.SX, cs.SX, cs.SX]:
        assert name not in self.vars, f"Variable {name} already exists."
        lb, ub = np.broadcast_to(lb, dims), np.broadcast_to(ub, dims)
        assert np.all(lb < ub), "Improper variable bounds."

        var = cs.SX.sym(name, *dims)
        self.vars[name] = var
        self.x = cs.vertcat(self.x, cs.vec(var))
        self.lbx = np.concatenate((self.lbx, cs.vec(lb).full().flatten()))
        self.ubx = np.concatenate((self.ubx, cs.vec(ub).full().flatten()))

        # create also the multiplier associated to the variable
        lam_lb = cs.SX.sym(f"lam_lb_{name}", *dims)
        self.lam_lbx = cs.vertcat(self.lam_lbx, cs.vec(lam_lb))
        lam_ub = cs.SX.sym(f"lam_ub_{name}", *dims)
        self.lam_ubx = cs.vertcat(self.lam_ubx, cs.vec(lam_ub))
        return var, lam_lb, lam_ub

    def add_con(
        self, name: str, expr1: cs.SX, op: str, expr2: cs.SX
    ) -> Tuple[cs.SX, cs.SX]:
        assert name not in self.cons, f"Constraint {name} already exists."
        expr = expr1 - expr2
        dims = expr.shape
        if op in {"=", "=="}:
            is_eq = True
            lb, ub = np.zeros(dims), np.zeros(dims)
        elif op in {"<", "<="}:
            is_eq = False
            lb, ub = np.full(dims, -np.inf), np.zeros(dims)
        elif op in {">", ">="}:
            is_eq = False
            expr = -expr
            lb, ub = np.full(dims, -np.inf), np.zeros(dims)
        else:
            raise ValueError(f"Unrecognized operator {op}.")
        expr = cs.simplify(expr)
        lb, ub = cs.vec(lb).full().flatten(), cs.vec(ub).full().flatten()
        self.cons[name] = expr
        group = "g" if is_eq else "h"
        setattr(self, group, cs.vertcat(getattr(self, group), cs.vec(expr)))
        setattr(self, f"lb{group}", np.concatenate((getattr(self, f"lb{group}"), lb)))
        setattr(self, f"ub{group}", np.concatenate((getattr(self, f"ub{group}"), ub)))
        lam = cs.SX.sym(f"lam_{group}_{name}", *dims)
        setattr(
            self, f"lam_{group}", cs.vertcat(getattr(self, f"lam_{group}"), cs.vec(lam))
        )
        return expr, lam

    def minimize(self, objective: cs.SX) -> None:
        self.f = objective

    def init_solver(self, opts: Dict) -> None:
        g = cs.vertcat(self.g, self.h)
        nlp = {"x": self.x, "p": self.p, "g": g, "f": self.f}
        self.solver = cs.nlpsol(f"nlpsol_{self.name}", "ipopt", nlp, opts)
        self.opts = opts

    def solve(
        self, pars: Dict[str, np.ndarray], vals0: Dict[str, np.ndarray] = None
    ) -> QuadRotorSolution:
        assert self.solver is not None, "Solver uninitialized."
        assert len(self.pars.keys() - pars.keys()) == 0, (
            "Trying to solve the MPC with unspecified parameters: "
            + ", ".join(self.pars.keys() - pars.keys())
            + "."
        )
        p = subsevalf(self.p, self.pars, pars)
        kwargs = {
            "p": p,
            "lbx": self.lbx,
            "ubx": self.ubx,
            "lbg": np.concatenate((self.lbg, self.lbh)),
            "ubg": np.concatenate((self.ubg, self.ubh)),
        }
        if vals0 is not None:
            kwargs["x0"] = np.clip(
                subsevalf(self.x, self.vars, vals0), self.lbx, self.ubx
            )
        sol: Dict[str, cs.DM] = self.solver(**kwargs)
        lam_lbx = -np.minimum(sol["lam_x"], 0)
        lam_ubx = np.maximum(sol["lam_x"], 0)
        lam_g = sol["lam_g"][: self.ng, :]
        lam_h = sol["lam_g"][self.ng :, :]
        S = cs.vertcat(
            self.p, self.x, self.lam_g, self.lam_h, self.lam_lbx, self.lam_ubx
        )
        D = cs.vertcat(p, sol["x"], lam_g, lam_h, lam_lbx, lam_ubx)
        get_value = partial(subsevalf, old=S, new=D)
        vals = {name: get_value(var) for name, var in self.vars.items()}
        return QuadRotorSolution(
            f=float(sol["f"]),
            vars=self.vars.copy(),
            vals=vals,
            get_value=get_value,
            stats=self.solver.stats().copy(),
        )

    def __str__(self) -> str:
        msg = "not initialized" if self.solver is None else "initialized"
        C = len(self.cons)
        return (
            f"{type(self).__name__} {{\n"
            f"  name: {self.name}\n"
            f"  #variables: {len(self.vars)} (nx={self.nx})\n"
            f"  #parameters: {len(self.pars)} (np={self.np})\n"
            f"  #constraints: {C} (ng={self.ng}, nh={self.nh})\n"
            f"  CasADi solver {msg}.\n}}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}: {self.name}"


def subsevalf(
    expr: cs.SX,
    old: Union[cs.SX, Dict[str, cs.SX], List[cs.SX], Tuple[cs.SX]],
    new: Union[cs.SX, Dict[str, cs.SX], List[cs.SX], Tuple[cs.SX]],
    eval: bool = True,
) -> Union[cs.SX, np.ndarray]:
    if isinstance(old, dict):
        for name, o in old.items():
            expr = cs.substitute(expr, o, new[name])
    elif isinstance(old, (tuple, list)):
        for o, n in zip(old, new):
            expr = cs.substitute(expr, o, n)
    else:
        expr = cs.substitute(expr, old, new)

    if eval:
        expr = cs.evalf(expr).full().squeeze()
    return expr


ConfigType = TypeVar("ConfigType")


def init_config(
    config: Optional[Union[ConfigType, Dict]], cls: Type[ConfigType]
) -> ConfigType:
    if config is None:
        return cls()
    if isinstance(config, cls):
        return config
    if isinstance(config, dict):
        if not hasattr(cls, "__dataclass_fields__"):
            raise ValueError("Configiration class must be a dataclass.")
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: config[k] for k in keys if k in config})
    raise ValueError(
        "Invalid configuration type; expected None, dict or "
        f"a dataclass, got {cls} instead."
    )


@dataclass
class QuadRotorMPCConfig:
    N: int = 10
    solver_opts: Dict = field(
        default_factory=lambda: {
            "expand": True,
            "print_time": False,
            "ipopt": {
                "max_iter": 500,
                "tol": 1e-6,
                "barrier_tol_factor": 1,
                "sb": "yes",
                # for debugging
                "print_level": 0,
                "print_user_options": "no",
                "print_options_documentation": "no",
            },
        }
    )


class QuadRotorMPC(GenericMPC):
    def __init__(
        self,
        env: QuadRotorEnv,
        config: Union[dict, QuadRotorMPCConfig] = None,
        mpctype: str = "V",
    ) -> None:
        assert mpctype in {
            "V",
            "Q",
        }, "MPC must be either V (state value func) or Q (action value func)"
        super().__init__(name=mpctype)
        self.config = init_config(config, QuadRotorMPCConfig)
        N = self.config.N

        # ======================= #
        # Variable and Parameters #
        # ======================= #
        lbx, ubx = env.config.x_bounds[:, 0], env.config.x_bounds[:, 1]
        not_red = ~(np.isneginf(lbx) & np.isposinf(ubx))
        not_red_idx = np.where(not_red)[0]
        lbx, ubx = lbx[not_red].reshape(-1, 1), ubx[not_red].reshape(-1, 1)
        nx, nu = env.nx, env.nu
        x, _, _ = self.add_var("x", nx, N)
        u, _, _ = self.add_var("u", nu, N)
        ns = not_red_idx.size + nu
        s, _, _ = self.add_var("slack", ns * N - not_red_idx.size, 1, lb=0)
        sx: cs.SX = s[: not_red_idx.size * (N - 1)].reshape((-1, N - 1))
        su: cs.SX = s[-nu * N :].reshape((-1, N))

        # 2) create model parameters
        for name in (
            "g",
            "thrust_coeff",
            "pitch_d",
            "pitch_dd",
            "pitch_gain",
            "roll_d",
            "roll_dd",
            "roll_gain",
        ):
            self.add_par(name, 1, 1)

        # =========== #
        # Constraints #
        # =========== #

        # 1) constraint on initial conditions
        x0 = self.add_par("x0", env.nx, 1)
        x_ = cs.horzcat(x0, x)

        # 2) constraints on dynamics
        A, B, e = env.get_dynamics(
            g=self.pars["g"],
            thrust_coeff=self.pars["thrust_coeff"],
            pitch_d=self.pars["pitch_d"],
            pitch_dd=self.pars["pitch_dd"],
            pitch_gain=self.pars["pitch_gain"],
            roll_d=self.pars["roll_d"],
            roll_dd=self.pars["roll_dd"],
            roll_gain=self.pars["roll_gain"],
        )
        self.add_con("dyn", x_[:, 1:], "==", A @ x_[:, :-1] + B @ u + e)

        # 3) constraint on state (soft, backed off, without infinity in g, and
        # removing redundant entries, no constraint on first state)
        # constraint backoff parameter and bounds
        bo = self.add_par("backoff", 1, 1)

        # set the state constraints as
        #  - soft-backedoff minimum constraint: (1+back)*lb - slack <= x
        #  - soft-backedoff maximum constraint: x <= (1-back)*ub + slack
        # NOTE: there is a mistake here in the old code, since we are excluding the
        # first state from constraints which is actually the second.
        self.add_con("x_min", (1 + bo) * lbx - sx, "<=", x[not_red_idx, 1:])
        self.add_con("x_max", x[not_red_idx, 1:], "<=", (1 - bo) * ubx + sx)
        self.add_con("u_min", env.config.u_bounds[:, 0] - su, "<=", u)
        self.add_con("u_max", u, "<=", env.config.u_bounds[:, 1] + su)

        # ========= #
        # Objective #
        # ========= #
        J = 0  # (no initial state cost not required since it is not economic)
        s = cs.blockcat([[cs.SX.zeros(sx.size1(), 1), sx], [su]])
        xf = self.add_par("xf", nx, 1)
        uf = cs.vertcat(0, 0, self.pars["g"])
        w_x = self.add_par("w_x", nx, 1)  # weights for stage/final state
        w_u = self.add_par("w_u", nu, 1)  # weights for stage/final control
        w_s = self.add_par("w_s", ns, 1)  # weights for stage/final slack
        J += sum(
            (
                quad_form(w_x, x[:, k] - xf)
                + quad_form(w_u, u[:, k] - uf)
                + cs.dot(w_s, s[:, k])
            )
            for k in range(N - 1)
        )
        J += (
            quad_form(w_x, x[:, -1] - xf)
            + quad_form(w_u, u[:, -1] - uf)
            + cs.dot(w_s, s[:, -1])
        )
        self.minimize(J)

        # ====== #
        # Others #
        # ====== #
        if mpctype == "Q":
            u0 = self.add_par("u0", nu, 1)
            self.add_con("init_action", u[:, 0], "==", u0)
        else:
            perturbation = self.add_par("perturbation", nu, 1)
            self.f += cs.dot(perturbation, u[:, 0])
        self.init_solver(self.config.solver_opts)


MPCType = TypeVar("MPCType", bound=GenericMPC)


class DifferentiableMPC(Generic[MPCType]):
    def __init__(self, mpc: MPCType) -> None:
        self._mpc = mpc

    @property
    def mpc(self) -> MPCType:
        return self._mpc

    @property
    def _non_redundant_x_bound_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.where(self._mpc.lbx != -np.inf)[0],
            np.where(self._mpc.ubx != np.inf)[0],
        )

    @property
    def lagrangian(self) -> cs.SX:
        idx_lbx, idx_ubx = self._non_redundant_x_bound_indices
        h_lbx = self._mpc.lbx[idx_lbx, None] - self._mpc.x[idx_lbx]
        h_ubx = self._mpc.x[idx_ubx] - self._mpc.ubx[idx_ubx, None]
        return (
            self._mpc.f
            + cs.dot(self._mpc.lam_g, self._mpc.g)
            + cs.dot(self._mpc.lam_h, self._mpc.h)
            + cs.dot(self._mpc.lam_lbx[idx_lbx], h_lbx)
            + cs.dot(self._mpc.lam_ubx[idx_ubx], h_ubx)
        )

    def __getattr__(self, name) -> Any:
        return getattr(self._mpc, name)


T = TypeVar("T")


class ReplayMemory(Deque[T]):
    def __init__(
        self, iterable: Iterable[T] = (), maxlen: int = None, seed: int = None
    ) -> None:
        super().__init__(iterable, maxlen=maxlen)
        seed_seq = np.random.SeedSequence(seed)
        self.np_random = np.random.Generator(np.random.PCG64(seed_seq))

    def sample(
        self, n: Union[int, float], include_last_n: Union[int, float]
    ) -> Iterable[T]:
        length = len(self)
        if isinstance(n, float):
            n = int(self.maxlen * n)
        n = np.clip(n, min(1, length), length)
        if isinstance(include_last_n, float):
            include_last_n = int(n * include_last_n)
        include_last_n = np.clip(include_last_n, 0, n)
        last_n = range(length - include_last_n, length)
        sampled = self.np_random.choice(
            range(length - include_last_n), n - include_last_n, replace=False
        )
        yield from (self[i] for i in chain(last_n, sampled))


@dataclass
class RLParameter:
    name: str
    value: np.ndarray
    bounds: np.ndarray
    symV: cs.SX
    symQ: cs.SX

    @property
    def size(self) -> int:
        return self.symV.shape[0]  # since rl pars are all column vectors

    def __post_init__(self) -> None:
        shape = self.symV.shape
        assert shape == self.symQ.shape, (
            f"Parameter {self.name} has different shapes in "
            f"Q ({self.symQ.shape}) and V ({self.symV.shape})."
        )
        assert self.symV.is_column(), f"Parameter {self.name} must be a column vector."
        self.bounds = np.broadcast_to(self.bounds, (shape[0], 2))
        self.update_value(self.value)

    def update_value(self, new_val: np.ndarray) -> None:
        """Updates the parameter's current value to the new one."""
        new_val = np.broadcast_to(new_val, self.bounds.shape[0])
        assert (
            (self.bounds[:, 0] <= new_val) | np.isclose(new_val, self.bounds[:, 0])
        ).all() and (
            (new_val <= self.bounds[:, 1]) | np.isclose(new_val, self.bounds[:, 1])
        ).all(), "Parameter value outside bounds."
        self.value = np.clip(new_val, self.bounds[:, 0], self.bounds[:, 1])


class RLParameterCollection(Sequence[RLParameter]):
    """Collection of learnable RL parameters, which can be accessed by string as a
    dictionary or by index as a list."""

    def __init__(self, *parameters: RLParameter) -> None:
        """Instantiate the collection from another iterable, if provided."""
        self._list: List[RLParameter] = []
        self._dict: Dict[str, RLParameter] = {}
        for parameter in parameters:
            self._list.append(parameter)
            self._dict[parameter.name] = parameter

    @property
    def n_theta(self) -> int:
        return sum(self.sizes())

    @property
    def as_dict(self) -> Dict[str, RLParameter]:
        return self._dict

    def values(self, as_dict: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if as_dict:
            return {name: p.value for name, p in self.items()}
        return np.concatenate([p.value for p in self._list])

    def bounds(self, as_dict: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if as_dict:
            return {name: p.bounds for name, p in self.items()}
        return np.row_stack([p.bounds for p in self._list])

    def symQ(self, as_dict: bool = False) -> Union[cs.SX, Dict[str, cs.SX]]:
        if as_dict:
            return {name: p.symQ for name, p in self.items()}
        return cs.vertcat(*(p.symQ for p in self._list))

    def sizes(self, as_dict: bool = False) -> Union[List[int], Dict[str, int]]:
        if as_dict:
            return {p.name: p.size for p in self._list}
        return [p.size for p in self._list]

    def update_values(
        self, new_vals: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]
    ) -> None:
        if isinstance(new_vals, np.ndarray):
            new_vals = np.split(new_vals, np.cumsum(self.sizes())[:-1])
            for p, val in zip(self._list, new_vals):
                p.update_value(val)
        elif isinstance(new_vals, list):
            for p, val in zip(self._list, new_vals):
                p.update_value(val)
        elif isinstance(new_vals, dict):
            for n in self._dict.keys():
                self._dict[n].update_value(new_vals[n])

    def items(self) -> Iterable[Tuple[str, RLParameter]]:
        return self._dict.items()

    def __getitem__(
        self, index: Union[str, Iterable[str], int, slice, Iterable[int]]
    ) -> Union[RLParameter, List[RLParameter]]:
        if isinstance(index, str):
            return self._dict[index]
        if isinstance(index, (int, slice)):
            return self._list[index]
        if isinstance(index, Iterable):
            return [self._list[i] for i in index]

    def __iter__(self) -> Iterator[RLParameter]:
        return iter(self._list)

    def __next__(self) -> RLParameter:
        return next(self._list)

    def __len__(self) -> int:
        return len(self._list)


class QuadRotorBaseAgent(ABC):
    def __init__(
        self,
        env: QuadRotorEnv,
        agentname: str = None,
        agent_config: Union[Dict[str, Any], Any] = None,
        fixed_pars: Dict[str, np.ndarray] = None,
        mpc_config: Union[Dict, QuadRotorMPCConfig] = None,
        seed: int = None,
    ) -> None:
        super().__init__()
        self.name = "Agent" if agentname is None else agentname
        self.env = env
        self.config = (
            init_config(agent_config, self.config_cls)
            if hasattr(self, "config_cls")
            else None
        )
        self.fixed_pars = {} if fixed_pars is None else fixed_pars
        self.seed = seed
        seed_seq = np.random.SeedSequence(seed)
        self.np_random = np.random.Generator(np.random.PCG64(seed_seq))
        self.perturbation_chance = 0.0
        self.perturbation_strength = 0.0
        self.last_solution: Solution = None
        self.Q = QuadRotorMPC(env, config=mpc_config, mpctype="Q")
        self.V = QuadRotorMPC(env, config=mpc_config, mpctype="V")

    @property
    def unwrapped(self) -> "QuadRotorBaseAgent":
        return self

    def reset(self) -> None:
        self.last_solution = None
        self.Q.failures = 0
        self.V.failures = 0

    def solve_mpc(
        self,
        type: str,
        state: np.ndarray = None,
        sol0: Dict[str, np.ndarray] = None,
    ) -> Solution:
        mpc: QuadRotorMPC = getattr(self, type)
        if state is None:
            state = self.env.x
        pars = self.fixed_pars.copy()
        pars["x0"] = state
        pars.update(self._merge_mpc_pars_callback())
        if sol0 is None:
            if self.last_solution is None:
                g = float(pars.get("g", 0))
                sol0 = {
                    "x": np.tile(state, (mpc.vars["x"].shape[1], 1)).T,
                    "u": np.tile([0, 0, g], (mpc.vars["u"].shape[1], 1)).T,
                    "slack": 0,
                }
            else:
                sol0 = self.last_solution.vals
        self.last_solution = mpc.solve(pars, sol0)
        return self.last_solution

    def predict(
        self,
        state: np.ndarray = None,
        deterministic: bool = False,
        perturb_gradient: bool = True,
        **solve_mpc_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Solution]:
        perturbation_in_dict = "perturbation" in self.fixed_pars
        if perturbation_in_dict:
            self.fixed_pars["perturbation"] = 0
        if deterministic or self.np_random.random() > self.perturbation_chance:
            sol = self.solve_mpc(type="V", state=state, **solve_mpc_kwargs)
            u_opt = sol.vals["u"][:, 0]
        else:
            u_bnd = self.env.config.u_bounds
            rng = self.np_random.normal(
                scale=self.perturbation_strength * np.diff(u_bnd).flatten(),
                size=self.V.vars["u"].shape[0],
            )
            if perturb_gradient:
                assert (
                    perturbation_in_dict
                ), "No parameter 'perturbation' found to perturb gradient."
                self.fixed_pars["perturbation"] = rng
            sol = self.solve_mpc(type="V", state=state, **solve_mpc_kwargs)
            u_opt = sol.vals["u"][:, 0]
            if not perturb_gradient:
                u_opt = np.clip(u_opt + rng, u_bnd[:, 0], u_bnd[:, 1])
        x_next = sol.vals["x"][:, 0]
        return u_opt, x_next, sol

    def _merge_mpc_pars_callback(self) -> Dict[str, np.ndarray]:
        return {}

    @staticmethod
    def _make_seed_list(seed: Optional[Union[int, List[int]]], n: int) -> List[int]:
        if seed is None:
            return [None] * n
        if isinstance(seed, int):
            return [seed + i for i in range(n)]
        assert len(seed) == n, "Seed sequence with invalid length."
        return seed


class QuadRotorBaseLearningAgent(QuadRotorBaseAgent, ABC):
    def __init__(
        self,
        *args,
        init_learnable_pars: Dict[str, Tuple[np.ndarray, np.ndarray]],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.V = DifferentiableMPC[QuadRotorMPC](self.V)
        self.Q = DifferentiableMPC[QuadRotorMPC](self.Q)
        self._init_learnable_pars(init_learnable_pars)
        self._init_learning_rate()
        self._epoch_n = None  # keeps track of epoch number just for logging

    @abstractmethod
    def update(self) -> np.ndarray:
        pass

    @abstractmethod
    def learn_one_epoch(
        self,
        n_episodes: int,
        perturbation_decay: float = 0.75,
        seed: Union[int, List[int]] = None,
        return_info: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]:
        pass

    def learn(
        self,
        n_epochs: int,
        n_episodes: int,
        perturbation_decay: float = 0.75,
        seed: Union[int, List[int]] = None,
        throw_on_exception: bool = False,
        return_info: bool = True,
    ) -> Union[
        Tuple[bool, np.ndarray],
        Tuple[bool, np.ndarray, List[np.ndarray], List[Dict[str, np.ndarray]]],
    ]:
        ok = True
        results = []
        for e in range(n_epochs):
            self._epoch_n = e  # just for logging
            try:
                results.append(
                    self.learn_one_epoch(
                        n_episodes=n_episodes,
                        perturbation_decay=perturbation_decay,
                        seed=None if seed is None else seed + n_episodes * e,
                        return_info=return_info,
                    )
                )
            except (MpcSolverError, UpdateError) as ex:
                if throw_on_exception:
                    raise ex
                ok = False
                break
        if not results:
            return (ok, np.nan, [], []) if return_info else (ok, np.nan)
        if not return_info:
            return ok, np.stack(results, axis=0)
        returns, grads, weightss = list(zip(*results))
        return ok, np.stack(returns, axis=0), grads, weightss

    def _init_learnable_pars(
        self, init_pars: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Initializes the learnable parameters of the MPC."""
        required_pars = sorted(
            set(self.Q.pars)
            .intersection(self.V.pars)
            .difference({"x0", "xf"})
            .difference(self.fixed_pars)
        )
        self.weights = RLParameterCollection(
            *(
                RLParameter(
                    name, *init_pars[name], self.V.pars[name], self.Q.pars[name]
                )
                for name in required_pars
            )
        )

    def _init_learning_rate(self) -> None:
        cfg = self.config
        if cfg is None or not hasattr(cfg, "lr"):
            return
        n_pars, n_theta = len(self.weights), self.weights.n_theta
        lr = np.asarray(cfg.lr).squeeze()
        if lr.ndim == 0:
            lr = np.full((n_theta,), lr)
        elif lr.size == n_pars and lr.size != n_theta:
            lr = np.concatenate([np.full(p.size, r) for p, r in zip(self.weights, lr)])
        assert lr.shape == (
            n_theta,
        ), "Learning rate must have the same size as the learnable parameter vector."
        cfg.lr = lr

    def _merge_mpc_pars_callback(self) -> Dict[str, np.ndarray]:
        return self.weights.values(as_dict=True)

    @staticmethod
    def _get_percentage_bounds(
        theta: np.ndarray,
        theta_bounds: np.ndarray,
        max_perc_update: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        max_delta = np.maximum(np.abs(max_perc_update * theta), 0.1)
        lb = np.maximum(theta_bounds[:, 0], theta - max_delta)
        ub = np.minimum(theta_bounds[:, 1], theta + max_delta)
        return lb, ub


@dataclass
class QuadRotorLSTDQAgentConfig:
    init_pars: Dict[str, Tuple[float, Tuple[float, float]]] = field(
        default_factory=lambda: {
            "g": (9.81, (1, 40)),
            "thrust_coeff": (0.3, (0.1, 4)),
            "backoff": (0.1, (1e-3, 0.5)),
        }
    )
    fixed_pars: Dict[str, float] = field(
        default_factory=lambda: {
            "pitch_d": 12,
            "pitch_dd": 5,
            "pitch_gain": 12,
            "roll_d": 13,
            "roll_dd": 6,
            "roll_gain": 8,
            "w_x": 1e1,
            "w_u": 1e0,
            "w_s": 1e2,
        }
    )
    replay_maxlen: float = 20
    replay_sample_size: float = 10
    replay_include_last: float = 5
    gamma: float = 1.0
    lr: float = 1e-1
    max_perc_update: float = np.inf


class QuadRotorLSTDQAgent(QuadRotorBaseLearningAgent):
    config_cls: type = QuadRotorLSTDQAgentConfig

    def __init__(
        self,
        env: QuadRotorEnv,
        agentname: str = None,
        agent_config: Union[dict, QuadRotorLSTDQAgentConfig] = None,
        mpc_config: Union[dict, QuadRotorMPCConfig] = None,
        seed: int = None,
    ) -> None:
        # create base agent
        agent_config = init_config(agent_config, self.config_cls)
        fixed_pars, init_pars = agent_config.fixed_pars, agent_config.init_pars
        fixed_pars.update({"xf": env.config.xf, "perturbation": np.nan})
        super().__init__(
            env,
            agentname=agentname,
            agent_config=agent_config,
            fixed_pars=fixed_pars,
            init_learnable_pars=init_pars,
            mpc_config=mpc_config,
            seed=seed,
        )
        self.perturbation_chance = 0.0
        self.perturbation_strength = 0.0
        self.replay_memory = ReplayMemory[List[Tuple[np.ndarray, ...]]](
            maxlen=self.config.replay_maxlen, seed=seed
        )
        self._episode_buffer: List[Tuple[np.ndarray, ...]] = []
        self._init_derivative_symbols()
        self._init_qp_solver()

    def save_transition(self, cost: float, solQ: Solution, solV: Solution) -> None:
        target = cost + self.config.gamma * solV.f
        td_err = target - solQ.f
        dQ = solQ.value(self.dQdtheta).reshape(-1, 1)
        d2Q = solQ.value(self.d2Qdtheta)
        g = -td_err * dQ
        H = dQ @ dQ.T - td_err * d2Q
        self._episode_buffer.append((g, H))

    def consolidate_episode_experience(self) -> None:
        if len(self._episode_buffer) == 0:
            return
        self.replay_memory.append(self._episode_buffer.copy())
        self._episode_buffer.clear()

    def update(self) -> np.ndarray:
        # sample the memory
        cfg: QuadRotorLSTDQAgentConfig = self.config
        sample = self.replay_memory.sample(
            cfg.replay_sample_size, cfg.replay_include_last
        )
        g, H = (np.mean(o, axis=0) for o in zip(*chain.from_iterable(sample)))
        R = cholesky_added_multiple_identities(H)
        p = cho_solve((R, True), g).flatten()
        theta = self.weights.values()
        pars = np.block([theta, p, cfg.lr])
        lb, ub = self._get_percentage_bounds(
            theta, self.weights.bounds(), cfg.max_perc_update
        )
        sol = self._solver(p=pars, lbx=lb, ubx=ub, x0=theta - cfg.lr * p)
        if not self._solver.stats()["success"]:
            raise UpdateError(f"RL update failed in epoch {self._epoch_n}.")
        self.weights.update_values(sol["x"].full().flatten())
        return p

    def learn_one_epoch(
        self,
        n_episodes: int,
        perturbation_decay: float = 0.75,
        seed: Union[int, List[int]] = None,
        return_info: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]:
        env, name, epoch_n = self.env, self.name, self._epoch_n
        returns = np.zeros(n_episodes)
        seeds = self._make_seed_list(seed, n_episodes)

        for e in range(n_episodes):
            state, _ = env.reset(seed=seeds[e])
            self.reset()
            truncated, terminated, t = False, False, 0
            action = self.predict(state, deterministic=False)[0]
            while not (truncated or terminated):
                # compute Q(s, a)
                self.fixed_pars.update({"u0": action})
                solQ = self.solve_mpc("Q", state)
                # step the system
                state, r, truncated, terminated, _ = env.step(action)
                returns[e] += r
                # compute V(s+)
                action, _, solV = self.predict(state, deterministic=False)
                if solQ.success and solV.success:
                    self.save_transition(r, solQ, solV)
                else:
                    raise MpcSolverError(f"{name}|{epoch_n}|{e}|{t}: mpc failed.")
                t += 1
            self.consolidate_episode_experience()

        update_grad = self.update()
        self.perturbation_strength *= perturbation_decay
        self.perturbation_chance *= perturbation_decay
        return (
            (returns, update_grad, self.weights.values(as_dict=True))
            if return_info
            else returns
        )

    def _init_derivative_symbols(self) -> None:
        theta = self.weights.symQ()
        lagr = self.Q.lagrangian
        d2Qdtheta, dQdtheta = cs.hessian(lagr, theta)
        self.dQdtheta = cs.simplify(dQdtheta)
        self.d2Qdtheta = cs.simplify(d2Qdtheta)

    def _init_qp_solver(self) -> None:
        n_theta = self.weights.n_theta
        theta: cs.SX = cs.SX.sym("theta", n_theta, 1)
        theta_new: cs.SX = cs.SX.sym("theta+", n_theta, 1)
        dtheta = theta_new - theta
        p: cs.SX = cs.SX.sym("p", n_theta, 1)
        lr: cs.SX = cs.SX.sym("lr", n_theta, 1)
        qp = {
            "x": theta_new,
            "f": 0.5 * dtheta.T @ dtheta + (lr * p).T @ dtheta,
            "p": cs.vertcat(theta, p, lr),
        }
        opts = {"print_iter": False, "print_header": False}
        self._solver = cs.qpsol(f"qpsol_{self.name}", "qrqp", qp, opts)


AgentType = TypeVar("AgentType", bound=QuadRotorBaseLearningAgent)


class RecordLearningData(Generic[AgentType]):
    def __init__(self, agent: AgentType) -> None:
        self.agent = agent

        # initialize storages
        self.weights_history: Dict[str, List[np.ndarray]] = {
            n: [p.value] for n, p in agent.weights.as_dict.items()
        }
        self.update_gradient: List[np.ndarray] = []

    @property
    def unwrapped(self) -> AgentType:
        return self.agent

    def learn_one_epoch(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        returns, grad, weights = self.agent.learn_one_epoch(*args, **kwargs)
        self._save(grad, weights)
        return returns, grad

    def learn(
        self, *args, **kwargs
    ) -> Tuple[bool, np.ndarray, List[np.ndarray], List[Dict[str, np.ndarray]]]:
        ok, returns, grads, weightss = self.agent.learn(*args, **kwargs)
        for grad, weights in zip(grads, weightss):
            self._save(grad, weights)
        return ok, returns, grads, weightss

    def _save(self, grad: np.ndarray, weights: Dict[str, np.ndarray]) -> None:
        self.update_gradient.append(grad)
        for n, w in self.weights_history.items():
            w.append(weights[n])

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited.")
        return getattr(self.agent, name)


# ==================================================================================== #
# ----------------------------------- END OLD CODE ----------------------------------- #
# ==================================================================================== #


class QuadRotorMpcActual(Mpc):
    def __init__(self, env: QuadRotorEnv) -> None:
        N = QuadRotorMPCConfig.N
        super().__init__(Nlp(sym_type="SX"), prediction_horizon=N, shooting="multi")

        # ======================= #
        # Variable and Parameters #
        # ======================= #
        lbx, ubx = env.config.x_bounds[:, 0], env.config.x_bounds[:, 1]
        not_red = ~(np.isneginf(lbx) & np.isposinf(ubx))
        not_red_idx = np.where(not_red)[0]
        lbx, ubx = lbx[not_red].reshape(-1, 1), ubx[not_red].reshape(-1, 1)
        nx, nu = env.nx, env.nu
        x, _ = self.state("x", nx)
        u, _ = self.action("u", nu)
        ns = not_red_idx.size + nu
        s, _, _ = self.variable("slack", (ns * N - not_red_idx.size, 1), lb=0)
        sx: cs.SX = s[: not_red_idx.size * (N - 1)].reshape((-1, N - 1))
        su: cs.SX = s[-nu * N :].reshape((-1, N))

        # 2) create model parameters
        for name in (
            "g",
            "thrust_coeff",
            "pitch_d",
            "pitch_dd",
            "pitch_gain",
            "roll_d",
            "roll_dd",
            "roll_gain",
        ):
            self.parameter(name, (1, 1))

        # =========== #
        # Constraints #
        # =========== #
        A, B, e = env.get_dynamics(
            g=self.parameters["g"],
            thrust_coeff=self.parameters["thrust_coeff"],
            pitch_d=self.parameters["pitch_d"],
            pitch_dd=self.parameters["pitch_dd"],
            pitch_gain=self.parameters["pitch_gain"],
            roll_d=self.parameters["roll_d"],
            roll_dd=self.parameters["roll_dd"],
            roll_gain=self.parameters["roll_gain"],
        )
        self.set_dynamics(lambda x, u: A @ x + B @ u + e, n_in=2, n_out=1)

        # 3) constraint on state
        bo = self.parameter("backoff", (1, 1))
        self.constraint("x_min", (1 + bo) * lbx - sx, "<=", x[not_red_idx, 2:])
        self.constraint("x_max", x[not_red_idx, 2:], "<=", (1 - bo) * ubx + sx)
        self.constraint("u_min", env.config.u_bounds[:, 0] - su, "<=", u)
        self.constraint("u_max", u, "<=", env.config.u_bounds[:, 1] + su)

        # ========= #
        # Objective #
        # ========= #
        J = 0  # (no initial state cost not required since it is not economic)
        s = cs.blockcat([[cs.SX.zeros(sx.size1(), 1), sx], [su]])
        xf = self.parameter("xf", (nx, 1))
        uf = cs.vertcat(0, 0, self.parameters["g"])
        w_x = self.parameter("w_x", (nx, 1))  # weights for stage/final state
        w_u = self.parameter("w_u", (nu, 1))  # weights for stage/final control
        w_s = self.parameter("w_s", (ns, 1))  # weights for stage/final slack
        J += sum(
            (
                quad_form(w_x, x[:, k + 1] - xf)
                + quad_form(w_u, u[:, k] - uf)
                + cs.dot(w_s, s[:, k])
            )
            for k in range(N - 1)
        )
        J += (
            quad_form(w_x, x[:, -1] - xf)
            + quad_form(w_u, u[:, -1] - uf)
            + cs.dot(w_s, s[:, -1])
        )
        self.minimize(J)
        self.init_solver(
            QuadRotorMPCConfig.__dataclass_fields__["solver_opts"].default_factory()
        )


class TestQuadRotorQlearning(unittest.TestCase):
    def test(self):
        # for comparison
        # - replay maxlen must be 1, i.e., use only the latest episode for updates
        # - no exploration since np_randoms are placed differently
        seed = 42
        Tlimit = 20
        env = TimeLimit(QuadRotorEnv(), Tlimit)
        agent_config = {
            "gamma": 0.9792,
            "lr": [0.498],
            "max_perc_update": np.inf,
            "replay_maxlen": 1,
            "replay_sample_size": 1.0,
            "replay_include_last": 1,
            "perturbation_decay": 0.885,
        }
        agent_expected = RecordLearningData(
            QuadRotorLSTDQAgent(
                env=env, agentname="LSTDQ_0", agent_config=agent_config, seed=seed
            )
        )
        results_expected = agent_expected.learn(
            n_epochs=2,
            n_episodes=1,
            perturbation_decay=agent_config["perturbation_decay"],
            seed=seed + 1,
            throw_on_exception=True,
        )
        self.assertTrue(results_expected[0])

        mpc = QuadRotorMpcActual(env)
        fp_field = QuadRotorLSTDQAgentConfig.__dataclass_fields__["fixed_pars"]
        fixed_pars = fp_field.default_factory()
        fixed_pars["xf"] = env.config.xf
        lp_field = QuadRotorLSTDQAgentConfig.__dataclass_fields__["init_pars"]
        learnable_pars = LearnableParametersDict[cs.SX](
            (
                LearnableParameter(
                    name=name,
                    size=1,
                    value=init,
                    lb=lb,
                    ub=ub,
                    sym=cs.vec(mpc.parameters[name]),
                )
                for name, (init, (lb, ub)) in lp_field.default_factory().items()
            )
        )
        agent_actual = RecordUpdates(
            LstdQLearningAgent(
                mpc=mpc,
                discount_factor=agent_config["gamma"],
                learning_rate=agent_config["lr"][0],
                learnable_parameters=learnable_pars,
                fixed_parameters=fixed_pars,
                exploration=E.EpsilonGreedyExploration(
                    S.ExponentialScheduler(0.0, agent_config["perturbation_decay"]),
                    S.ExponentialScheduler(0.0, agent_config["perturbation_decay"]),
                    seed=seed,
                ),
                experience=ExperienceReplay(maxlen=Tlimit, sample_size=1.0),
                update_strategy=Tlimit,
            )
        )
        results_actual = LstdQLearningAgent.train(
            agent_actual,
            env=env,
            episodes=2,
            seed=seed + 1,
        )

        np.testing.assert_allclose(results_actual, results_expected[1].flatten())
        for n, weights in agent_actual.updates_history.items():
            np.testing.assert_allclose(weights, agent_expected.weights_history[n])


if __name__ == "__main__":
    unittest.main()
