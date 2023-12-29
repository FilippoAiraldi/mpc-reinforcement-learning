from typing import Generic, Literal, Optional, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from ..core.schedulers import NoScheduling, Scheduler
from .base_optimizer import BaseOptimizer, SymType

LrType = TypeVar("LrType", npt.NDArray[np.floating], float)


class GradientBasedOptimizer(BaseOptimizer[SymType], Generic[SymType, LrType]):
    """Base class for first- and second-order gradient-based optimization algorithms."""

    _order: Literal[1, 2]
    """Order of the optimizer: 1 for first-order, 2 for second-order."""

    _hessian_sparsity: Literal["dense", "diag"]
    """Sparsity of the hessian. It can be overridden by each subclass, in case a
    particular structure is known, e.g., diagonal."""

    def __init__(
        self,
        learning_rate: Union[LrType, Scheduler[LrType]],
        hook: Literal["on_update", "on_episode_end", "on_timestep_end"] = "on_update",
        max_percentage_update: float = float("+inf"),
    ) -> None:
        """Instantiates the optimizer.

        Parameters
        ----------
        learning_rate : float/array, scheduler
            The learning rate of the optimizer. A float/array can be passed in case the
            learning rate must stay constant; otherwise, a scheduler can be passed which
            will be stepped `on_update` by default (see `hook` argument).
        hook : {'on_update', 'on_episode_end', 'on_timestep_end'}, optional
            Specifies when to step the optimizer's learning rate's scheduler to decay
            its value (see `step` method also). This allows to vary the rate over the
            learning iterations. The options are:
             - `on_update` steps the learning rate after each agent's update
             - `on_episode_end` steps the learning rate after each episode's end
             - `on_timestep_end` steps the learning rate after each env's timestep.

            By default, 'on_update' is selected.
        max_percentage_update : float, optional
            A positive float that specifies the maximum percentage change the learnable
            parameters can experience in each update. For example,
            `max_percentage_update=0.5` means that the parameters can be updated by up
            to 50% of their current value. By default, it is set to `+inf`.
        """
        super().__init__(max_percentage_update)
        if not isinstance(learning_rate, Scheduler):
            learning_rate = NoScheduling[LrType](learning_rate)
        self.lr_scheduler: Scheduler[LrType] = learning_rate
        self._hook = hook
        self._update_solver: cs.Function

    @property
    def hook(self) -> Optional[str]:
        """Specifies to which callback to hook, i.e., when to step the learning rate's
        scheduler to decay its value (see `step` method also). Can be `None` in case no
        hook is needed."""
        # return hook only if the learning rate scheduler requires to be stepped
        return None if isinstance(self.lr_scheduler, NoScheduling) else self._hook

    def step(self, *_, **__) -> None:
        """Steps/decays the learning rate according to its scheduler."""
        self.lr_scheduler.step()

    def _init_update_solver(self) -> Optional[cs.Function]:
        """Internal utility to initialize, if the learnable parameters are constrained,
        a constrained update solver which, by default, is a QP. Otherwise, no solver is
        required to perform the update.
        """
        if (
            self.max_percentage_update == float("+inf")
            and np.isneginf(self.learnable_parameters.lb).all()
            and np.isposinf(self.learnable_parameters.ub).all()
        ):
            return None
        n_params = self.learnable_parameters.size
        qp = {"h": getattr(cs.Sparsity, self._hessian_sparsity)(n_params, n_params)}
        opts = {
            "error_on_fail": False,
            "osqp": {
                "verbose": False,
                "polish": True,
                "scaling": 20,
                "eps_abs": 1e-9,
                "eps_rel": 1e-9,
                "eps_prim_inf": 1e-10,
                "eps_dual_inf": 1e-10,
                "max_iter": 6000,
            },
        }
        return cs.conic(f"qpsol_{id(self)}", "osqp", qp, opts)

    def update(
        self,
        gradient: npt.NDArray[np.floating],
        hessian: Optional[npt.NDArray[np.floating]] = None,
    ) -> Optional[str]:
        """Computes the gradient update of the learnable parameters dictated by the
        current RL algorithm.

        Parameters
        ----------
        gradient : 1D array
            The gradient of the learnable parameters.
        hessian : None, or 2D array
            The hessian of the learnable parameters. When the optimizer is firt-order,
            it is expected to be `None` (as it is unused). When the optimizer is
            second-order, it is expected to be a 2D array.

        Returns
        -------
        status : str, optional
            An optional string containing the status of the update, e.g., the status of
            the QP solver, if used.
        """
        if self._order == 1:
            theta_new, status = self._first_order_update(gradient)
        else:
            theta_new, status = self._second_order_update(gradient, hessian)
        # theta_new = np.clip(
        #     theta_new, self.learnable_parameters.lb, self.learnable_parameters.ub
        # )
        self.learnable_parameters.update_values(theta_new)
        return status

    def _first_order_update(
        self, gradient: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], Optional[str]]:
        """Internally runs a first order update."""
        raise NotImplementedError(
            f"`{self.__class__.__name__}` optimizer does not implement "
            "`_first_order_update`"
        )

    def _second_order_update(
        self, gradient: npt.NDArray[np.floating], hessian: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], Optional[str]]:
        """Internally runs a second order update."""
        raise NotImplementedError(
            f"`{self.__class__.__name__}` optimizer does not implement "
            "`_second_order_update`"
        )

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        hookstr = "None" if self.hook is None else f"'{self.hook}'"
        mp = self.max_percentage_update
        return f"{cn}(lr={self.lr_scheduler},hook={hookstr},max%={mp})"
