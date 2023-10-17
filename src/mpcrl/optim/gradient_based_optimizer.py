from typing import Optional, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from mpcrl.core.learning_rate import LearningRate, LrType
from mpcrl.core.parameters import LearnableParametersDict, SymType
from mpcrl.core.schedulers import Scheduler


class GradientBasedOptimizer:
    """Base class for first- and second-order gradient-based optimization algorithms."""

    _hessian_sparsity = "dense"
    """This is the default sparsity of the hessian, which is dense. It can be overridden
    by each subclass, in case a particular structure is known, e.g., diagonal."""

    def __init__(
        self,
        learning_rate: Union[LrType, Scheduler[LrType], LearningRate[LrType]],
        max_percentage_update: float = float("+inf"),
    ) -> None:
        """Instantiates the optimizer.

        Parameters
        ----------
        learning_rate : float/array, scheduler or LearningRate
            The learning rate of the optimizer. A float/array can be passed in case the
            learning rate must stay constant; otherwise, a scheduler can be passed which
            will be stepped `on_update` by default. Otherwise, a `LearningRate` object
            can be passed, allowing to specify both the scheduling and stepping
            strategies of this fundamental hyper-parameter.
        max_percentage_update : float, optional
            A positive float that specifies the maximum percentage change the learnable
            parameters can experience in each update. For example,
            `max_percentage_update=0.5` means that the parameters can be updated by up
            to 50% of their current value. By default, it is set to `+inf`.
        """
        if not isinstance(learning_rate, LearningRate):
            learning_rate = LearningRate(learning_rate, "on_update")
        self.learning_rate: LearningRate[LrType] = learning_rate
        self.max_percentage_update = max_percentage_update
        self.learnable_parameters: LearnableParametersDict[SymType] = None
        self._update_solver: cs.Function = None

    def set_learnable_parameters(self, pars: LearnableParametersDict[SymType]) -> None:
        """Makes the optimization class aware of the dictionary of the learnable
        parameters.

        Parmeters
        ---------
        pars : LearnableParametersDict
            The dictionary of the learnable parameters.
        """
        self.learnable_parameters = pars
        self._update_solver = self._init_update_solver()

    def _get_update_bounds(
        self, theta: np.ndarray, eps: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Internal utility to retrieve the current bounds on the learnable parameters.
        Only useful if the update problem is not constrained, i.e., there are either
        some lb or ub, or a maximum percentage update."""
        lb = self.learnable_parameters.lb - theta
        ub = self.learnable_parameters.ub - theta
        perc = self.max_percentage_update
        if perc != float("+inf"):
            max_update_delta = np.maximum(np.abs(perc * theta), eps)
            lb = np.maximum(lb, -max_update_delta)
            ub = np.minimum(ub, +max_update_delta)
        return lb, ub

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
        gradient : array
            The gradient of the learnable parameters.
        hessian : array, optional
            The hessian of the learnable parameters. When the optimizer is firt-order,
            must be `None`.

        Returns
        -------
        status : str, optional
            An optional string containing the status of the update, e.g., the status of
            the QP solver, if used.
        """
        if hessian is None:
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
        self,
        gradient: npt.NDArray[np.floating],
        hessian: Optional[npt.NDArray[np.floating]] = None,
    ) -> tuple[npt.NDArray[np.floating], Optional[str]]:
        """Internally runs a second order update."""
        raise NotImplementedError(
            f"`{self.__class__.__name__}` optimizer does not implement "
            "`_second_order_update`"
        )