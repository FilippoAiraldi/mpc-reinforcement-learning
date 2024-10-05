------------------------
Gradient-free approaches
------------------------

So far, we tackled the problem of tuning the parameters of the MPC controller with
RL methods that require the computation of gradients; in particula, of the sensitivities
of the MPC optimization problem with respect to its parametrization. However, in many
practical cases, it is often convienient to optimize the performance of the controller
with gradient-free methods. These are particularly useful when the objective function is
expensive to evaluate or non-differentiable.

Global optimization techniques fall under this category. These methods aim to find the
global minimum of a function without the need of its gradient. One of the most popular
global optimization techniques is likely Bayesian Optimization (BO). BO is a sequential
decision making process that uses a probabilistic model to approximate the objective
function and its uncertainty. The model is then used to decide where to evaluate the
objective function next. The process is repeated until a stopping criterion is met.

In the context of MPC, BO can be used to tune the parameters of the controller by
searching for the optimal set of parameters that minimize the cost function. The search
is often efficient and requires a relatively small number of evaluations of the cost
function (usually done via simulation, once a candidate set of parameters is provided by
the algorithm).

TODO: add reference to each example from the theory
