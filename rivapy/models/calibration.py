from typing import Protocol, Tuple, Callable
import abc
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from rivapy.tools.interfaces import OptionCalibratableModel


def calibrate(model: OptionCalibratableModel, 
              call_prices: np.ndarray, 
              ttm: np.ndarray, 
              x_strikes: np.ndarray,
              **optim_args) -> np.ndarray:
    """Calibrates the model to the given call prices. To get rid of dividend handling, we assume that the call prices are prices derived from the Buehler dividend model.

    Args:
        model (OptionCalibratableModel): _description_
        call_prices (np.ndarray): _description_
        ttm (np.ndarray): _description_
        xStrikes (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    def loss(params: np.ndarray) -> float:
        model.set_parameters(params)
        return np.sum((call_prices - np.array([model.compute_call_price(1., K, t) for t, K in zip(ttm, x_strikes)]))**2)
    initial_guess = model.get_parameters()
    bounds = model.get_bounds()
    if bounds is not None:
        bounds = [(b[0],b[1]) for b in zip(bounds[0], bounds[1])]
    constraints = []
    nonlin_con = model.get_nonlinear_constraints()
    if nonlin_con is not None:
        constraints.append(NonlinearConstraint( nonlin_con[1], lb=nonlin_con[0], ub=nonlin_con[2]))
    lin_constraints = model.get_linear_constraints()
    if lin_constraints is not None:
        constraints.append(LinearConstraint(lin_constraints[1], lin_constraints[0], 
                                           lin_constraints[2], keep_feasible=True))
    if len(constraints) == 0:
        constraints = None
    if optim_args is None:
        optim_args={'method':'trust-constr',
                'jac':'2-point',#hess=SR1(),#bounds=bounds,
                'constraints' : nonlin_con, 'tol':1e-4, 
                'options':{"maxiter":1000}} 
    result = minimize(loss, x0=initial_guess, bounds=bounds, constraints=constraints, **optim_args)
    model.set_parameters(result.x)
    return result