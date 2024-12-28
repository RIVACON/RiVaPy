from typing import Protocol, Tuple, Callable
import abc
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

class OptionCalibratableModel(Protocol):
    @abc.abstractmethod 
    def compute_call_price(self, S0: float, K: float, ttm: float) -> float:
        ...
    @abc.abstractmethod
    def get_parameters(self) -> np.ndarray:
        ...
    @abc.abstractmethod
    def set_parameters(self, params: np.ndarray) -> None:
        ...
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]|None :
        return None
    
    def get_nonlinear_constraints(self) -> Tuple[np.ndarray, Callable, np.ndarray]|None:
        return None
    


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
    lb, f_constr, ub = model.get_nonlinear_constraints()
    nonlin_con = NonlinearConstraint( f_constr, lb=lb, ub=ub)
    bounds = model.get_bounds()
    if optim_args is None:
        optim_args={'method':'trust-constr',
                'jac':'2-point',#hess=SR1(),#bounds=bounds,
                'constraints' : nonlin_con, 'tol':1e-4, 
                'options':{"maxiter":1000}} 
    result = minimize(loss, x0=initial_guess, bounds=bounds, constraints=nonlin_con, **optim_args)
    model.set_parameters(result.x)
    return result