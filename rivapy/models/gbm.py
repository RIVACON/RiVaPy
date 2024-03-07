from typing import Union, Callable
import numpy as np
import scipy
from rivapy.tools.interfaces import FactoryObject

class GBM(FactoryObject):

    def _eval_grid(f, timegrid):
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result

    def __init__(self, drift: Union[float, Callable],
                    volatility: Union[float, Callable]):
        """Geometric brownian motion stochastic process.

        .. math:: dX = \\mu X dt + \\sigma X dW_t
            
        where :math:`\\mu` is the drift and :math:`\sigma` is the volatility of the process. 

        
        Args:
            drift (Union[float, Callable]): The 
            volatility (Union[float, Callable]): _description_
        """
        self.drift = drift
        self.volatility = volatility
        self._timegrid = None

    def _to_dict(self) -> dict:
        return {'drift': self.drift, 'volatility': self.volatility}

    def _set_timegrid(self, timegrid):
        self._timegrid = np.copy(timegrid)
        self._delta_t = self._timegrid[1]-self._timegrid[0]
        self._sqrt_delta_t = np.sqrt(self._delta_t)

    def _set_params(self,start_value,M,n):
        self.start_value = start_value #S0
        self.n_sims = M 
        self.n = n #length of timegrid


    def simulate(self, timegrid, start_value, M,n):
        """ Simulate the GBM Paths
            
            .. math:: 
                
                X_{t} = X_0 * exp( (\\drift - 0.5 \\sigma**2) t + \\sigma W_t )

            where :math:`W_t` is a (0,1)-normal random variate.
        
        Args:
            timegrid (np.ndarray): One dimensional array containing the time points where the process will be simulated (containing 0.0 as the first timepoint).
            start_value (Union[float, np.ndarray]): Either a float or an array (for each path) with the start value of the simulation.
            M = number of simulations
            n = number of timesteps
        Returns:
            np.ndarray: Array r containing the simulations where r[:,i] is the path of the i-th simulation (r.shape[0] equals number of timepoints, r.shape[1] the number of simulations). 
        """
        self._set_params(start_value,M,n)
        self._set_timegrid(timegrid)
        St = np.exp( (self.drift - self.volatility ** 2 / 2) * self._delta_t + self.volatility * np.random.normal(0, np.sqrt(self._delta_t), size=(M,n)).T)
        St = np.vstack([np.ones(M), St]) 
        result = start_value * np.cumprod(St,axis=0)
        return result



