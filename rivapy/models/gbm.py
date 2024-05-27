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
        self.modelname = 'GBM'
        self.v0 = 0.

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


    def simulate(self, timegrid, S0, v0, M,n, model_name):
        """ Simulate the GBM Paths
            
            .. math:: 
                
                X_{t} = X_0 * exp( (\\drift - 0.5 \\sigma**2) t + \\sigma W_t )

            where :math:`W_t` is a (0,1)-normal random variate.
        
        Args:
            timegrid (np.ndarray): One dimensional array containing the time points where the process will be simulated (containing 0.0 as the first timepoint).
            S0 (Union[float, np.ndarray]): Either a float or an array (for each path) with the start value of the simulation.
            M = number of simulations
            n = number of timesteps
            v0 and model_name are currently not used, defined just to be consistent with HestonModel for Deep Hedging.
        Returns:
            np.ndarray: Array r containing the simulations where r[:,i] is the path of the i-th simulation (r.shape[0] equals number of timepoints, r.shape[1] the number of simulations). 
        """
        self._set_params(S0,M,n)
        self._set_timegrid(timegrid)
        St = np.exp( (self.drift - self.volatility ** 2 / 2) * self._delta_t + self.volatility * np.random.normal(0, np.sqrt(self._delta_t), size=(M,n)).T)
        St = np.vstack([np.ones(M), St]) 
        result = S0 * np.cumprod(St,axis=0)
        return result


    def compute_BS_delta(self, S,timegrid):
        self._set_timegrid(timegrid)
        d1 = (np.log(S/1.) + (self.drift + self.volatility*self.volatility*0.5)*(self._delta_t))/(self.volatility*np.sqrt(self._delta_t))
        delta_BS = scipy.stats.norm.cdf(d1)
        return delta_BS


    def compute_call_price(self, S0: Union[float, np.ndarray], K: float, ttm: float):
        """Computes the price of a call option with strike K and time to maturity ttm for a spot following the GBM.
            -> Black Scholes closed formula.

        """
        if ttm < 1e-5:
            return np.maximum(S0 - K, 0.0)
        d1 = (np.log(S0 / K) + (self.drift + self.volatility*self.volatility*0.5) * ttm) / (self.volatility * np.sqrt(ttm))
        d2 = (np.log(S0 / K) + (self.drift - self.volatility*self.volatility*0.5) * ttm) / (self.volatility * np.sqrt(ttm))

        return S0 * scipy.stats.norm.cdf(d1) - K * np.exp(-self.drift*ttm)*scipy.stats.norm.cdf(d2)
