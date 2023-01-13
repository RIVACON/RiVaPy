from typing import Union, Callable
import numpy as np

class GeometricBrownianMotion:

    def _eval_grid(f, timegrid):
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result

    def __init__(self, drift: Union[float, Callable], 
                    volatility: Union[float, Callable]):
        self.mu = drift
        self.sigma = volatility
        self._timegrid = None

    def _set_timegrid(self, timegrid):
        self._timegrid = np.copy(timegrid)
        self._delta_t = self._timegrid[1:]-self._timegrid[:-1]
        self._sqrt_delta_t = np.sqrt(self._delta_t)

        self._mu = GeometricBrownianMotion._eval_grid(self.mu, timegrid)
        self._sigma = GeometricBrownianMotion._eval_grid(self.sigma, timegrid)
        
    def simulate(self, timegrid, start_value, rnd):
        self._set_timegrid(timegrid)
        result = np.empty((self._timegrid.shape[0], rnd.shape[1]))
        result[0,:] = start_value

        for i in range(self._timegrid.shape[0]-1):
            result[i+1,:] = result[i,:]*np.exp((self.mu-0.5*self.sigma**2)*self._delta_t[i] + self.sigma*self._sqrt_delta_t[i]*rnd[i,:])
                        
        return result