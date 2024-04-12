from typing import Union, Callable
import numpy as np
import scipy
import scipy.stats as ss
from rivapy.tools.interfaces import FactoryObject

class HestonWithJumps(FactoryObject):

    def _eval_grid(f, timegrid):
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result

    def __init__(self, rate_of_mean_reversion: Union[float, Callable],long_run_average: Union[float, Callable],
                  vol_of_vol: Union[float, Callable], correlation_rho: Union[float, Callable],
                  muj: Union[float, Callable],sigmaj: Union[float, Callable],lmbda: Union[float, Callable]):
        """Heston Model with Jumps as in https://perswww.kuleuven.be/~u0009713/ScSiTi03.pdf.
        """
        self.rate_of_mean_reversion = rate_of_mean_reversion
        self.long_run_average = long_run_average
        self.vol_of_vol = vol_of_vol
        self.correlation_rho = correlation_rho
        self.muj = muj
        self.sigmaj = sigmaj
        self.lmbda = lmbda
        self._timegrid = None
        self.modelname = 'Heston with Jumps'

    def _to_dict(self) -> dict:
        return {'rate_of_mean_reversion': self.rate_of_mean_reversion, 'long_run_average': self.long_run_average,
                'vol_of_vol':self.vol_of_vol , 'correlation_rho': self.correlation_rho,'muj':self.muj,'sigmaj':self.sigmaj,'lmbda':self.lmbda}

    def _set_timegrid(self, timegrid):
        self._timegrid = np.copy(timegrid)
        self._delta_t = self._timegrid[1]-self._timegrid[0]
        self._sqrt_delta_t = np.sqrt(self._delta_t)

    def _set_params(self,S0,v0,M,n):
        self.S0 = S0
        self.v0 = v0
        self.n_sims = M 
        self.n = n #length of timegrid


    def simulate(self, timegrid, S0, v0, M,n,model_name):
        """ Simulate the Heston Model Paths
        
        
        Args:
            timegrid (np.ndarray): One dimensional array containing the time points where the process will be simulated (containing 0.0 as the first timepoint).
            S0 (Union[float, np.ndarray]): Either a float or an array (for each path) with the start value of the simulation.
            v0 (Union[float, np.ndarray]): Either a float or an array (for each path) with the start value of the simulation.
            M = number of simulations
            n = number of timesteps
        Returns:
            np.ndarray: Array r containing the simulations where r[:,i] is the path of the i-th simulation (r.shape[0] equals number of timepoints, r.shape[1] the number of simulations). 
        """
        self._set_params(S0,v0,M,n)
        self._set_timegrid(timegrid)

        S = np.zeros((self._timegrid.shape[0]+1, M))
        V =  np.zeros((self._timegrid.shape[0]+1, M))
        S[0, :] = S0
        V[0, :] = v0
        
        # Generate correlated Brownian motions
        z1 = np.random.normal(size=(self._timegrid.shape[0], M))
        z2 = self.correlation_rho * z1 + np.sqrt(1 - self.correlation_rho ** 2) * np.random.normal(size=(self._timegrid.shape[0], M))

        # Generate stock price and volatility paths
        for t in range(1, self._timegrid.shape[0] + 1):
            # Calculate volatility
            vol = np.sqrt(V[t - 1, :])

            P = ss.poisson.rvs(self.lmbda * self._delta_t*t,size=M)
            jumps = np.asarray([np.sum(ss.norm.rvs(self.muj, self.sigmaj, int(i))) for i in P])

            # Update the stock price and volatility
            S[t, :] = S[t - 1, :]  - self.lmbda*self.muj*self._delta_t + vol * np.sqrt(self._delta_t) * z1[t - 1, :] + jumps[:]
            V[t, :] = np.maximum(
                0.0, V[t - 1, :] + self.rate_of_mean_reversion * (self.long_run_average - V[t - 1, :]) * self._delta_t 
                + self.vol_of_vol * np.sqrt(V[t - 1, :]) * np.sqrt(self._delta_t) * z2[t - 1, :]
            )
        return S
        


  




