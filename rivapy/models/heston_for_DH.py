from typing import Union, Callable
import numpy as np
import scipy
from rivapy.tools.interfaces import FactoryObject

class HestonForDeepHedging(FactoryObject):

    def _eval_grid(f, timegrid):
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result

    def __init__(self, rate_of_mean_reversion: Union[float, Callable],long_run_average: Union[float, Callable],
                  vol_of_vol: Union[float, Callable], correlation_rho: Union[float, Callable]):
        """Heston Model.

        .. math:: dS_t = \\sqrt{V_t} S_t dB_t; dV_t = \\kappa (\\theta - V_t) dt + \\sigma \\sqrt{V_t} dW_t
            
        where :math:`\\kappa` is the rate of mean reversion, :math:`\\theta` is th long run average variance, :math:`\\sigma` is the vol of vol,
        :math:`dB_t` and :math:`dW_t` are Wiener Processes with correlation_rho

        
        Args:
            rate of mean reversion (Union[float, Callable]): _description_ 
            long_run_average (Union[float, Callable]): _description_
            vol_of_vol (Union[float, Callable]): _description_
            correlation_rho  (Union[float, Callable]): _description_
        """
        self.rate_of_mean_reversion = rate_of_mean_reversion
        self.long_run_average = long_run_average
        self.vol_of_vol = vol_of_vol
        self.correlation_rho = correlation_rho
        self._timegrid = None
        self.modelname = 'Heston'

    def _to_dict(self) -> dict:
        return {'rate_of_mean_reversion': self.rate_of_mean_reversion, 'long_run_average': self.long_run_average,
                'vol_of_vol':self.vol_of_vol , 'correlation_rho': self.correlation_rho}

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
        L = np.zeros((self._timegrid.shape[0]+1,M))
        X = np.zeros((self._timegrid.shape[0]+1, M, 2))
        S[0, :] = S0
        V[0, :] = v0
        
        # Generate correlated Brownian motions
        z1 = np.random.normal(size=(self._timegrid.shape[0], M))
        z2 = self.correlation_rho * z1 + np.sqrt(1 - self.correlation_rho ** 2) * np.random.normal(size=(self._timegrid.shape[0], M))

        # Generate stock price and volatility paths
        for t in range(1, self._timegrid.shape[0] + 1):
            # Calculate volatility
            vol = np.sqrt(V[t - 1, :])


            # Calculate S_k^1 and S_k^2 as in Deep Hedging by Bühler et al. 2019 Section 5.2:
            X[t-1,:,0] = S[t-1,:]
            ttm = self._timegrid[-1]-self._timegrid[t-1]
            L[t-1,:] = ((V[t-1, :] - self.long_run_average)/self.rate_of_mean_reversion)*(1. - np.exp(-self.rate_of_mean_reversion*ttm)) + self.long_run_average*ttm
            X[t-1,:,1] = np.sum(V[:t-1, :],axis=0) + L[t-1,:]


            # Update the stock price and volatility
            S[t, :] = S[t - 1, :] * np.exp((- 0.5 * vol**2) * self._delta_t + vol * np.sqrt(self._delta_t) * z1[t - 1, :])
            V[t, :] = np.maximum(
                0.0, V[t - 1, :] + self.rate_of_mean_reversion * (self.long_run_average - V[t - 1, :]) * self._delta_t 
                + self.vol_of_vol * np.sqrt(V[t - 1, :]) * np.sqrt(self._delta_t) * z2[t - 1, :]
            )

        # Calculate S_k^1 and S_k^2 as in Deep Hedging by Bühler et al. 2019 Section 5.2:
        t = self._timegrid.shape[0] + 1
        X[t-1,:,0] = S[t-1,:]
        X[t-1,:,1] = np.sum(V[:t-1, :],axis=0) 
        
        if model_name == 'Heston with Volswap':
            return X
        else:
            return S
        


    def _characteristic_func(self, xi, s0, v0, tau):
        """Characteristic function needed internally to compute call prices with analytic formula.
		"""
        ixi = 1j * xi
        d = np.sqrt((self.correlation_rho*self.vol_of_vol*ixi - self.rate_of_mean_reversion)**2 - self.vol_of_vol**2*(-ixi-xi**2))
        g = (self.rate_of_mean_reversion - self.correlation_rho*self.vol_of_vol*ixi - d) / (self.rate_of_mean_reversion - ixi * self.correlation_rho * self.vol_of_vol + d)
        ee = np.exp(-d * tau)
        C = self.rate_of_mean_reversion * self.long_run_average / self.vol_of_vol**2 * (
			(self.rate_of_mean_reversion - ixi * self.correlation_rho * self.vol_of_vol - d) * tau - 2. * np.log((1 - g * ee) / (1 - g))
		)
        D = (self.rate_of_mean_reversion - ixi * self.correlation_rho * self.vol_of_vol - d) / self.vol_of_vol**2 * (
			(1 - ee) / (1 - g * ee)
		)
        return np.exp(C + D*v0 + ixi * np.log(s0))
    
	    
    def compute_call_price(self, s0: float, v0: float, K: Union[np.ndarray, float], ttm: Union[np.ndarray, float])->Union[np.ndarray, float]:
        """Computes a call price for the Heston model via integration over characteristic function.
		Args:
			s0 (float): current spot
			v0 (float): current variance
			K (float): strike
			ttm (float): time to maturity
		"""
        if isinstance(ttm, np.ndarray):
            result = np.empty((ttm.shape[0], K.shape[0], ))
            for i in range(ttm.shape[0]):
				#for j in range(K.shape[0]):
					#result[i,j] = self.call_price(s0,v0,K[j], tau[i])
                result[i,:] = self.compute_call_price(s0,v0,K, ttm[i])
            return result

        def integ_func(xi, s0, v0, K, tau, num):
            ixi = 1j * xi
            if num == 1:
                return (self._characteristic_func(xi - 1j, s0, v0, tau) / (ixi * self._characteristic_func(-1j, s0, v0, tau)) * np.exp(-ixi * np.log(K))).real
            else:
                return (self._characteristic_func(xi, s0, v0, tau) / (ixi) * np.exp(-ixi * np.log(K))).real

        if ttm < 1e-3:
            res = (s0-K > 0) * (s0-K)
        else:
            "Simplified form, with only one integration. "
            h = lambda xi: s0 * integ_func(xi, s0, v0, K, ttm, 1) - K * integ_func(xi, s0, v0, K, ttm, 2)
            res = 0.5 * (s0 - K) + 1/scipy.pi * scipy.integrate.quad_vec(h, 0, 500.)[0]  #vorher 500
        return res




