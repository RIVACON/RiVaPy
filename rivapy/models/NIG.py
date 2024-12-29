from typing import Union, Tuple
import numpy as np
import scipy
from rivapy.tools.interfaces import FactoryObject, ModelDeepHedging, OptionCalibratableModel
import scipy.stats as ss

class NIG(FactoryObject, ModelDeepHedging, OptionCalibratableModel):

    def _eval_grid(f, timegrid):
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result

    def __init__(self, alpha: float, 
                 beta:float,
                 delta:float,
                 v0: float):
        """NIG process as in https://perswww.kuleuven.be/~u0009713/ScSiTi03.pdf

            
        """
        self.v0 = v0
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        
    def _to_dict(self) -> dict:
        return {'alpha':self.alpha, 'beta':self.beta,'delta':self.delta, 'v0': self.v0}

    def _set_timegrid(self, timegrid):
        self._timegrid = np.copy(timegrid)
        self._delta_t = self._timegrid[1]-self._timegrid[0]
        self._sqrt_delta_t = np.sqrt(self._delta_t)

    def simulate(self, timegrid: np.ndarray, S0:float|np.ndarray, n_sims: int):
        """ Simulate the paths
            
        """
        
        self._set_timegrid(timegrid)

        np.random.seed(seed=42)

        X = np.zeros((self._timegrid.shape[0], n_sims))
        X[0, :] = S0
        for t in range(1, self._timegrid.shape[0]):
            lam = self.delta*np.sqrt(self.alpha*self.alpha-self.beta*self.beta)
            IG = ss.invgauss.rvs(mu=1, scale=lam, size=self.n_sims)  # The IG RV
            Norm = ss.norm.rvs(0, 1, self.n_sims)  # The normal RV
            nk = self.delta*self.delta*self.beta * IG + self.delta * np.sqrt(IG) * Norm
            X[t,:] = X[t-1,:] + nk
        return X

    def _characteristic_func(self, xi, s0, v0, tau):
        """Characteristic function as in https://perswww.kuleuven.be/~u0009713/ScSiTi03.pdf.
		"""
        ixi = 1j * xi
        sqrt1 = np.sqrt(self.alpha*self.alpha - (self.beta + ixi)**2)
        sqrt2 = np.sqrt(self.alpha*self.alpha - self.beta*self.beta)
        return np.exp(-self.delta*(sqrt1 - sqrt2))

    def compute_call_price(self, s0: float, K: Union[np.ndarray, float], ttm: Union[np.ndarray, float])->Union[np.ndarray, float]:
        """Computes a call price for the NIG model (https://perswww.kuleuven.be/~u0009713/ScSiTi03.pdf) via integration over characteristic function.
		Args:
			s0 (float): current spot
			K (float): strike
			ttm (float): time to maturity
		"""
        if isinstance(ttm, np.ndarray):
            result = np.empty((ttm.shape[0], K.shape[0], ))
            for i in range(ttm.shape[0]):
				#for j in range(K.shape[0]):
					#result[i,j] = self.call_price(s0,v0,K[j], tau[i])
                result[i,:] = self.compute_call_price(s0, K, ttm[i])
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
            h = lambda xi: s0 * integ_func(xi, s0, self.v0, K, ttm, 1) - K * integ_func(xi, s0, self.v0, K, ttm, 2)
            res = 0.5 * (s0 - K) + 1/scipy.constants.pi * scipy.integrate.quad_vec(h, 0, 500.)[0]  #vorher 500
        return res
  
    def set_parameters(self, params: np.ndarray):
        self.alpha = params[0]
        self.beta = params[1]
        self.delta = params[2]
        self.v0 = params[3]

    def get_parameters(self) -> np.ndarray:
        return np.array([self.alpha, self.beta, self.delta, self.v0])

    def get_linear_constraints(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lb = np.array([1e-8, 1e-8])
        ub = np.array([np.inf, np.inf])
        A = np.array([1.0,1.0,0.0,0.0],
                     [1.0,-1.0,0.0,0.0])
        return lb, A, ub
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([1e-8, -np.inf, 1e-8, 1e-8]), np.array([np.inf, np.inf, np.inf, np.inf])   

