from typing import Union, Callable
import numpy as np
import scipy
from rivapy.tools.interfaces import FactoryObject, ModelDeepHedging, OptionCalibratableModel


class VG(FactoryObject):

    def _eval_grid(f, timegrid):
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result

    def __init__(self, C: Union[float, Callable], G:Union[float, Callable],M:Union[float, Callable]):
        """variance gamma process as in https://perswww.kuleuven.be/~u0009713/ScSiTi03.pdf

            
        """
        self.C = C
        self.G = G
        self.M = M
        self._timegrid = None

    def _to_dict(self) -> dict:
        return {'C':self.C, 'G':self.G,'M':self.M}

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
        for t in range(0, self._timegrid.shape[0]):
            G1 = np.random.gamma(self.C, self.M, self.n_sims)
            G2 =np.random.gamma(self.C, self.G, self.n_sims)
            X[t,:] = G1 - G2
        return X
    
    def _characteristic_func(self, xi, s0, v0, tau):
        """Characteristic function as in https://perswww.kuleuven.be/~u0009713/ScSiTi03.pdf.
		"""
        ixi = 1j * xi
        nom = self.G*self.M
        denom = self.G*self.M + (self.M - self.G)*ixi + xi*xi
        return (nom/denom)**self.C

    def compute_call_price(self, s0: float, K: Union[np.ndarray, float], ttm: Union[np.ndarray, float])->Union[np.ndarray, float]:
        """Computes a call price for the Variance-Gamma model (https://perswww.kuleuven.be/~u0009713/ScSiTi03.pdf) via integration over characteristic function.
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
                result[i,:] = self.compute_call_price(s0,  K, ttm[i])
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
  

