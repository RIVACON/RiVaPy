from typing import Union, Callable
import numpy as np
import scipy
from rivapy.tools.interfaces import FactoryObject
import scipy.stats as ss

class NIG(FactoryObject):

    def _eval_grid(f, timegrid):
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result

    def __init__(self, alpha: Union[float, Callable], beta:Union[float, Callable],delta:Union[float, Callable]):
        """NIG process as in https://perswww.kuleuven.be/~u0009713/ScSiTi03.pdf

            
        """
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self._timegrid = None
        self.modelname = 'NIG'

    def _to_dict(self) -> dict:
        return {'alpha':self.alpha, 'beta':self.beta,'delta':self.delta}

    def _set_timegrid(self, timegrid):
        self._timegrid = np.copy(timegrid)
        self._delta_t = self._timegrid[1]-self._timegrid[0]
        self._sqrt_delta_t = np.sqrt(self._delta_t)

    def _set_params(self,S0,v0,M,n):
        self.S0 = S0
        self.v0 = v0 
        self.n_sims = M 
        self.n = n 


    def simulate(self, timegrid, S0, v0, M,n):
        """ Simulate the paths
            
        """
        
        self._set_params(S0,v0,M,n)
        self._set_timegrid(timegrid)

        np.random.seed(seed=42)

        X = np.zeros((self._timegrid.shape[0], M))
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





