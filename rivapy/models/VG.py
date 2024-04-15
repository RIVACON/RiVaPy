from typing import Union, Callable
import numpy as np
import scipy
from rivapy.tools.interfaces import FactoryObject
import scipy.stats as ss

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
        self.modelname = 'VG'

    def _to_dict(self) -> dict:
        return {'C':self.C, 'G':self.G,'M':self.M}

    def _set_timegrid(self, timegrid):
        self._timegrid = np.copy(timegrid)
        self._delta_t = self._timegrid[1]-self._timegrid[0]
        self._sqrt_delta_t = np.sqrt(self._delta_t)

    def _set_params(self,S0,v0,M,n):
        self.S0 = S0
        self.v0 = v0 
        self.n_sims = M 
        self.n = n 


    def simulate(self, timegrid, S0, v0, M,n, model_name):
        """ Simulate the paths
            
        """
        
        self._set_params(S0,v0,M,n)
        self._set_timegrid(timegrid)

        np.random.seed(seed=42)

        X = np.zeros((self._timegrid.shape[0], M))
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



