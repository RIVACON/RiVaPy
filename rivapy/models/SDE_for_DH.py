from typing import Union, Callable
import numpy as np
import scipy
from rivapy.tools.interfaces import FactoryObject

class SDEForDeepHedging(FactoryObject):

    def _eval_grid(f, timegrid):
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result

    def __init__(self, a0: Union[float, Callable], a1:Union[float, Callable],b0:Union[float, Callable],b1:Union[float, Callable],gamma:Union[float, Callable]):
        """stochastic process as in https://arxiv.org/pdf/2106.10024.pdf

        .. math:: dX_t = (b0 + b1 X_t)dt + (a0 + a1 X_t)^gamma dW_t
            
        """
        self.a0 = a0
        self.a1 = a1
        self.b0 = b0
        self.b1 = b1
        self.gamma = 2.*gamma
        self._timegrid = None
        self.modelname = 'SDEforDH'

    def _to_dict(self) -> dict:
        return {'a0':self.a0, 'a1':self.a1,'b0':self.b0,'b1':self.b1,'gamma':self.gamma}

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
        S = np.zeros((self._timegrid.shape[0]+1, M))
        S[0, :] = S0
        z1 = np.random.normal(size=(self._timegrid.shape[0]+1, M))
        for t in range(1, self._timegrid.shape[0]):
            S[t,:] = S[t-1,:] + (self.b0 + self.b1 * S[t-1,:])*self._delta_t + (self.a0 + self.a1*np.max(S[t-1,:],0))**self.gamma *  np.sqrt(self._delta_t) * z1[t - 1, :]


        return S



