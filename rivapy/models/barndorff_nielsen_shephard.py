from typing import Union, Callable
import numpy as np
import scipy
import scipy.stats as ss
from rivapy.tools.interfaces import FactoryObject

class BNS(FactoryObject):

    def _eval_grid(f, timegrid):
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result

    def __init__(self, rho: Union[float, Callable],lmbda: Union[float, Callable],
                  b: Union[float, Callable], a: Union[float, Callable]):
        """Barndorff-Nielson-Shephard Model as in https://perswww.kuleuven.be/~u0009713/ScSiTi03.pdf.
        """
        self.rho = rho
        self.lmbda = lmbda
        self.b = b
        self.a = a
        self.k = self.a*self.rho/(self.b - self.rho)
        self._timegrid = None
        self.modelname = 'BNS'

    def _to_dict(self) -> dict:
        return {'rho': self.rho, 'lmbda':self.lmbda,'b':self.b,'a':self.a}

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
        """ Simulate the BNS Model paths
        """
        self._set_params(S0,v0,M,n)
        self._set_timegrid(timegrid)

        S = np.zeros((self._timegrid.shape[0]+1, M))
        V =  np.zeros((self._timegrid.shape[0]+1, M))
        S[0, :] = np.log(S0)
        V[0, :] = v0
        
        # Generate correlated Brownian motions
        z1 = np.random.normal(size=(self._timegrid.shape[0], M))

        # Generate stock price and volatility paths
        for t in range(1, self._timegrid.shape[0] + 1):
            # Calculate volatility
            vol = np.sqrt(V[t - 1, :])

            P = ss.poisson.rvs(self.a * self._delta_t*t,size=M)
            jumps = np.asarray([np.sum(np.random.exponential(self.b, int(i))) for i in P])

            # Update the stock price and volatility
            S[t, :] = S[t - 1, :]  - (self.lmbda*self.k + 0.5*vol*vol)*self._delta_t + vol * np.sqrt(self._delta_t) * z1[t - 1, :] + self.rho*jumps[:]
            V[t, :] =  V[t - 1, :] - self.lmbda*V[t - 1, :]*self._delta_t
        return np.exp(S)
        


  




