from typing import Union, Callable
import numpy as np
import scipy
from rivapy.tools.interfaces import FactoryObject
import scipy.stats as ss
from rivapy.models.NIG import NIG
from scipy.interpolate import interp1d


class NIG_GammaOU(FactoryObject):

    def _eval_grid(f, timegrid):
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result

    def __init__(self, alpha: Union[float, Callable], beta:Union[float, Callable],delta:Union[float, Callable],
                 lmbda:Union[float, Callable], a: Union[float, Callable], b: Union[float, Callable],
                 y0:Union[float, Callable]):
        """NIG_GammaOU process as in https://perswww.kuleuven.be/~u0009713/ScSiTi03.pdf

            
        """
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.lmbda = lmbda
        self.a = a
        self.b = b
        self.y0 = y0
        self._timegrid = None
        self.modelname = 'NIG_GammaOU'

    def _to_dict(self) -> dict:
        return {'alpha':self.alpha, 'beta':self.beta,'delta':self.delta, 'lmbda':self.lmbda, 'a':self.a,'b':self.b,'y0':self.y0}


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

        # simulate the rate of change process
        P = np.zeros((self._timegrid.shape[0]+1))
        jumps = []
        count = 0
        for t in self._timegrid:
            P[count] = ss.poisson.rvs(self.a*self.lmbda * t, size=1)
            count = count + 1
        P[count] = ss.poisson.rvs(self.a*self.lmbda * (self._timegrid[-1]+self._delta_t), size=1)
        for i in P:
            bla = np.sum((- np.log(np.random.uniform(0, 1,int(i)))/self.b)*np.exp(-self.lmbda*self._delta_t*np.random.uniform(0, 1,int(i))))
            jumps.append(bla)       


        y = np.zeros((self._timegrid.shape[0]+1))
        y[0] = self.y0
        for t in range(1, self._timegrid.shape[0]+1):
            y[t] = (1. - self.lmbda*self._delta_t)*y[t-1] + jumps[t]

        # calculate the time change
        YY = np.zeros((self._timegrid.shape[0]+1))
        for t in range(0, self._timegrid.shape[0]+1):
            YY[t] =  np.sum(y[0:t])/365.

        #simulate the Levy process
        model = NIG(alpha=self.alpha,beta=self.beta,delta=self.delta) 
        X = model.simulate(YY, S0=0., v0=1, M=self.n_sims,n=self.n,model_name='NIG')


        # calculate time changed Levy process
        X_Y = np.zeros((self._timegrid.shape[0]+1, M))
        #for i in range(self.n_sims):
        interp_func = interp1d(YY,X,axis=0,  # interpolate along columns
                bounds_error=False,
                kind='linear',
                fill_value=(X[0], X[-1]))
        X_Y[:-1] = interp_func(self._timegrid)
        X_Y[-1] = interp_func(self._timegrid[-1]+self._delta_t)

        # calculate S
        S = np.zeros((self._timegrid.shape[0]+1, M))
        S[0,:] = S0
        for t in range(1, self._timegrid.shape[0]+1):
            for i in range(self.n_sims):
                S[t,i] = S0*np.exp(X_Y[t,i])
        return S





