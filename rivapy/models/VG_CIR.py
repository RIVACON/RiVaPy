from typing import Union, Callable
import numpy as np
import scipy
from rivapy.tools.interfaces import FactoryObject
import scipy.stats as ss
from rivapy.models.VG import VG
from scipy.interpolate import interp1d


class VG_CIR(FactoryObject):

    def _eval_grid(f, timegrid):
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result

    def __init__(self, C: Union[float, Callable], G:Union[float, Callable],M:Union[float, Callable],
                 kappa:Union[float, Callable], eta: Union[float, Callable], lmbda: Union[float, Callable],
                 y0:Union[float, Callable]):
        """VG_CIR process as in https://perswww.kuleuven.be/~u0009713/ScSiTi03.pdf

            
        """
        self.C = C
        self.G = G
        self.M = M
        self.kappa = kappa
        self.eta = eta
        self.lmbda = lmbda
        self.y0 = y0
        self._timegrid = None
        self.modelname = 'VG_CIR'

    def _to_dict(self) -> dict:
        return {'C':self.C, 'G':self.G,'M':self.M, 'kappa':self.kappa, 'eta':self.eta,'lmbda':self.lmbda,'y0':self.y0}

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
        y = np.zeros((self._timegrid.shape[0]+1))
        y[0] = self.y0
        for t in range(1, self._timegrid.shape[0]+1):
            y[t] = y[t-1] + self.kappa*(self.eta - y[t-1])*self._delta_t + np.random.normal(0, 1)*np.sqrt(self._delta_t)*self.lmbda*(y[t-1])**0.5

        # calculate the time change
        YY = np.zeros((self._timegrid.shape[0]+1))
        for t in range(0, self._timegrid.shape[0]+1):
            YY[t] =  np.sum(y[0:t])/365.


        #simulate the Levy process
        model = VG(C=self.C,M=self.M,G=self.G) 
        X = model.simulate(YY, S0=1., v0=1, M=self.n_sims,n=self.n,model_name='VG')


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



