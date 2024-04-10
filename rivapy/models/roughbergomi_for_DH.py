from typing import Union, Callable
import numpy as np
from scipy import stats
from scipy.optimize import least_squares
from scipy.stats import norm
from scipy.linalg import sqrtm
from scipy.special import hyp2f1
from scipy.interpolate import splrep, splev
from rivapy.tools.interfaces import FactoryObject

class rBergomiForDeepHedging(object):
    """
    Class for generating paths of the rBergomi model. https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/212261203---Callum-Rough---ROUGH_CALLUM_01333836.pdf
    """

    def _eval_grid(f, timegrid):
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result


    def __init__(self, H: Union[float, Callable], eta: Union[float, Callable], rho: Union[float, Callable]):
        """
        Rough Bergomi Model.
        """
        self.H = H # Hurst parameter
        self.gamma = 0.5 - H
        self.eta = eta
        self.rho = rho # correlation

        self._timegrid = None
        self.modelname = 'rBergomi'

    def _to_dict(self) -> dict:
        return {'H': self.H, 'eta': self.eta, 'rho':self.rho}

    def _set_timegrid(self, timegrid):
        self._timegrid = np.copy(timegrid)
        self._delta_t = self._timegrid[1]-self._timegrid[0]
        self._sqrt_delta_t = np.sqrt(self._delta_t)


    def _set_params(self,S0,v0,M,n):
        self.S0 = S0
        self.v0 = v0 
        self.n_sims = M 
        self.s = n 

    
    def covW_fun_aux(self, x):
        assert x <= 1
        return ((1 - 2 * self.gamma) / (1 - self.gamma)) * (x**(self.gamma)) * hyp2f1(1, self.gamma, 2 - self.gamma, x)

    def covW_fun(self,u, v):
        if u < v:
            return self.covW_fun(v, u)
        return v**(2*self.H) * self.covW_fun_aux(v/u)

    def covWZ_fun(self,u, v):
        H_tilde = self.H + .5
        D = np.sqrt(2*self.H) / H_tilde
        return self.rho * D * (u ** H_tilde - (u - min(u, v)) ** H_tilde)

    def covW_Z(self, timegrid):
        time_range = timegrid[1:]
        covWW2 = np.zeros((self.s, self.s))
        for i in range(self.s-1):
            for j in range(self.s-1):
                covWW2[i][j] = self.covW_fun(time_range[i], time_range[j])


        covWZ2 = np.zeros((self.s, self.s))
        for i in range(self.s-1):
            for j in range(self.s-1):
                covWZ2[i, j] = self.covWZ_fun(time_range[i], time_range[j])


        covZZ2 = np.zeros((self.s, self.s))
        for i in range(self.s-1):
            for j in range(self.s-1):
                covZZ2[i, j] = min(time_range[i], time_range[j])
        
        cov_matrix = np.bmat([[covWW2, covWZ2], [covWZ2.T, covZZ2]])
        return cov_matrix
    
    def simul_W_Z(self, timegrid):
        self.cov_matrix=self.covW_Z(timegrid)
        self.sqrtm_cov_matrix = sqrtm(self.cov_matrix)
        G = np.random.randn(2 * self.s) 
        WZ_sample = np.dot(self.sqrtm_cov_matrix, G) 
        W_sample, Z_sample = WZ_sample[:self.s], WZ_sample[self.s:]
        W_sample = np.insert(W_sample,0,0)
        Z_sample = np.insert(Z_sample,0,0)
        return W_sample, Z_sample
    
    def simulate(self, timegrid, S0, v0, M,n):

        self._set_params(S0,v0,M,n)
        self._set_timegrid(timegrid)
        
        Ss = np.zeros((self._timegrid.shape[0], M))
        for i in range(self.n_sims):
            W_sample,Z_sample = self.simul_W_Z(self._timegrid)
            v_sample = self.v0**2 * np.exp(self.eta * W_sample[:-1] - 0.5 * (self.eta**2) * self._timegrid**(2*self.H))
            int_sqrtv_dZ = np.cumsum(np.sqrt(v_sample[:]) * (Z_sample[1:] - Z_sample[:-1]))
            int_sqrtv_dZ = np.insert(int_sqrtv_dZ,0,0)
            v_sample[0] = 0
            int_v_dt = np.cumsum(v_sample[:]* self._delta_t)
            S = S0*np.exp(int_sqrtv_dZ[:-1] - .5 * int_v_dt)
            Ss[:,i] = S
        return Ss
        