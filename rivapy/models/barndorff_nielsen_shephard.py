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
                  b: Union[float, Callable], a: Union[float, Callable], v0: Union[float, Callable]):
        """Barndorff-Nielson-Shephard Model as in https://perswww.kuleuven.be/~u0009713/ScSiTi03.pdf.
        """
        self.rho = rho
        self.lmbda = lmbda
        self.b = b
        self.a = a
        self.k = self.a*self.rho/(self.b - self.rho)
        self._timegrid = None
        self.modelname = 'BNS'
        self.v0 = v0

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

            P = ss.poisson.rvs(self.a * self.lmbda*self._delta_t,size=M)
            jumps = np.asarray([np.sum(np.random.exponential(1./self.b, size=int(i))) for i in P])

            # Update the stock price and volatility
            S[t, :] = S[t - 1, :]  + (-self.lmbda*self.k - 0.5*vol*vol)*self._delta_t + vol * np.sqrt(self._delta_t) * z1[t - 1, :] + self.rho*jumps
            V[t, :] =  np.maximum(
                0.0,V[t - 1, :] - self.lmbda*V[t - 1, :]*self._delta_t + jumps)
        return np.exp(S)
        


    def _characteristic_func(self, xi, s0, v0, tau):
            """Characteristic function needed internally to compute call prices with analytic formula.
            """
            ixi = 1j * xi
            f1 = ixi*self.rho - 0.5*(xi**2 + ixi)*(1.-np.exp(-self.lmbda*tau))/self.lmbda
            f2 = ixi*self.rho - 0.5*(xi**2 + ixi)/self.lmbda
            A = - self.a * self.lmbda *self.rho*tau/(self.b-self.rho)
            B = - 0.5*(xi**2 + ixi)*(1.-np.exp(-self.lmbda*tau))/self.lmbda
            C = (self.b * np.log((self.b - f1)/(self.b - ixi*self.rho)) + f2*self.lmbda*tau)*self.a/(self.b-f2)
            return np.exp(ixi * np.log(s0) + A + v0*B + C)
        
	    
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
  




