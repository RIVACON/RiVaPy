import abc
from typing import Union, List, Set, Tuple
import numpy as np
from scipy.linalg import cholesky
from rivapy.models.base_model import BaseFwdModel, ForwardSimulationResult

class GasFwdModel2Factor(BaseFwdModel):

    class ForwardSimulationResult(abc.ABC):
        def __init__(self, paths:np.ndarray, expiries:List[float], udl: str, simulate_spot: bool):
            self._paths = paths
            self._expiries = expiries
            self._udl = udl
            self._simulated_spot = simulate_spot
        
        def n_forwards(self)->int:
            return len(self._expiries)

        def udls(self)->Set[str]:
            return set([self.udl])

        def keys(self)->List[str]:
            result = set()
            for i in range(self.n_forwards()):
                result.add(BaseFwdModel.get_key(self._udl, i))
            if self._simulated_spot:
                result.add(self._udl+'_SPOT')
            return result

        def get(self, key: str)->np.ndarray:
            if '_SPOT' in key:
                if not self._simulated_spot:
                    raise Exception('No spot simulated.')
                return self._paths[:,:,0]
            offset = 0
            if self._simulated_spot:
                offset = 1
            i = BaseFwdModel.get_expiry_from_key(key)
            return self._paths[:,:,i+offset]


    def __init__(self,
                 vol1:Union[float,Tuple[List[float], List[float]]], 
                 alpha1: float, 
                 vol2: Union[float,Tuple[List[float], List[float]]], 
                 alpha2: float,
                 corr: float,
                 udl:str, 
                 S0: float = None):
        """This model simulates forward prices (optionally together with spot) using a Heath-Jarrow-Morton like model.
        A forward price :math:`F(t,T)` for a forward price expiring at :math:`T`is modeled by

        .. math::

            dF(t,T) = F(t,T)\\left( e^{-\\alpha_1(T-t)}\\sigma_1(t,T) dW_t^1 + e^{-\\alpha_2(T-t)}\\sigma_2(t,T) dW_t^2  \\right)    

        In addition to the forwards, a spot may be modeled that is just the result of the above formula with :math:`F(t,t)`. Note that this is just a rough 
        approximation for gas forwards since normally they have a delivery period (and not just one timepoint).

        Args:
            vol1 (Union[float,Tuple[List[float], List[float]]]): First volatility (:math:`\\sigma_1(t,T)`), either a fixed value (time independent volatility)  or a tuple of timepoints and values that will be interpolated.
            alpha1 (float): Parameter (:math:`\\alpha_1`) defining the decay of the first volatility with respect to teh time to maturity. 
            vol2 (Union[float,Tuple[List[float], List[float]]]):Second volatility (:math:`\\sigma_2(t,T)`), either a fixed value (time independent volatility)  or a tuple of timepoints and values that will be interpolated.
            alpha2 (float): Parameter (:math:`\\alpha_2`) defining the decay of the second volatility with respect to teh time to maturity. 
            corr (float): Correlation between the two brownian motions.
            udl (str): Identifier of the underlying.
            simulate_spot (bool): Indicates if also a spot will be modeled where we use the equation above with :math:`T=t` in each timepoint.
            S0 (float): If not None, a spot will be modeled where we use the equation above with :math:`T=t` in each timepoint.
        Examples:

            .. highlight:: python
		    .. code-block:: python

            >>> 
        """
        self.udl = udl
        self.vol=[vol1,vol2]
        self.alpha = [alpha1,alpha2]
        self.corr = corr
        self._S0 = S0
        self._simulate_spot = (self._S0 is not None)

    def _to_dict(self)->dict:
        return {'udl': self.udl, 'vol1': self.vol[0], 
                'alpha1': self.alpha[0], 'vol2': self.vol[1],
                'alpha2': self.alpha[1], 'corr': self.corr, 
                'S0': self._S0}
    
    def rnd_shape(self, n_sims: int, n_timesteps: int)->tuple:
        return (n_timesteps-1, n_sims,  2)
    
    def udls(self)->Set[str]:
        """Return the name of all underlyings modeled

        Returns:
            Set[str]: Set of the modeled underlyings.
        """
        return set([self.udl])

    def _step(self,t1: float, t2: float):
        zeta1 = np.exp(2.0*self.alpha1*t2)/(2.0*self.alpha1) - np.exp(2.0*self.alpha1*t1)/(2.0*self.alpha1) 
        zeta2 = np.exp(2.0*self.alpha2*t2)/(2.0*self.alpha2) - np.exp(2.0*self.alpha2*t1)/(2.0*self.alpha2) 
        alpha12 = self.alpha1+self.alpha2
        zeta12 = np.exp(2.0*alpha12*t2)/(2.0*alpha12) - np.exp(2.0*alpha12*t1)/(2.0*alpha12)
        r = zeta12/np.sqrt(zeta1*zeta2)

        return r, zeta1, zeta2

    def _compute_vol_grids(self, timegrid: np.ndarray, expiries: List[float]):
        if self._simulate_spot:
            result = np.zeros((timegrid.shape[0],2,len(expiries) + 1))
        else:
            result = np.zeros((timegrid.shape[0],2,len(expiries) ))
        for k in [0,1]:
            for i in range(len(expiries)):
                result[:,k,i] = np.exp(-self.alpha[k]*(expiries[i]-timegrid))
                if isinstance(self.vol[k],(float,int)):
                    result[:,k,i] = result[:,k,i]*self.vol[k]
                else:
                    result[:,k,i] = result[:,k,i]*np.interp(timegrid, self.vol[k][0], self.vol[k][1])
            if self._simulate_spot:
                if isinstance(self.vol[k],(float,int)):
                    result[:,k,-1] = self.vol[k]
                else:
                    result[:,k,-1] = np.interp(timegrid, self.vol[k][0], self.vol[k][1])
        return result
    
    def _simulate_simple(self, timegrid: np.ndarray, 
                rnd: np.ndarray,
                expiries: List[float],
                fwd0: List[float])->ForwardSimulationResult:
        upper_chol = cholesky(np.array([[1.0,self.corr],[self.corr,1.0]]))
        vols = self._compute_vol_grids(timegrid, expiries)
        offset = 0
        if self._simulate_spot:
            result = np.zeros((rnd.shape[0]+1, rnd.shape[1], len(expiries)+1))
            result[0,:,0] = self._S0
            offset = 1
        else:
            result = np.zeros((rnd.shape[0]+1, rnd.shape[1], len(expiries)))
        for i in range(len(expiries)): # set initial values
            result[0,:,i+offset] = fwd0[i]
        for i in range(timegrid.shape[0]-1):
            corr_rnd = rnd[i,:,:] @ upper_chol
            for j in range(len(expiries)):
                if timegrid[i] <= expiries[j]:
                    result[i+1,:,j+offset] = result[i,:,j+offset] + result[i,:,j+offset]*(vols[i,0,j]*corr_rnd[:,0] + vols[i,1,j]*corr_rnd[:,1])*np.sqrt(timegrid[i+1]-timegrid[i])
                else:
                    result[i+1,:,j+offset] = result[i,:,j+offset]
            if self._simulate_spot:
                result[i+1,:,0] = result[i,:,0] + result[i,:,0]*(vols[i,0,-1]*corr_rnd[:,0] + vols[i,1,-1]*corr_rnd[:,1])*np.sqrt(timegrid[i+1]-timegrid[i])
        return result


    def simulate(self, timegrid: np.ndarray, 
                rnd: np.ndarray,
                expiries: List[float],
                fwd0: List[float])->ForwardSimulationResult:
        result =  self._simulate_simple(timegrid, rnd, expiries, fwd0)
        return GasFwdModel2Factor.ForwardSimulationResult(result, expiries, self.udl, simulate_spot=self._simulate_spot)
        result = np.zeros((rnd.shape[0], rnd.shape[1], len(expiries)))
        if isinstance(self.vol1, float):
            pass
        for i in range(len(expiries)):
            result[0,:,i] = fwd0[i]
        #s1 = 
        for i in range(timegrid.shape[0]-1):
            t1 = timegrid[i]
            t2 = timegrid[i+1]
            dt = t2-t1
            r,zeta1,zeta2 = self._step(timegrid[i], timegrid[i+1])

if __name__=='__main__':
    import matplotlib.pyplot as plt
    model = GasFwdModel2Factor(vol1=0.5, 
                 alpha1=5.0, 
                 vol2 = 0.2, 
                 alpha2 = 0.0,
                 corr = 0.0,#0.5,
                 S0 = 100,
                 udl='TTF')
    timegrid = np.linspace(0.0,1.0,365)
    expiries=[1.0,1.5,2.0,3.0,4.0,5.0]
    rnd = np.random.normal(size=model.rnd_shape(n_sims=10_000, n_timesteps=timegrid.shape[0]))
    simulated_values = model.simulate(timegrid, rnd, expiries=expiries, 
                                    fwd0=[100]*len(expiries))