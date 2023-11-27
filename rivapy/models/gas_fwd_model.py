import abc
from typing import Union, Callable, List, Set, Tuple
import numpy as np
from rivapy.models.base_model import BaseFwdModel, ForwardSimulationResult

class GassFwdModel2Factor(BaseFwdModel):

    class ForwardimulationResult(abc.ABC):
        def __init__(self, paths:np.ndarray, expiries:List[float], udl: str):
            self._paths = paths
            self._expiries = expiries
            self._udl = udl
        
        def n_forwards(self)->int:
            return len(self.expiries)

        @abc.abstractmethod
        def udls(self)->Set[str]:
            return set([self.udl])

        def keys(self)->List[str]:
            result = set()
            for udl in self.udls():
                for i in range(self.n_forwards()):
                    result.add(BaseFwdModel.get_key(udl, i))
            return result

        @abc.abstractmethod
        def get(self, key: str)->np.ndarray:
            pass


    def __init__(self,
                 vol1:Union[float,Tuple[List[float], List[float]]], 
                 alpha1: float, 
                 vol2: Union[float,Tuple[List[float], List[float]]], 
                 alpha2: float,
                 corr: float,
                 udl:str):
        self.udl = udl
        self.vol1 = vol1
        self.alpha1 = alpha1
        self.vol2 = vol2
        self.alpha2 = alpha2
        self.corr = corr
    
    def rnd_shape(self, n_sims: int, n_timesteps: int)->tuple:
        return (n_timesteps-1, 2*n_sims)

    def _step(self,t1: float, t2: float):
        zeta1 = np.exp(2.0*self.alpha1*t2)/(2.0*self.alpha1) - np.exp(2.0*self.alpha1*t1)/(2.0*self.alpha1) 
        zeta2 = np.exp(2.0*self.alpha2*t2)/(2.0*self.alpha2) - np.exp(2.0*self.alpha2*t1)/(2.0*self.alpha2) 
        alpha12 = self.alpha1+self.alpha2
        zeta12 = np.exp(2.0*alpha12*t2)/(2.0*alpha12) - np.exp(2.0*alpha12*t1)/(2.0*alpha12)
        r = zeta12/np.sqrt(zeta1*zeta2)

        return r, zeta1, zeta2


    def simulate(self, timegrid: np.ndarray, 
                rnd: np.ndarray,
                expiries: List[float],
                fwd0: List[float])->ForwardSimulationResult:
        result = np.zeros((rnd.shape[0], rnd.shape[1], len(expiries)))
        for i in range(len(expiries)):
            result[0,:,i] = fwd0[i]
        for i in range(timegrid.shape[0]-1):
            t1 = timegrid[i]
            t2 = timegrid[i+1]
            dt = t2-t1