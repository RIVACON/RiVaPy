import abc
from typing import Union, Callable, List, Set
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


    def __init__(self,vol1: Union[float, Callable[], 
                 alpha1: float, 
                 vol2: float, 
                 alpha2: float, 
                 udl:str):
        self.udl = udl
        self.vol1 = vol1
        self.alpha1 = alpha1
        self.vol2 = vol2
        self.alpha2 = alpha2
    
    def rnd_shape(self, n_sims: int, n_timesteps: int)->tuple:
        return (n_timesteps-1, 2*n_sims)

    def simulate(self, timegrid, 
                rnd: np.ndarray,
                expiries: List[float],
                fwd0: List[float])->ForwardSimulationResult:
        