import abc
from typing import Set, List, Union
from numpy import ndarray
from rivapy.tools.interfaces import FactoryObject


class BaseModel(FactoryObject):
    @abc.abstractmethod
    def udls(self)->Set[str]:
        """Return the name of all underlyings modeled

        Returns:
            Set[str]: Set of the modeled underlyings.
        """
        pass

    

class BaseFwdModel(BaseModel):
   
    @staticmethod
    def get_key(udl:str, fwd_expiry: int)->str:
        return udl+'_FWD'+str(fwd_expiry)

    @staticmethod
    def get_expiry_from_key(key: str)->int:
        return int(key.split('_FWD')[-1])

    @staticmethod
    def get_udl_from_key(key: str)->int:
        return key.split('_FWD')[0]
    
class ForwardSimulationResult(abc.ABC):
    @abc.abstractmethod
    def n_forwards(self)->int:
        pass

    @abc.abstractmethod
    def udls(self)->Set[str]:
        pass

    def keys(self)->List[str]:
        result = set()
        for udl in self.udls():
            for i in range(self.n_forwards()):
                result.add(BaseFwdModel.get_key(udl, i))
        return result

    @abc.abstractmethod
    def get(self, key: str)->ndarray:
        pass
