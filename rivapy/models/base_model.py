import abc
from typing import Set
from rivapy.tools.interfaces import FactoryObject

class BaseModel(FactoryObject):
    @abc.abstractmethod
    def udls(self)->Set[str]:
        """Return the name of all underlyings modeled

        Returns:
            Set[str]: Set of the modeled underlyings.
        """
        pass


class BaseFwdModel(FactoryObject):
    @abc.abstractmethod
    def udls(self)->Set[str]:
        """Return the name of all underlyings modeled

        Returns:
            Set[str]: Set of the modeled underlyings.
        """
        pass

    @staticmethod
    def get_key(udl:str, fwd_expiry: int)->str:
        return udl+'_FWD'+str(fwd_expiry)

    @staticmethod
    def get_expiry_from_key(key: str)->int:
        return int(key.split('_FWD')[-1])

    @staticmethod
    def get_udl_from_key(key: str)->int:
        return key.split('_FWD')[0]