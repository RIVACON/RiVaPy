
import abc
from typing import List, Tuple, Protocol, Callable
import datetime as dt
import numpy as np
import json
import hashlib
from rivapy.tools.datetime_grid import DateTimeGrid

class DateTimeFunction(abc.ABC):
    @abc.abstractmethod
    def compute(self, ref_date: dt.datetime, dt_grid: DateTimeGrid)->np.ndarray:
        pass

class _JSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        ret = {}
        for key, value in obj.items():
            if key in {'timestamp', 'whatever'}:
                ret[key] = dt.fromisoformat(value) 
            else:
                ret[key] = value
        return ret

class _JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dt.date, dt.datetime)):#, pd.Timestamp)):
            return obj.isoformat()
        return json.JSONEncoder.default(obj)
        
class FactoryObject(abc.ABC):

    def to_dict(self):
        result = self._to_dict()
        result['cls'] = type(self).__name__
        return result

    def to_json(self):
        return json.dumps(self.to_dict(), cls=_JSONEncoder).encode()

    @classmethod
    def from_json(cls, json_str: str):
        tmp = json.loads(json_str, cls=_JSONDecoder)
        return cls.from_dict(tmp)

    @staticmethod
    def hash_for_dict(data: dict):
        return hashlib.sha1(json.dumps(data, cls=_JSONEncoder).encode()).hexdigest()
    

    def hash(self):
        return FactoryObject.hash_for_dict(self.to_dict())
        
    @abc.abstractmethod
    def _to_dict(self)->dict:
        pass

    @classmethod
    def from_dict(cls, data: dict)->object:
        return cls(**{k:v for k,v in data.items() if k != 'cls'})

class BaseDatedCurve(abc.ABC):
    @abc.abstractmethod
    def value(self, ref_date: dt.datetime, d: dt.datetime)->np.ndarray:#, dt_grid: DateTimeGrid)->np.ndarray:
        pass


class HasExpectedCashflows(abc.ABC):
    @abc.abstractmethod
    def expected_cashflows(self)->List[Tuple[dt.datetime, float]]:
        pass

class ModelDeepHedging(Protocol):
    """Class to define the interfaces for the model used in deep hedging.
    """
    def simulate(self, timegrid: np.ndarray, S0: float, n_sims: int)->np.ndarray:
        """Simulate the model.

        Args:
            timegrid (np.ndarray): Timegrid used for simulation.
            S0 (float): Initial value of the model.
            n_sims (int): Number of simulations.

        Returns:
            np.ndarray: The simulated paths.
        """
        ...

class OptionCalibratableModel(Protocol):
    @abc.abstractmethod 
    def compute_call_price(self, S0: float, K: float, ttm: float) -> float:
        ...
    @abc.abstractmethod
    def get_parameters(self) -> np.ndarray:
        ...
    @abc.abstractmethod
    def set_parameters(self, params: np.ndarray) -> None:
        ...
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]|None :
        return None
    
    def get_nonlinear_constraints(self) -> Tuple[np.ndarray, Callable, np.ndarray]|None:
        return None
    
    def get_linear_constraints(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]|None:
        """Get linear constraints for the optimization.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]|None: The lower bounds, the matrix of the linear constraints and the upper bounds.
        """
        return None
    
