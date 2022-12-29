
import abc
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

    def hash(self):
        return hashlib.sha1(self.to_json()).hexdigest()
    
    @abc.abstractmethod
    def _to_dict(self)->dict:
        pass

    @classmethod
    def from_dict(cls, data: dict)->object:
        return cls(**{k:v for k,v in data.items() if k != 'cls'})

class BaseDatedCurve(abc.ABC):
    @abc.abstractmethod
    def compute(self, ref_date: dt.datetime)->np.ndarray:#, dt_grid: DateTimeGrid)->np.ndarray:
        pass