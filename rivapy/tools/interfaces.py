
import abc
from typing import List, Tuple
import datetime as dt
import numpy as np
import pandas as pd
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
        if '__class__'  in obj:
            if obj['__class__'] == 'datetime':
                res = pd.Timestamp(dt.datetime.strptime(obj['__value__'], "%Y-%m-%d %H:%M:%S"))
                if '__tz__' in obj:
                    if not obj['__tz__'] is None:
                        # saved as UTC
                        res = res.tz_localize('UTC')
                        res = res.tz_convert(obj['__tz__'])
                return res
            elif obj['__class__'] == 'date':
                res = dt.datetime.strptime(obj['__value__'], "%Y-%m-%d").date()
                return res
        return obj

class _JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, dt.datetime) or isinstance(obj, pd.Timestamp):
            if obj.tzinfo is None:
                mytz = None
            else:
                # saving as UTC
                mytz = str(obj.tzinfo)
                obj = pd.Timestamp(obj).tz_convert('UTC')
            return  {'__class__': 'datetime',
                    '__tz__'   : mytz,
                    '__value__' : obj.strftime("%Y-%m-%d %H:%M:%S")
                }
        elif isinstance(obj, dt.date):
            return  {'__class__': 'date',
                    '__value__': str(obj)
                }               
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
