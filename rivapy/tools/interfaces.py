import abc
import enum
from typing import List, Tuple
import datetime as dt
import numpy as np
import json
import hashlib
from rivapy.tools.datetime_grid import DateTimeGrid
from rivapy.tools.holidays_compat import ECB, UnitedStates, Germany


class DateTimeFunction(abc.ABC):
    @abc.abstractmethod
    def compute(self, ref_date: dt.datetime, dt_grid: DateTimeGrid) -> np.ndarray:
        pass


class _JSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        ret = {}
        for key, value in obj.items():
            if key in {"timestamp", "whatever"}:
                ret[key] = dt.fromisoformat(value)
            else:
                ret[key] = value
        return ret


class _JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dt.date, dt.datetime)):  # , pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, enum.Enum):
            return obj.value
        # return json.JSONEncoder.default(obj)
        return super().default(obj)


class FactoryObject(abc.ABC):

    def to_dict(self):
        result = self._to_dict()
        result["cls"] = type(self).__name__
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
    def _to_dict(self) -> dict:
        pass

    @classmethod
    def from_dict(cls, data: dict) -> object:
        from datetime import datetime, date

        def parse_date(val):
            if isinstance(val, str):
                try:
                    # Try parsing as datetime first
                    return datetime.fromisoformat(val)
                except ValueError:
                    return val
            return val

        CALENDAR_REGISTRY = {
            "ECB": ECB,
            "UnitedStates": UnitedStates,
            "Germany": Germany,
            # Add more as needed
        }

        def calendar_from_name(name):
            cls = CALENDAR_REGISTRY.get(name)
            if cls:
                return cls()
            raise ValueError(f"Unknown calendar: {name}")

        # List of keys that may be date/datetime fields (customize as needed)
        date_keys = ["issue_date", "trade_date", "maturity_date", "start_date", "end_date", "rate_start_date", "rate_end_date", "fixing_date"]
        parsed_data = {}
        for k, v in data.items():
            if k != "cls" and k in date_keys:
                parsed_data[k] = parse_date(v)
            elif k == "calendar":
                parsed_data[k] = calendar_from_name(v)
            elif k != "cls":
                parsed_data[k] = v
        return cls(**parsed_data)


class BaseDatedCurve(abc.ABC):
    @abc.abstractmethod
    def value(self, ref_date: dt.datetime, d: dt.datetime) -> np.ndarray:  # , dt_grid: DateTimeGrid)->np.ndarray:
        pass
