from typing import Union, Callable
import datetime as dt
import abc 
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit as curve_fit
from scipy.interpolate import interp1d
from rivapy.tools.enums import DayCounterType
from rivapy.tools.datetools import DayCounter


class DateTimeGrid:
    def __init__(self, 
                datetime_grid: pd.DatetimeIndex=None,
                start:Union[dt.datetime, pd.Timestamp]=None, 
                end:Union[dt.datetime, pd.Timestamp]=None, 
                freq: str='1H', 
                daycounter:Union[str, DayCounterType]=DayCounterType.Act365Fixed, 
                tz=None,
                inclusive = 'left'):
        if (start is not None) and (datetime_grid is not None):
            raise ValueError('Either datetime_grid or start must be None.')
        if start is not None:
            self.dates = pd.date_range(start, end, freq=freq, tz=tz, inclusive=inclusive).to_pydatetime()
        else:
            self.dates = datetime_grid
        if self.dates is not None:
            if start is None:
                start = self.dates[0]
            self.timegrid = np.array(DayCounter(daycounter).yf(start, self.dates))
            self.shape = self.timegrid.shape
            self.df = pd.DataFrame({'dates': self.dates, 'tg': self.timegrid})
        else:
            self.dates = None
            self.timegrid = None
            self.shape = None
            self.df = None

        
    def get_daily_subgrid(self):
        df = self.df.groupby(by=['dates']).min()
        df = df.reset_index()
        result = DateTimeGrid(None, None, freq='1D')
        result.dates=np.array([d.to_pydatetime() for d in df['dates']])
        result.timegrid = df['tg'].values
        result.shape = result.timegrid.shape
        result.df = pd.DataFrame({'dates': result.dates, 'tg': result.timegrid})
        return result

    # def get_grid_indices(self, dates):
    #     df = pd.DataFrame({'dates': self.dates, 'tg': self.timegrid})
    #     df = df.reset_index()
    #     df = df.set_index('dates')
    #     df_tg = pd.DataFrame({'dates_': dates})
    #     df.join(df_tg)

    def get_day_of_year(self):
        if 'day_of_year' not in self.df.columns:
            self.df['day_of_year'] = self.df.dates.dt.dayofyear
        return self.df['day_of_year']

    def get_day_of_week(self):
        if 'day_of_week' not in self.df.columns:
            self.df['day_of_week'] = self.df.dates.dt.dayofweek
        return self.df['day_of_week']

    def get_hour_of_day(self):
        if 'hour_of_day' not in self.df.columns:
            self.df['hour_of_day'] = self.df.dates.dt.hour
        return self.df['hour_of_day']

    def get_minute_of_day(self):
        if 'minute_of_day' not in self.df.columns:
            self.df['minute_of_day'] = self.df.dates.dt.minute
        return self.df['minute_of_day']


class __TimeGridFunction(abc.ABC):
    @abc.abstractmethod
    def _compute(self, d: dt.datetime)->float:
        pass

    def compute(self, tg: DateTimeGrid, x=None)->np.ndarray:
        if x is None:
            x = np.empty(tg.shape)
        for i in range(tg.shape[0]):
            x[i] = self._compute(tg.dates[i])
        return x
    
class _Add(__TimeGridFunction):
    def __init__(self, f1,f2):
        self._f1 = f1
        self._f2 = f2
    
    def _compute(self, d: dt.datetime)->float:
        return self._f1._compute(d)+self._f2._compute(d)

class _Mul(__TimeGridFunction):
    def __init__(self, f1,f2):
        self._f1 = f1
        self._f2 = f2
    
    def _compute(self, d: dt.datetime)->float:
        return self._f1._compute(d)*self._f2._compute(d)

class _TimeGridFunction(__TimeGridFunction):
    """Abstract base class for all functions that are defined on datetimes

    Args:
        _TimeGridFunction (_type_): _description_

    Returns:
        _type_: _description_
    """
    @abc.abstractmethod
    def _compute(self, d: dt.datetime)->float:
        pass
    
    def __add__(self, other):
        return _Add(self, other)

    def __mul__(self, other):
        return _Mul(self, other)

class MonthlyConstantFunction(_TimeGridFunction):
    def __init__(self, values:list):
        """Function that is constant across a month.

        Args:
            values (list): values[i] contains the value for the (i+1)-th month
        """
        self.values = values

    def _compute(self, d: dt.datetime)->float:
        return self.values[d.month-1]

class HourlyConstantFunction(_TimeGridFunction):
    def __init__(self, values:list):
        """Function that is constant on hours.

        Args:
            values (list): values[i] contains the value for the i-th hour
        """
        self.values = values

    def _compute(self, d: dt.datetime)->float:
        return self.values[d.hour]

class ParametrizedFunction:
    def __init__(self, x: np.ndarray):
        self.x
        
    def __call__(self, x):
        pass
class PeriodicFunction(_TimeGridFunction):
    def __init__(self, f: Callable, frequency: str, ignore_leap_day: bool=True, granularity='D'):
        self.f = f
        self.ignore_leap_day = ignore_leap_day
        self.frequency = frequency
        self.granularity = granularity

    def _compute(self, d: dt.datetime)->float:
        raise NotImplemented()
        return self.f(d.dayofyear)

    def compute(self, tg: DateTimeGrid, x=None)->np.ndarray:
        if self.frequency == 'Y':
            x = tg.get_day_of_year().values
            if self.ignore_leap_day:
                x = np.minimum(x, 365)
            scaler = 1.0/365.0
        elif self.frequency == 'W':
            x = tg.get_day_of_week().values
            #x = x/6.0
            scaler = 1.0/6.0
        else:
            raise ValueError('Unknown frequency ' + self.frequency)
        if self.granularity == 'H':
            x = x + tg.get_hour_of_day().values/(24.0)
        elif self.granularity == 'T':
            x = x + tg.minute_of_day().values/(24.0*60.0)
        x = scaler*x
        return self.f(x)

    def calibrate(self, dates: Union[pd.DatetimeIndex, DateTimeGrid], values: np.ndarray):
        def f(x, *params):
            for i in range(self.f.x.shape[0]):
                self.f.x[i] = params[i]
            return self.compute(x)
        tg = dates
        if not isinstance(tg, DateTimeGrid):
            tg = DateTimeGrid(datetime_grid=dates)
        popt, pcov = curve_fit(f, tg,values,self.f.x)
        f.x = popt
        
    

        
class InterpolatedFunction(_TimeGridFunction):
    def __init__(self, datetime_grid: pd.DatetimeIndex, values: np.ndarray, kind: str='linear', bounds_error=False):
        self._values = values
        self._tg = DateTimeGrid(datetime_grid)
        self.kind = kind
        self._f = interp1d(self._tg.timegrid, self._values, kind=self.kind, fill_value=(values[0], values[-1]), bounds_error=bounds_error)

    def _compute(self, d: dt.datetime)->float:
        raise NotImplemented()
    
    def compute(self, datetime_grid: pd.DatetimeIndex=None)->np.ndarray:
        x = (datetime_grid-self._tg.dates[0]).total_seconds()/pd.Timedelta(days=365).total_seconds()
        return self._f(x)