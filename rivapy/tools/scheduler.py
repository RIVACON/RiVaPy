from typing import Union, Set, List
import numpy as np
import pandas as pd
import datetime as dt
import rivapy.tools.interfaces as interfaces
from rivapy.tools.datetime_grid import DateTimeGrid

class SimpleSchedule(interfaces.FactoryObject):
	def __init__(self, start: dt.datetime, end:dt.datetime, 
					freq: str='1H', weekdays: Set[int] = None, 
					hours: Set[int] = None, tz: str=None ):
		"""Simple schedule of fixed datetime points.

		Args:
			start (dt.datetime): Start of schedule (including this timepoint).
			end (dt.datetime): End of schedule (excluding this timepoint).
			freq (str, optional): Frequency of timepoints. Defaults to '1H'. See documentation for pandas.date_range for further details on freq.
			weekdays (Set[int], optional): List of integers representing the weekdays where the schedule is defined. 
											Integers according to datetime weekdays (0->Monay, 1->Tuesday,...,6->Sunday). 
											If None, all weekdays are used. Defaults to None.
			hours (Set[int], optional): List of hours where schedule is defined. If None, all hours are included. Defaults to None.
			tz (str or tzinfo): Time zone name for returning localized datetime points, for example ‘Asia/Hong_Kong’. 
								By default, the resulting datetime points are timezone-naive. See documentation for pandas.date_range for further details on tz.
		Examples:
		
		.. highlight:: python
		.. code-block:: python

			>>> simple_schedule = SimpleSchedule(dt.datetime(2023,1,1), dt.datetime(2023,1,1,4,0,0), freq='1H')
			>>> simple_schedule.get_schedule()
			[datetime(2023,1,1,0,0,0), datetime(2023,1,1,1,0,0), datetime(2023,1,1,2,0,0), datetime(2023,1,1,3,0,0)]

			# We include only hours 2 and 3 into schedule
			>>> simple_schedule = SimpleSchedule(dt.datetime(2023,1,1), dt.datetime(2023,1,1,4,0,0), freq='1H', hours=[2,3])
			>>> simple_schedule.get_schedule()
			[datetime.datetime(2023, 1, 1, 2, 0), datetime.datetime(2023, 1, 1, 3, 0)]

			# We restrict further to only mondays as weekdays included
			>>> simple_schedule = SimpleSchedule(dt.datetime(2023,1,1), dt.datetime(2023,1,2,4,0,0), freq='1H', hours=[2,3], weekdays=[0])
			>>> simple_schedule.get_schedule()
			[datetime.datetime(2023, 1, 2, 2, 0), datetime.datetime(2023, 1, 2, 3, 0)]
		"""
		self.start = start
		self.end = end
		self.freq = freq
		self.weekdays = weekdays
		self.hours = hours
		self.tz = tz
		self._df = None

	def _to_dict(self)->dict:
		return {
					'start': self.start, 'end':self.end, 
					'freq': self.freq, 'weekdays': self.weekdays, 
					'hours': self.hours, 'tz': self.tz
		}
 
	def get_schedule(self, refdate: dt.datetime = None)->np.ndarray:
		"""Return vector of datetime values belonging to the schedule.

		Args:
			refdate (dt.datetime): All schedule dates are ignored before this reference date. If None, all schedule dates are returned. Defaults to None.

		Returns:
			np.ndarray: Vector of all datetimepoints of the schedule.
		"""
		d_ = pd.date_range(self.start, self.end, freq=self.freq, tz=self.tz, inclusive='left').to_pydatetime()
		if self.weekdays is not None:
			d_ = [d for d in d_ if d.weekday() in self.weekdays]
		if self.hours is not None:
			d_ = [d for d in d_ if d.hour in self.hours]
		if refdate is not None:
			d_ = [d for d in d_ if d >= refdate]
		return d_

	def get_df(self)->pd.DataFrame:
		if self._df is None:
			self._df = pd.DataFrame({'dates': pd.date_range(self.start, self.end, freq=self.freq, tz=self.tz, inclusive='left').to_pydatetime()}).reset_index()
		return self._df

	def applies(self, dates: DateTimeGrid, index: bool)->List[Union[bool, int]]:
		dates.dates
	def get_params(self)->dict:
		"""Return all params as json serializable dictionary.

		Returns:
			dict: Dictionary of all parameters.
		"""
		return {'start': self.start, 'end': self.end, 'freq': self.freq, 'weekdays': self.weekdays, 'hours': self.hours, 'tz': self.tz}

	@staticmethod
	def _create_sample(n_samples: int, seed: int = None, ref_date = None):
		if ref_date is None:
			ref_date = dt.datetime(1980,1,1)
		if seed is not None:
			np.random.seed(seed)
		result = []
		for i in range(n_samples):
			start = ref_date + dt.timedelta(days=np.random.randint(0,100))
			end = start + + dt.timedelta(days=np.random.randint(5,365))
			result.append(SimpleSchedule(start=start, end=end))
		return result