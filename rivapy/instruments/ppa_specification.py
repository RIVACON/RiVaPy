from typing import Union, Tuple, Iterable, Set, List
import abc
import numpy as np
import pandas as pd
import datetime as dt
import rivapy.tools.interfaces as interfaces
from rivapy.instruments.factory import create as _create
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
	
class PPASpecification(interfaces.FactoryObject):
	def __init__(self, 
				udl: str,
				amount: Union[float, np.ndarray], 
				schedule: Union[SimpleSchedule, List[dt.datetime]],
				fixed_price: float,
				id:str = None):
		"""Specification for a simple power purchase agreement (PPA).

		Args:
			udl (str): Name of underlying (power) that is delivered (just use for consistency checking within pricing against simulated model values).
			amount (Union[None, float, np.ndarray]): Amount of power delivered at each timepoint/period. Either a single value s.t. all volumes delivered are constant or a load table. If None, a non-constant amount (e.g. by production from renewables) is assumed.
			schedule (Union[SimpleSchedule, List[dt.datetime]): Schedule describing when power is delivered.
			fixed_price (float): The fixed price paif for the power.
			id (str): Simple id of the specification. If None, a uuid will be generated. Defaults to None.
		"""
		self.id = id
		self.udl = udl
		if id is None:
			self.id = type(self).__name__+'/'+str(dt.datetime.now())
		self.amount = amount
		if isinstance(schedule, dict): #if schedule is a dict we try to create it from factory
			self.schedule = _create(schedule)
		else:
			self.schedule = schedule
		self.fixed_price = fixed_price
		if isinstance(schedule, list):
			self._schedule_df = pd.DataFrame({'dates': self.schedule}).reset_index()
		else:
			self._schedule_df = self.schedule.get_df().set_index(['dates']).sort_index()
		self._schedule_df['amount'] = amount
		self._schedule_df['flow'] = None

	@staticmethod
	def _create_sample(n_samples: int, seed: int = None, ref_date = None):
		schedules = SimpleSchedule._create_sample(n_samples, seed, ref_date)
		result = []
		for schedule in schedules:
			amount = np.random.uniform(low=50., high=100.0)
			fixed_price = np.random.uniform(low=0.5, high=1.5)
			result.append(PPASpecification(udl='Power', amount=amount, schedule=schedule, fixed_price=fixed_price))
		return result
	
	def get_schedule(self)->List[dt.datetime]:
		if not isinstance(self.schedule, list):
			return self.schedule.get_schedule()
		return self.schedule

	def _to_dict(self)->dict:
		try: # if isinstance(self.schedule, interfaces.FactoryObject):
			schedule = self.schedule.to_dict()
		except  Exception as e:
			schedule = self.schedule
		return {
			'udl': self.udl,
			'id': self.id,
			'amount': self.amount,
			'schedule': schedule,
			'fixed_price': self.fixed_price
		}

	def set_amount(self, amount):
		self.amount = amount
		self._schedule_df['amount'] = amount

	def n_deliveries(self):
		return self._schedule_df.shape[0]

	def compute_flows(self, refdate: dt.datetime, pfc, result: pd.DataFrame=None, result_col = None)->pd.DataFrame:
		df = pfc.get_df()
		if result is None:
			self._schedule_df['flow'] = self._schedule_df['amount']*(df.loc[self._schedule_df.index]['values']-self.fixed_price)
			return self._schedule_df
		result[result_col] = self._schedule_df['amount']*(df.loc[self._schedule_df.index]['values']-self.fixed_price)

class GreenPPASpecification(PPASpecification):
	def __init__(self,
				schedule: Union[SimpleSchedule, List[dt.datetime]],
				fixed_price: float,
				max_capacity: float,
				technology: str,
				udl: str,
				location: str=None,
				id:str = None):
		""" :term:`Specification` for a green power purchase agreement.

		In contrast to a normal PPA the quantities of this PPA are related to some kind of 
		renewable energy such as wind or solar, i.e. the quantity is related to some uncertain production.
		
		Args:
			schedule (Union[SimpleSchedule, List[dt.datetime]]): Delivery schedule.
			fixed_price (float): Fixed price paid for the power.
			max_capacity (float): The absolute maximal capacity of the renewable energy source. This is used to derive the production amount of the plant by multiplying forecasts with the factor max_capacity/total_capacity (where total capacity may be time dependent).
			technology (str): Identifier for the technology. This is used to retrieve the simulated values for production of this technology from a model
			location (str, optional): Identifier for the location. This is used to retrieve the simulated values for production of this technology at this location from a model that supports this feature. Defaults to None.
			udl (str, optional): Name of underlying (power) that is delivered (just use for consistency checking within pricing against simulated model values). It is used within pricing when the respective simulated price must be retrieved from a model's simulation results.
			id (str, optional): Unique identifier of this contract. Defaults to None.
		"""
		super().__init__(udl, None, schedule, fixed_price, id)
		self.technology = technology
		self.max_capacity = max_capacity
		self.location = location

	@staticmethod
	def _create_sample(n_samples: int, seed: int = None, ref_date = None):
		schedules = SimpleSchedule._create_sample(n_samples, seed, ref_date)
		result = []
		for schedule in schedules:
			max_capacity = np.random.uniform(low=50., high=100.0)
			fixed_price = np.random.uniform(low=0.5, high=1.5)
			result.append(GreenPPASpecification(udl='Power', technology='Wind',  location='Onshore', fixed_price=fixed_price, max_capacity=max_capacity, schedule = schedule))
		return result
	
	def _to_dict(self)->dict:
		result = super()._to_dict()
		del result['amount']
		result['technology'] = self.technology
		result['max_capacity'] = self.max_capacity
		result['location'] = self.location
		return result

	def compute_flows(self, refdate: dt.datetime, pfc, forecast_amount: np.ndarray, result: pd.DataFrame=None, result_col = None)->pd.DataFrame:
		self.set_amount(forecast_amount)
		return super().compute_flows(refdate, pfc, result, result_col)

