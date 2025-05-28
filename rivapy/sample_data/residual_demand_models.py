import numpy as np
from rivapy.models.residual_demand_model import SolarProfile, SolarPowerModel, WindPowerModel, SupplyFunction, ResidualDemandModel, LoadModel
from rivapy.models import OrnsteinUhlenbeck


class WagnerModel:

	@staticmethod
	def solar_profile():
		_monthly_solar_profile = np.zeros((12,25))
		_monthly_solar_profile[0][9:16] = 1.0
		_monthly_solar_profile[1][9:16] = 1.0
		_monthly_solar_profile[2][8:16] = 1.0
		_monthly_solar_profile[3][8:17] = 1.0
		_monthly_solar_profile[4][8:18] = 1.0
		_monthly_solar_profile[5][7:19] = 1.0
		_monthly_solar_profile[6][6:20] = 1.0
		_monthly_solar_profile[7][6:20] = 1.0
		_monthly_solar_profile[8][7:19] = 1.0
		_monthly_solar_profile[9][8:18] = 1.0
		_monthly_solar_profile[10][9:16] = 1.0
		_monthly_solar_profile[11][9:16] = 1.0
		def __monthly_solar_profile(d):
			return _monthly_solar_profile[d.month-1, d.hour]
		return SolarProfile(__monthly_solar_profile)

	@staticmethod
	def solar():
		profile = WagnerModel.solar_profile()
		def mean_level(a1,b1,a2,b2,c):
			def _mean_level(t):
				return a1*np.cos(2.0*np.pi*t.timegrid + b1)+a2*np.cos(4.0*np.pi*t.timegrid + b2)+c
			return _mean_level
		daily_model = OrnsteinUhlenbeck(speed_of_mean_reversion = 261.817, volatility=16.087)
		return SolarPowerModel(daily_model, profile, mean_level(-1.23,0.476,-0.614,0.093,-0.798))

	@staticmethod
	def wind():
		def mean_level(a,b,c):
			def _mean_level(t):
				return a*np.cos(2.0*np.pi*t.timegrid + b)+c
			return _mean_level
		deviation_model = OrnsteinUhlenbeck(speed_of_mean_reversion = 91.151, volatility=15.155)
		return WindPowerModel(deviation_model, seasonal_function=mean_level(0.311, 0.002, -1.999))

	@staticmethod
	def supply():
		return SupplyFunction(floor=(10,-3000), cap=(85,3000), peak=(39.425, 528.343), offpeak=(43.0,-713.804, 0.491), peak_hours=set([8,9,10,11,12,13,14,15,16,17,18]))

	@staticmethod
	def load():
		def __monthly_load_profile(d):
			__monthly_load_profile = 0.5*np.ones((12,25))
			__monthly_load_profile[0][9:16] = 1.0
			__monthly_load_profile[1][9:16] = 1.0
			__monthly_load_profile[2][8:16] = 1.0
			__monthly_load_profile[3][8:17] = 1.0
			__monthly_load_profile[4][8:18] = 1.0
			__monthly_load_profile[5][7:19] = 1.0
			__monthly_load_profile[6][6:20] = 1.0
			__monthly_load_profile[7][6:20] = 1.0
			__monthly_load_profile[8][7:19] = 1.0
			__monthly_load_profile[9][8:18] = 1.0
			__monthly_load_profile[10][9:16] = 1.0
			__monthly_load_profile[11][9:16] = 1.0
			for i in range(__monthly_load_profile.shape[0]):
				__monthly_load_profile[i] *= 20
				__monthly_load_profile[i] += 30
			return __monthly_load_profile[d.month-1, d.hour]
		profile = SolarProfile(__monthly_load_profile) # include w (non-business day correction)
		def mean_level(a,b,c):
			def _mean_level(t):
				return a*np.cos(2.0*np.pi*t + b)+c
			return _mean_level
		daily_model = OrnsteinUhlenbeck(speed_of_mean_reversion = 336.609, volatility=89.265)
		return LoadModel(daily_model, profile)


	@staticmethod
	def residual_demand_model(capacity_wind=25, capacity_solar=20):
		solar = WagnerModel.solar()
		wind = WagnerModel.wind()
		supply = WagnerModel.supply()
		load = WagnerModel.load()
		model = ResidualDemandModel(wind, capacity_wind, solar, capacity_solar,
			      load, supply, power_name='power')
		return model

if __name__=='__main__':
	import datetime as dt
	import matplotlib.pyplot as plt
	from rivapy.tools.datetime_grid import DateTimeGrid
	dg = DateTimeGrid(start=dt.datetime(2022, 1, 1), end=dt.datetime(2022,2,1), freq='1H')
	n_sims = 500
	residual_demand_model = WagnerModel.residual_demand_model()
	result = residual_demand_model.simulate(dg, 
                                start_value_wind = 0.0,
                                start_value_solar = 0.0,
                                start_value_load = 0.0,
                                n_sims = n_sims)

	plt.plot(result['solar'].mean(axis=1) + result['wind'].mean(axis=1))
	plt.show()