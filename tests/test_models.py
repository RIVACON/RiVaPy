import unittest
import numpy as np
import datetime as dt

import rivapy.models as models
import rivapy.marketdata as mktdata
import rivapy.pricing.analytics as analytics
import rivapy.tools.enums as enums
from rivapy import _pyvacon_available

class LocalVolModelTest(unittest.TestCase):
	
	def test_local_vol_mc_with_ssvi(self):
		"""Simple test where call price from Local Vol MC simulation is compared against BS price
		"""
		if not _pyvacon_available:
			self.assertAlmostEqual(1,1)
			return
		ssvi = mktdata.VolatilityParametrizationSSVI(expiries=[1.0/365, 30/365, 0.5, 1.0], fwd_atm_vols=[0.25, 0.3, 0.28, 0.25], rho=-0.9, eta=0.5, gamma=0.5)
		x_strikes = np.linspace(0.5,1.5,100)
		time_grid = np.linspace(0.0,1.0,100)
		lv = models.LocalVol(ssvi, x_strikes, time_grid)
		n_sims = 100_000
		S = np.ones((n_sims,1))
		np.random.seed(42)
		for i in range(1,time_grid.shape[0]):
			rnd = np.random.normal(size=(n_sims, 1))
			lv.apply_mc_step(S,time_grid[i-1], time_grid[i],rnd, inplace=True)
		strike = 1.0
		call_price = np.mean(np.maximum(S-strike, 0))
		call_price_ref = analytics.compute_european_price_Buehler(strike = strike, maturity=1.0, volatility=ssvi.calc_implied_vol(1.0,strike))
		self.assertAlmostEqual(call_price, call_price_ref, places=2)
	
	def test_local_vol_with_flat_input_vol(self):
		"""Simple test where the flat input volatility is compared with the local volatility calculated with call prices from black scholes
		"""
		x_strikes = np.linspace(0.5,1.5,100)
		time_grid = np.linspace(0.0,1.0,365)
		input_vol = 0.3
		call_prices = np.array([[analytics.compute_european_price_Buehler(strike=k, maturity=t, volatility=input_vol) for k in x_strikes] for t in time_grid])
		lv_model = models.LocalVol(vol_param=None, x_strikes=x_strikes, time_grid=time_grid, call_prices=call_prices)
		var = lv_model.compute_local_var(vol_param=None, call_param = call_prices, x_strikes = x_strikes, time_grid = time_grid)
		vol = np.sqrt(np.abs(var))

		for i,t in enumerate(time_grid):
			if t < 1/365: 
				continue
			elif t < 2/365: 
				factor = 1.0
			else: 
				factor = 2.0
			range_low = np.exp(-factor * input_vol * np.sqrt(t))
			range_up = np.exp(factor * input_vol * np.sqrt(t))
			val_in_range = vol[i,(x_strikes>range_low)&(x_strikes<range_up)]

			self.assertAlmostEqual(input_vol, np.mean(val_in_range), places=2)
	
	def test_compare_local_var_implied_and_call(self):
		"""Simple test where the local volatility of a volatility surface is compared with the local volatility of the corresponding call price surface
		"""
		if not _pyvacon_available:
			self.assertAlmostEqual(1,1)
			return
		x_strikes = np.linspace(0.5,1.5,100)
		time_grid = np.linspace(0.0,1.0,100)
		
		ssvi = mktdata.VolatilityParametrizationSSVI(expiries=[1.0/365, 30/365, 0.5, 1.0], fwd_atm_vols=[0.25, 0.3, 0.28, 0.25], rho=-0.9, eta=0.5, gamma=0.5)
		var_vol = np.sqrt(models.LocalVol.compute_local_var(ssvi, x_strikes, time_grid))
		
		input_vol = np.array([[ssvi.calc_implied_vol(ttm=t, strike=k) for k in x_strikes] for t in time_grid])		
		call_prices = np.array([[analytics.compute_european_price_Buehler(strike=k, maturity=t, volatility=input_vol[i,j]) for j,k in enumerate(x_strikes)] for  i,t in enumerate(time_grid)])

		lv_model = models.LocalVol(vol_param=None, x_strikes=x_strikes, time_grid=time_grid, call_prices=call_prices)
		var_call = np.sqrt(lv_model.compute_local_var(vol_param=None, call_param = call_prices, x_strikes = x_strikes, time_grid = time_grid))
				
		for i,t in enumerate(time_grid):
			if t < 1/365: 
				continue
			range_low = np.exp(-input_vol[i,:] * np.sqrt(t))
			range_up = np.exp(input_vol[i,:] * np.sqrt(t))
			var_vol_range = var_vol[i,(x_strikes>range_low)&(x_strikes<range_up)]
			var_call_range = var_call[i,(x_strikes>range_low)&(x_strikes<range_up)]
			error = np.mean(var_vol_range) - np.mean(var_call_range)
			self.assertLess(error, 2E-2)


class HestonModelTest(unittest.TestCase):
	
	def test_callprice_formula(self):
		"""Test analytic call price formula by comparing with MC simulated values
		"""
		heston = models.HestonModel(long_run_variance=0.3**2, 
                             mean_reversion_speed=0.5 , 
                             vol_of_vol=0.2, 
                             initial_variance=0.1**2, 
                             correlation = -0.9)

		n_sims = 40_000
		np.random.seed(42)
		strikes = np.array([0.9,1.0,1.1])
		timegrid = np.linspace(0.0,1.0,365)
		
		simulated_values = np.empty((n_sims, 2))
		simulated_values[:,0] = 1.0
		simulated_values[:,1] = heston._initial_variance

		
		cp_anayltic = heston.call_price(1.0, heston._initial_variance, K = strikes, ttm = 1.0)

		for i in range(1, timegrid.shape[0]):
			rnd = np.random.normal(size=(n_sims,2))
			heston.apply_mc_step(simulated_values, timegrid[i-1], timegrid[i], rnd, inplace=True)

		for i in range(strikes.shape[0]):
			cp_mc = np.mean(np.maximum(simulated_values[:,0]-strikes[i], 0.0))
			self.assertAlmostEqual(cp_anayltic[i], cp_mc, delta=1e-3)
		
class HestonLocalVolModelTest(unittest.TestCase):
	@staticmethod
	def calc_imlied_vol_grid(expiries, strikes, call_prices):
		vols = np.zeros((expiries.shape[0],strikes.shape[0]))
		for i in range(expiries.shape[0]):
			for j in range(strikes.shape[0]):
				try:
					vols[i,j] = analytics.compute_implied_vol_Buehler(strikes[j], maturity=expiries[i], 
																	price=call_prices[i, j], min_vol=0.01)
				except:
					pass
			
		for i in range(expiries.shape[0]):
			extrapolation = False
			for j in range(int(strikes.shape[0]/2),strikes.shape[0]):
				if vols[i,j] <1e-6:
					vols[i,j] = vols[i,j-1]
					extrapolation = True
			for j in range(23,-1,-1):
				if vols[i,j] <1e-6:
					vols[i,j] = vols[i,j+1]
					extrapolation = True
			if extrapolation:
				print('Extrapolation necessary for expiry ' + str(i))
		return vols

	@staticmethod
	def calc_callprice_MC(time_grid, strikes, n_sims, model):
		if time_grid[0] > 1e-7:
			raise Exception('The time_grid must include 0.0 as first point.')
		call_prices = np.empty((time_grid.shape[0], strikes.shape[0]))
		x = np.empty((n_sims,2))
		x[:,0] = model.get_initial_value()[0]
		x[:,1] = model.get_initial_value()[1]
		for j in range(strikes.shape[0]):
			call_prices[0][j] = np.maximum(x[0,0]-strikes[j], 0.0)
		np.random.seed(42)
		t0 = 0
		for i in range(1, time_grid.shape[0]):
			rnd = np.random.normal(size=(n_sims, 2))
			model.apply_mc_step(x, t0, time_grid[i], rnd, inplace=True)
			for j in range(strikes.shape[0]):
				call_prices[i][j] = np.mean(np.maximum(x[:,0]-strikes[j], 0.0) )
			t0 = time_grid[i]
		return call_prices

	def test_simple(self):
		"""Simple test: The given implied volatility surface equals the heston surface->stoch local variance must equal 1
		"""
		heston = models.HestonModel(long_run_variance=0.3**2, 
                            mean_reversion_speed=0.5, 
                            vol_of_vol=0.2, 
                            initial_variance=0.3**2, 
                            correlation = -0.95)
		x_strikes = np.linspace(0.5, 1.5, num=240)
		time_grid = np.linspace(0.0, 1.0, num=240) 
		call_prices = heston.call_price(1.0, heston._initial_variance, x_strikes, time_grid)
		heston_lv = models.StochasticLocalVol(heston)
		heston_lv.calibrate_MC(None, x_strikes, time_grid, n_sims=10_000, call_prices=call_prices)
		call_prices_sim = HestonLocalVolModelTest.calc_callprice_MC(time_grid, x_strikes, 10_000, heston_lv)
		for t in [80,120,180,239]:
			for k in [80, 100, 120, 140, 160]:
				vol = analytics.compute_implied_vol_Buehler(x_strikes[k], maturity=time_grid[t], 
																			price=call_prices[t, k], min_vol=0.01)
				vol_sim = analytics.compute_implied_vol_Buehler(x_strikes[k], maturity=time_grid[t], 
																			price=call_prices_sim[t, k], min_vol=0.01)
				self.assertTrue(np.abs(vol-vol_sim)< 0.02, 
					'Vol from calibrated model ('+str(vol_sim)+') is not close enough to reference vol('+str(vol)+') for strike/expiry: '
					 + str(x_strikes[k])+'/'+str(time_grid[t]))

	def test_simple_2(self):
		"""Simple test 2: The given implied volatility surface equals a surface from a heston model. Try to calibrate Heston Local Vol with other heson parameters to fit the surface
		"""
		heston = models.HestonModel(long_run_variance=0.3**2, 
                            mean_reversion_speed=0.5, 
                            vol_of_vol=0.2, 
                            initial_variance=0.3**2, 
                            correlation = -0.95)
		heston_2 = models.HestonModel(long_run_variance=0.2**2, 
                            mean_reversion_speed=0.5, 
                            vol_of_vol=0.2, 
                            initial_variance=0.3**2, 
                            correlation = -0.95)
		x_strikes = np.linspace(0.5,1.5, num=240)
		time_grid = np.linspace(0.0, 1.0, num=240) 
		call_prices = heston.call_price(1.0, heston._initial_variance, x_strikes, time_grid)
		heston_lv = models.StochasticLocalVol(heston_2)
		heston_lv.calibrate_MC(None, x_strikes, time_grid, n_sims=10_000, call_prices=call_prices)
		call_prices_sim = HestonLocalVolModelTest.calc_callprice_MC(time_grid, x_strikes, 10_000, heston_lv)
		for t in [80,120,180,239]:
			for k in [80, 100, 120, 140, 160]:
				vol = analytics.compute_implied_vol_Buehler(x_strikes[k], maturity=time_grid[t], 
																			price=call_prices[t, k], min_vol=0.01)
				vol_sim = analytics.compute_implied_vol_Buehler(x_strikes[k], maturity=time_grid[t], 
																			price=call_prices_sim[t, k], min_vol=0.01)
				self.assertTrue(np.abs(vol-vol_sim)< 0.02, 
					'Vol from calibrated model ('+str(vol_sim)+') is not close enough to reference vol('+str(vol)+') for strike/expiry: '
					 + str(x_strikes[k])+'/'+str(time_grid[t]))


class OrnsteinUhlenbeckTest(unittest.TestCase):
	def test_calibration(self):
		"""Simple test for calibration: Simulate a path of a model with fixed params and calibrate new model. Test if parameters are equal (up to MC error).
		"""
		np.random.seed(42)
		timegrid = np.arange(0.0,30.0,1.0/365.0) # simulate on daily timegrid over 30 yrs horizon
		ou_model = models.OrnsteinUhlenbeck(speed_of_mean_reversion = 5.0, volatility=0.1)
		sim = ou_model.simulate(timegrid, start_value=0.2,rnd=np.random.normal(size=(timegrid.shape[0],1)))
		ou_model.calibrate(sim.reshape((-1)),dt=1.0/365.0, method = 'minimum_least_square')
		self.assertAlmostEqual(0.1, ou_model.volatility, places=3)
		self.assertAlmostEqual(0.0, ou_model.mean_reversion_level, places=2)
		self.assertAlmostEqual(5.0, ou_model.speed_of_mean_reversion, delta=0.5)

	def test_expectation(self):
		"""Simple test for analytical expectation: Compare analytical expectation with MC expectation
		"""
		np.random.seed(42)
		timegrid = np.linspace(0.0,1.0,365) # simulate on daily timegrid over 1 yr horizon
		ou_model = models.OrnsteinUhlenbeck(speed_of_mean_reversion = 5.0, volatility=0.1)
		n_sims = 1000
		sim = ou_model.simulate(timegrid, start_value=0.2,rnd=np.random.normal(size=(timegrid.shape[0],n_sims)))
		self.assertAlmostEqual(sim[-1,:].mean(), ou_model.compute_expected_value(0.2,1.0), places=3)

	def test_call_price(self):
		"""Simple test for call price: Compare analytical call price with MC call price
		"""
		K = 1.0
		T=1.0
		timegrid = np.linspace(0.0,T,num=240)
		ou_model = models.OrnsteinUhlenbeck(speed_of_mean_reversion = 2.0, volatility=0.4)
		n_sims = 10_000
		sim = ou_model.simulate(timegrid, start_value=0.2,rnd=np.random.normal(size=(timegrid.shape[0],n_sims)))
		mc_call_price = np.mean(np.maximum(sim[-1,:]-K,0.0))
		analytical_call_price = ou_model.compute_call_price(0.2, K, T)
		self.assertAlmostEqual(mc_call_price, analytical_call_price, places=2)
class LuciaSchwartzTest(unittest.TestCase):
	def test_expectation(self):
		"""Simple test for analytical expectation: Compare analytical expectation with MC expectation
		"""
		np.random.seed(42)
		timegrid = np.linspace(0.0,1.0,365) # simulate on daily timegrid over 1 yr horizon
		ls_model = models.LuciaSchwartz(rho=-0.9, kappa=1.0, sigma1=0.5, mu=0.15, sigma2=0.3)
		n_sims = 4_000
		sim = ls_model.simulate(timegrid, start_value=np.array([0.0,0.0]),rnd=np.random.normal(size=ls_model.rnd_shape(n_sims,timegrid.shape[0])))
		self.assertAlmostEqual(sim[-1,:].mean(), ls_model.compute_expected_value(x0=np.array([0.0,0.0]), T=1.0), places=2)

	def test_forward_simulation(self):
		"""Simple test for forward simulation: Compare forward simulation with initial value (martingale)
		"""
		np.random.seed(42)
		timegrid = np.linspace(0.0,1.0,365)
		ls_model = models.LuciaSchwartz(rho=-0.9, kappa=1.0, sigma1=0.5, mu=0.15, sigma2=0.3)
		n_sims = 40_000
		sim = ls_model.simulate(timegrid, start_value=np.array([0.0,0.0]),
			  forwards=[(1.0,1.0+1.0/365), (1.0,2.0)],
			  rnd=np.random.normal(size=ls_model.rnd_shape(n_sims,timegrid.shape[0])))
		initial_fwd_value = ls_model.compute_fwd_value(x0=np.array([0.0,0.0]), T1=1.0, T2=2.0)
		self.assertAlmostEqual(sim[-1,:,2].mean(), initial_fwd_value, places=2)
		#check that final spot value and forward value is close
		self.assertAlmostEqual(sim[-1,:,0].mean(), sim[-1,:,1].mean(), places=2)

class WindPowerForecastModel(unittest.TestCase):
	def test_initial_forecast(self):
		"""Simple test for initial forecast: Compare initial forecast with initial value (martingale) 
		and the mean at the first expiry of MC simulation
		"""
		params = models.WindPowerForecastModelParameter(n_call_strikes=40, min_strike=-7.0, max_strike=7.0)
		model = models.WindPowerForecastModel('Onshore', speed_of_mean_reversion=0.5, volatility=1.5, params=params)
		timegrid = np.linspace(0.0,1.0, 365)
		np.random.seed(42)
		rnd = np.random.normal(size=model.rnd_shape(1_000, timegrid.shape[0]))
		forecast = 0.8
		results = model.simulate(timegrid, rnd, expiries=[1.0], initial_forecasts=[forecast], startvalue=0.0)
		sim_fwd = results.get('Onshore_FWD0')
		self.assertAlmostEqual(sim_fwd[-1,:].mean(), forecast, places=2)
		self.assertAlmostEqual(sim_fwd[0,0], forecast, places=2)

if __name__ == '__main__':
    unittest.main()