import numpy as np
import bisect
from rivapy.models.local_vol import LocalVol
import rivapy.numerics.kernel_regression as kernel_regression

class StochasticLocalVol:
	def __init__(self, stoch_vol_model):
		""" Stochastic Local Volatility model

		Args:
			stochastic_vol_model (StochasticVolModel): The underlying stochastic vol model
		"""
		self._stoch_vol_model = stoch_vol_model
		self._stoch_local_variance = None #np.ones(shape=(time_grid.shape[0], x_strikes.shape[0]))
		self._x_strikes = None
		self._time_grid = None

	def calibrate_MC(self,
					vol_param, 
					x_strikes: np.ndarray,
					time_grid: np.ndarray, 
					n_sims, 
					local_var: np.ndarray=None,
					call_prices: np.ndarray=None):
		"""Calibrate the Heston Local Volatility Model using kernel regression.

		This method calibrates the local volatility part of the Heston Model given a volatility parametrization so that the 
		respective implied volatilities from the given vol parametrization are reproduced by the Heston Local Volatility model.
		The calibration is based on kernel regression as described in `Applied Machine Learning for Stochastic Local Volatility Calibration <https://www.frontiersin.org/articles/10.3389/frai.2019.00004/full>`_.

		Args:
			vol_param ([type]): [description]
			x_strikes (np.array): [description]
			time_grid (np.array): [description]
			n_sims ([type]): [description]
			local_var (np.ndarray, optional): [description]. Defaults to None.
			call_prices (np.ndarray, optional): Defaults to None.
		"""
		if local_var is None:
			local_var = LocalVol.compute_local_var(vol_param, x_strikes, time_grid, call_prices)
		self._stoch_local_variance = StochasticLocalVol._calibrate_MC(self._stoch_vol_model,
								x_strikes, time_grid,  n_sims, local_var)
		self._x_strikes = x_strikes
		self._time_grid = time_grid

	
		
	def apply_mc_step(self, x: np.ndarray, t0: float, t1: float, rnd: np.ndarray, inplace: bool = True):
		"""Apply a MC-Euler step for the Heston Local Vol Model for n different paths.

		Args:
			x (np.ndarray): 2-d array containing the start values for the spot and variance. The first column contains the spot, the second the variance values.
			t0 ([type]): [description]
			t1 ([type]): [description]
			rnd ([type]): [description]
		"""
		t0_index = bisect.bisect_left(self._time_grid, t0)
		if t0_index == 0:# or t0_index == self._time_grid.shape[0]:
			slv = self._stoch_local_variance[0]
		elif t0_index == self._time_grid.shape[0]:
			slv = self._stoch_local_variance[-1]
		else:
			dt = self._time_grid[t0_index] - self._time_grid[t0_index-1]
			w1 = (t0-self._time_grid[t0_index-1])/dt
			w2 = (self._time_grid[t0_index] - t0)/dt
			slv = w1*self._stoch_local_variance[t0_index] + w2*self._stoch_local_variance[t0_index-1]
		slv = np.interp(x[:,0], self._x_strikes, slv)
		return self._stoch_vol_model.apply_mc_step(x, t0, t1, rnd, inplace, slv)
		# if False:
		# 	rnd_S = rnd[:,0]
		# 	rnd_V = rnd[:,1]
		# 	rnd_corr_S = np.sqrt(1.0-self._stoch_vol_model._correlation**2)*rnd_S + self._stoch_vol_model._correlation*rnd_V
		# 	S = x_[:,0]
		# 	v = x_[:,1]
		# 	dt = t1-t0
		# 	sqrt_dt = np.sqrt(dt)
		# 	S *= np.exp(-0.5*v*slv*dt + np.sqrt(v*slv)*rnd_corr_S*sqrt_dt)
		# 	v += self._stoch_vol_model._mean_reversion_speed*(self._stoch_vol_model._long_run_variance-v)*dt + self._stoch_vol_model._vol_of_vol*np.sqrt(v)*rnd_V*sqrt_dt
		# 	x_[:,1] = np.maximum(v,0)
		# 	return x_
	def get_initial_value(self)->np.ndarray:
		"""Return the initial value (x0, v0)

		Returns:
			np.ndarray: Initial value.
		"""
		return self._stoch_vol_model.get_initial_value()

	@staticmethod
	def _calibrate_MC(stoch_vol,   
					x_strikes: np.array,
					time_grid: np.array, 
					n_sims, 
					local_var: np.ndarray,
					x0 = 1.0):
		
		# def apply_mc_step( x, t0, t1, rnd, stoch_local_var):
		# 	slv = np.interp(x[:,0], x_strikes, stoch_local_var)
		# 	rnd_S = rnd[:,0]
		# 	rnd_V = rnd[:,1]
		# 	rnd_corr_S = np.sqrt(1.0-stoch_vol._correlation**2)*rnd_S + stoch_vol._correlation*rnd_V
		# 	S = x[:,0]
		# 	v = x[:,1]
		# 	dt = t1-t0
		# 	sqrt_dt = np.sqrt(dt)
		# 	S *= np.exp((0.5*v*slv)*dt + np.sqrt(v*slv)*rnd_corr_S*sqrt_dt)
		# 	v += stoch_vol._mean_reversion_speed*(stoch_vol._long_run_variance-v)*dt + stoch_vol._vol_of_vol*np.sqrt(v)*rnd_V*sqrt_dt
		# 	x[:,1] = np.maximum(v,0)

		stoch_local_variance = np.empty(local_var.shape)
		stoch_local_variance[0] = local_var[0]/stoch_vol._initial_variance
		#now apply explicit euler to get new values for v and S and then apply kernel regression to estimate new local variance
		x = np.empty((n_sims,2))
		initial_value = stoch_vol.get_initial_value()
		x[:,0] = initial_value[0]
		x[:,1] = initial_value[1]
		for i in range(1,time_grid.shape[0]):
			rnd = np.random.normal(size=(n_sims,2))
			slv = np.interp(x[:,0], x_strikes, stoch_local_variance[i-1])
			stoch_vol.apply_mc_step(x, time_grid[i-1], time_grid[i], rnd, True, slv)
			gamma = ( (4.0*np.std(x[:,0])**5) / (3.0*x.shape[0]) )**(-1.0/5.0)
			kr = kernel_regression.KernelRegression(gamma = gamma).fit(x[:,0:1],x[:,1])
			stoch_local_variance[i] = local_var[i]/kr.predict(x_strikes.reshape((-1,1)))
			# overwrite all values at strikes that are outside the simulated spot range
			min_spot = np.min(x[:,0])
			stoch_local_variance[i][x_strikes<min_spot] = stoch_local_variance[i][x_strikes>min_spot][0]
			max_spot = np.max(x[:,0])
			stoch_local_variance[i][x_strikes>max_spot] = stoch_local_variance[i][x_strikes<max_spot][-1]
		return stoch_local_variance
