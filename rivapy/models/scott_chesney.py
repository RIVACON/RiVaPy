import numpy as np

class ScottChesneyModel:
	def __init__(self,
				 kappa: float,
				 theta: float,
				 alpha: float,
				 correlation: float,
				 y0: float,
				 ):
		""" Scott-Chesney Model
		Generates a timeseries according to

			.. math:: dS =  e^y S dW_S
			.. math:: dy = `{\kappa}` (`{\theta}`-y)dt `{\alpha}` dW_y
			.. math:: E[dW_s\\dot dW_y] = \\rho dt



		Args:
			kappa (float): speed of mean reversion
			theta (float): mean reversion level
			alpha (float): vol of (log)vol
			correlation (float): correlation between (log)vol and spot
			y0: start value (float): (log) vol
			
		"""
		self.kappa = kappa
		self.theta = theta
		self.alpha = alpha
		self._correlation =correlation
		self.y0 = y0


	def apply_mc_step(self, x: np.ndarray, t0: float, t1: float, rnd: np.ndarray, inplace: bool = True, slv: np.ndarray= None):
		"""Apply a MC-Euler step for the Scott-Chesney Model for n different paths.

		Args:
			x (np.ndarray): 2-d array containing the start values for the spot and variance. The first column contains the spot, the second the variance values.
			t0 ([type]): [description]
			t1 ([type]): [description]
			rnd ([type]): [description]
			slv (np.ndarray): Stochastic local variance (for each path) to be multiplied with the heston variance. This is used by the StochasticVolatilityModel.
		"""
		if not inplace:
			x_ = x.copy()
		else:
			x_ = x
		rnd_corr_S = np.sqrt(1.0-self._correlation**2)*rnd[:,0] + self._correlation*rnd[:,1]
		rnd_V = rnd[:,1]
		S = x_[:,0]
		y = x_[:,1]
		dt = t1-t0
		sqrt_dt = np.sqrt(dt)
		if slv is None:
			slv=1.0

		S += np.sqrt(slv)*np.exp(y) * S * rnd_corr_S * sqrt_dt
		y += self.kappa * (self.theta - y) * dt + self.alpha * rnd_V * sqrt_dt
		return x_


