import bisect
import numpy as np
from scipy import interpolate

def _interpolate_2D(time_grid, strikes, f, x, t):
    t_index = bisect.bisect_left(time_grid, t)
    if t_index == 0:# or t0_index == self._time_grid.shape[0]:
        result = f[0]
    elif t_index == time_grid.shape[0]-1:
        result = f[-1]
    else:
        dt = time_grid[t_index] - time_grid[t_index-1]
        w1 = (t-time_grid[t_index-1])/dt
        w2 = (time_grid[t_index] - t)/dt
        result = w1*f[t_index] + w2*f[t_index-1]
    return np.interp(x, strikes, result)

class LocalVol:

    def __init__(self, vol_param, x_strikes: np.array, time_grid: np.array, call_prices: np.ndarray=None, local_vol_grid: np.ndarray=None):
        """Local Volatility Class 

        Args:
            vol_param: a grid or a parametrisation of the volatility
            x_strikes (np.array): strikes
            time_grid (np.array): time_grid
            call_param (np.ndarray, optional): A grid of call prices. Not compatible with vol_param. Defaults to None.
        """

        if (vol_param is None) and (call_prices is None) and (local_vol_grid is None):
            raise Exception('Set vol_params, call_params or local_vol_grid!')

        if (vol_param is not None) and (call_prices is not None) and (local_vol_grid is not None):
            raise Exception('Set either vol_params or call_params or local_vol_grid, not all!')

        if (vol_param is not None) and (call_prices is not None):
            raise Exception('Set either vol_params or call_params, not both!')

        if (vol_param is not None) and (local_vol_grid is not None):
            raise Exception('Set either vol_params or local_vol_grid, not both!')

        if (local_vol_grid is not None) and (call_prices is not None):
            raise Exception('Set either local_vol_grid or call_params, not both!')

        self._x_strikes = x_strikes
        self._time_grid = time_grid

        if local_vol_grid is not None:
            self._local_variance = local_vol_grid**2
        else:
            self._local_variance = LocalVol.compute_local_var(vol_param, x_strikes, time_grid, call_prices)

        self._variance = interpolate.RectBivariateSpline(time_grid, x_strikes, self._local_variance, bbox=[None, None, None, None], kx=1, ky=1, s=0)
                #interpolation.interp2d(time_grid, x_strikes, self._local_variance.T)

    @staticmethod
    def _compute_local_var_from_vol(vol_param, x_strikes: np.array, time_grid: np.array):
        # setup grids 
        eps = 1e-8
        log_x_strikes = np.log(x_strikes)
        if isinstance(vol_param, np.ndarray):
            iv = np.copy(vol_param)
        else:
            iv = np.empty(shape=(time_grid.shape[0], x_strikes.shape[0])) #implied variance grid
            for i in range(time_grid.shape[0]):
                for j in range(x_strikes.shape[0]):
                    iv[i,j] = vol_param.calc_implied_vol(time_grid[i], x_strikes[j])
        iv *= iv
        tiv = np.maximum((time_grid*iv.T).T, eps)
        h = log_x_strikes[1:] - log_x_strikes[:-1]
        hm = h[:-1]
        hp = h[1:]
        fd1a = -1.0 / (2 * hm)
        fd1c = 1.0 / (2 * hp)
        fd1b = -(fd1c + fd1a)
        fd2a = 2.0 / (hm*(hm + hp))
        fd2c = 2.0 / (hp*(hm + hp))
        fd2b = -(fd2a + fd2c)

        min_lv = 0.01
        max_lv = 1.5
        inv_dt = 1.0/(time_grid[1:]-time_grid[:-1])

        dyw = fd1a*tiv[:,:-2] + fd1b*tiv[:,1:-1] + fd1c*tiv[:,2:]
        dyyw = fd2a*tiv[:,:-2] + fd2b*tiv[:,1:-1] + fd2c*tiv[:,2:]
        dtw = np.maximum((inv_dt*(tiv[1:,:]-tiv[:-1,:]).T).T,eps)

        p = log_x_strikes[1:-1] / tiv[:,1:-1]
        q = np.maximum(1 - p*dyw + 0.25*(-0.25 - 1.0 / tiv[:,1:-1] + p*p)*dyw*dyw + 0.5*dyyw, eps)
        local_var = np.empty(shape=(time_grid.shape[0], x_strikes.shape[0])) 
        local_var[1:-1,1:-1] = np.minimum(np.maximum(min_lv*min_lv, dtw[:-1,1:-1] / q[1:-1,:]), max_lv*max_lv)
        local_var[:,-1] = local_var[:,-2]
        local_var[:,0] = local_var[:, 1]
        local_var[0,:] = local_var[1,:]
        local_var[-1,:] = local_var[-2,:]
        return local_var

    @staticmethod
    def _compute_local_var_from_call(call_param: np.ndarray, x_strikes:np.ndarray, time_grid:np.ndarray):
        """
        Calculate the local volatility from a call price surface with Dupire's equation
            
        sigma^2(K,T) = d_T(C) / (1/2*K^2 * d_KK(C))
        
        with 
        
        dx1 = x_i - x_i-1
        dx2 = x_i+1 - x_i

        d_T[i,:] = 1/(d_t1+d_t2) * [(d_t1/d_t2)*(c[i+1,:]-c[i,:]) + (d_t2/d_t1)*(c[i,:]-c[i-1,:])]
        which simplifies on a uniform grid to: d_T[i,:] = (c[i+1,:]-c[i-1,:])/(2*d_t)
        for i = 1, ..., len(time_grid)-1
        
        d_KK[:,i] = 2.0*(c[:,i-1]/(d_k1*(d_k1+d_k2)) - (c[:,i]/(d_k1*d_k2)) + (c[:,i+1]/(d_k2*(d_k1+d_k2))))  
        which simplifies on a uniform grid to: d_KK[:,i] = (c[:,i-1]-2*c[:,i]+c[:,i+1])/(d_k**2)
        for i = 1, ..., len(strikes)-1

        The formula can be interpreted as an infinitesimal calendar / butterfly.

        The square root yields the local volatility.

        Args:
            call_param (np.ndarray): array of call prices (2D) of the form (n_expiries, n_strikes)
            x_strikes (np.ndarray): timegrid of x_strikes (1D)
            time_grid (np.ndarray): timegrid of expiries (1D)

        Returns:
            local variance as a 2D grid of expiries and x_strikes 
        """
        
        d_T = np.zeros((len(time_grid),len(x_strikes)))
        deltas_t = np.diff(time_grid) #time_grid[1:]-time_grid[:-1]
        if all(deltas_t-deltas_t[0]) < 1E-15: #uniform grid
            d_T[1:-1,:]  = 1/(2*deltas_t[0]) * (call_param[2:,:] - call_param[:-2,:])
        else: 
            d_t1 = time_grid[1:-1] - time_grid[:-2] #time_grid[i] - time_grid[i-1]
            d_t2 = time_grid[2:] - time_grid[1:-1] #time_grid[i+1] - time_grid[i] 
            d_T[1:-1,:] = np.multiply(
                            np.multiply((call_param[2:,:]-call_param[1:-1,:]).T, (d_t1/d_t2)) + 
                            np.multiply((call_param[1:-1,:]-call_param[:-2,:]).T, (d_t2/d_t1)), 
                            1./(d_t1+d_t2)).T
            
        d_KK = np.zeros((len(time_grid),len(x_strikes)))
        deltas_k = np.diff(x_strikes)
        if all(deltas_k-deltas_k[0]) < 1E-15: #uniform grid
            d_KK[:,1:-1]  = (1/(deltas_k[0]**2))*(call_param[:,:-2]-2*call_param[:,1:-1]+call_param[:,2:])
        else:        
            
            d_k1 = x_strikes[1:-1] - x_strikes[:-2] # x_strikes[i] - x_strikes[i-1]
            d_k2 = x_strikes[2:] - x_strikes[1:-1] # x_strikes[i+1] - x_strikes[i] 
            d_KK[:,1:-1] = 2.0 * (np.multiply((call_param[:,:-2]), 1/(d_k1*(d_k1+d_k2))) - 
                            np.multiply((call_param[:,1:-1]), 1/(d_k1*d_k2)) +
                            np.multiply((call_param[:,2:]), 1/(d_k2*(d_k1+d_k2))))
        
        # remove extreme cases (numerical inconsistencies)
        d_KK = np.maximum(d_KK, 1E-8)
        
        var = d_T / (1/2*(x_strikes**2)*d_KK)
                
        # boundary cases 
        var[0,:] = var[1,:]
        var[-1,:] = var[-2,:]
        var[:,0] = var[:,1]
        var[:,-1] = var[:,-2]

        # corner cases
        var[0,0] = var[1,1]
        var[-1,-1] = var[-2,-2]
        var[0,-1] = var[1,-2]
        var[-1,0] = var[-2,1]

        # remove extreme cases (numerical inconsistencies)
        var[var>2.5] = 2.5

        return var

    @staticmethod
    def compute_local_var(vol_param, x_strikes: np.array, time_grid: np.array, call_param: np.ndarray=None, min_lv = 0.01, max_lv = 1.5):
        """Calculate the local variance from vol_param or call_param for x_strikes on time_grid

        Args:
            vol_param: a grid or a parametrisation of the volatility
            x_strikes (np.array): strikes
            time_grid (np.array): time_grid
            call_param (np.ndarray, optional): A grid of call prices. Not compatible with vol_param. Defaults to None.

        Returns:
            local volatility surface on the grid
        """

        if (vol_param is None) and (call_param is None):
            raise Exception('Set vol_params or call_params!')

        if (vol_param is not None) and (call_param is not None):
            raise Exception('Set either vol_params or call_params, not both!')
        
        if vol_param is not None:
            local_var = LocalVol._compute_local_var_from_vol(vol_param, x_strikes, time_grid)
        elif call_param is not None:
            local_var = LocalVol._compute_local_var_from_call(call_param, x_strikes, time_grid)
        return np.minimum(np.maximum(min_lv*min_lv, local_var), max_lv*max_lv)

    def apply_mc_step(self, x: np.ndarray, t0: float, t1: float, rnd: np.ndarray, inplace: bool = True):
        """Apply a MC-Euler step for the LV Model for n different paths.

        Args:
            x (np.ndarray): 2-d array containing the start values for the spot and variance. The first column contains the spot, the second the variance values.
            t0 ([type]): [description]
            t1 ([type]): [description]
            rnd ([type]): [description]
        """
        if not inplace:
            x_ = x.copy()
        else:
            x_ = x
        S = x_[:,0]
        lv = _interpolate_2D(self._time_grid, self._x_strikes, self._local_variance, S, t0) #self._variance(t0, S).reshape((-1,))
        dt = t1-t0
        sqrt_dt = np.sqrt(dt)
        S *= np.exp(- 0.5*lv*dt + np.sqrt(lv)*rnd[:,0]*sqrt_dt)
        return x_
 