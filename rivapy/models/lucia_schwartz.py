from typing import Union, Callable, Tuple, List
import numpy as np
import scipy
import scipy.integrate
from rivapy.tools.interfaces import FactoryObject
from rivapy.models.ornstein_uhlenbeck import OrnsteinUhlenbeck

class LuciaSchwartz(FactoryObject):

    def _eval_grid(f, timegrid):
        if f is None:
            return np.zeros(timegrid.shape)
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result

    def __init__(self, 
                rho: float,
                kappa: Union[float, Callable]=None, 
                sigma1: Union[float, Callable]=None, 
                mu: Union[float, Callable]=None,
                sigma2:Union[float, Callable]=None,
                
                f: Callable[[Union[float, np.ndarray]],Union[float, np.ndarray]]=None):
        """Lucia Schwartz two factor model.

        The model may be used to simulate spot/forward prices via

        .. math:: 

            S(t) = f(t) + X_1(t) + X_2(t)

            dX_1(t) = -\\kappa X_1(t)+\sigma_1dW_1(t)

            dX_2(t) = \\mu dt + \sigma_2 dW_2
            
        where :math:`f(t)` is a deterministic function, :math:`\\kappa` the speed of mean reversion for 
        the first process that may be interpreted as the long-term factor and :math:`\\sigma_1` the respective volatility.
        The second factor :math:`X_2` may be interpreted as a short-term factor that is influenced by  :math:`W_2`
        and has drift :math:`\\mu`. :math:`X_1` and :math:`X_2` may be correlated with correlation :math:`\\rho`. Note that this class just simulates 

        
        Args:
            kappa (Union[float, Callable]): The speed of mean reversion for the first factor :math:`X_1`. Can be either constant or time dependent.
            sigma1 (Union[float, Callable]): The volatility of the first factor :math:`X_1`. Can be either constant or time dependent.
            mu (Union[float, Callable]): The drift of teh second factor :math:`X_2`. Can be either constant or time dependent.
            sigma2 (Union[float, Callable]): The volatility of the second factor :math:`X_2`. Can be either constant or time dependent.
            rho (float): Correlation between X1 and X2.
            f (Union[float, Callable], optional): Deterministic function of time. Defaults to 0.
        """
        self.X1 = OrnsteinUhlenbeck(kappa, sigma1, 0.0)
        self.mu = mu
        self.sigma2 = sigma2
        self.rho = rho
        self._timegrid = None
        self.f = f

    def _to_dict(self) -> dict:
        return {'kappa': self.X1.speed_of_mean_reversion, 'sigma1': self.X1.volatility,
                'mu': self.mu, 'sigma2': self.sigma2, 'f': self.f}

    def _set_timegrid(self, timegrid: np.ndarray):
        """
        Sets the timegrid for simulation.

        Args:
            timegrid (np.ndarray): Timegrid for simulation.
        """
        self._mu_grid = LuciaSchwartz._eval_grid(self.mu, timegrid)
        self._sigma2_grid = LuciaSchwartz._eval_grid(self.sigma2, timegrid)
        self._f_grid = LuciaSchwartz._eval_grid(self.f, timegrid)

    def rnd_shape(self, n_sims: int, n_timepoints: int)->tuple:
        return (n_timepoints-1, n_sims, 2)


    def compute_expected_value(self, x0: np.ndarray, T: Union[float, np.ndarray])->Union[float, np.ndarray]:
        """
        Computes the expected value of the model.

        Args:
            x0 (Union[float, np.ndarray]): Start value, either onedimensional (containing just the start value for X0 or X1) or twodimensional (different start values for X0 and X1 from e.g. a MC simulation).
            T (Union[float, np.ndarray]): Terminal time.

        Returns:
            Union[float, np.ndarray]: Expected value.
        """
        if callable(self.mu):
            raise NotImplementedError("Only implemented for fixed value of mu.")
        if callable(self.sigma2):
            raise NotImplementedError("Only implemented for fixed value of sigma2.")
        if len(x0.shape)==1:
            return self.X1.compute_expected_value(x0[0], T) + x0[1]+self.mu*T
        return self.X1.compute_expected_value(x0[:,0], T) + x0[:,1]+self.mu*T

    def compute_fwd_value(self, x0: np.ndarray, T1:float, T2:float, 
                           qm: Callable[[Callable,float],float]=None,
                           **qm_kwargs)->float:
        """Compute the forward value for a forward (swap) with continuos delivery between two time points.
        
        In more detail, the forward value is computed as

        .. math::

            F(t,t_1,t_2) = \\frac{E_t[\\int_{T_1}^{T_2} F(t,s) ds]}{T_2-T_1}

        where :math:`F(t,s)` is the expected spot price at time :math:`T`. The expectation is
        taken under the risk neutral measure :math:`Q^M`.

        Args:
            T1 (float): Start point of period.
            T2 (float): End point of period. If None, the expected value of the spot price :math:`F(t,T_1)` at T1 is returned.
            qm (Callable[[Callable,float,float]], optional): Quadrature method used for the integral. If None, scipy.integrate.romberg will be used. Defaults to None.
            **qm_kwargs: Keyword arguments for the quadrature method.

        Returns:
            float: Forward value.
        """
        if T2 is None:
            return self.compute_expected_value(x0,T1)
        if T2<=T1:
            raise ValueError("T2 must be larger than T1.")
        if qm is None:
            qm = scipy.integrate.quad
        result, err = qm(lambda t: self.compute_expected_value(x0,t), T1, T2, **qm_kwargs)
        return result/(T2-T1)

    def simulate(self, timegrid, start_value, rnd, 
                 forwards:List[Tuple[float,float]]=None,
                 forward_start_values:np.ndarray=None,):
        """
        Simulates the model.

        Args:
            timegrid (np.ndarray): Timegrid for simulation.
            start_value (Union[float, np.ndarray]): Start value for simulation.
            rnd (np.ndarray): Random numbers for simulation.
            forwards (List[Tuple[float,float]], optional): Forwards for simulation. Defaults to None.
            forward_start_values (np.ndarray, optional): Initial values for forwards. 
                Defaults to None. If forwards is specified and this argument is None, 
                the initial values will be computed with the method compute_fwd_value.
        """
        n_assets = 1
        if forwards is not None:
            n_assets = len(forwards)+1
        self._set_timegrid(timegrid)
        rnd_ = np.copy(rnd)
        rnd_[:,:,1] = self.rho*rnd[:,:,0] + np.sqrt(1.0-self.rho**2)*rnd[:,:,1]
        X2 = np.empty((timegrid.shape[0],rnd.shape[1],))
        if len(start_value.shape)==2:
            start_X1 = start_value[:,0]
            X2[0,:] = start_value[:,1]
        else:
            start_X1 = start_value[0]
            X2[0,:] = start_value[1]
        X1 = self.X1.simulate(timegrid, start_value=start_X1, rnd=rnd_[:,:,0])
        tmp = self._mu_grid[:-1]*self.X1._delta_t
        tmp2 =  self._sigma2_grid[:-1]*self.X1._sqrt_delta_t
        X2[1:,:] = tmp[:,np.newaxis] + tmp2[:,np.newaxis]*rnd_[:,:,1]
        X2 = X2.cumsum(axis=0)
        #for i in range(timegrid.shape[0]-1):
        #    X2[i+1,:] = X2[i,:] + self._mu_grid[i]*self.X1._delta_t[i] + self._sigma2_grid[i]*rnd[i,:,1]*self.X1._sqrt_delta_t[i]
        if forwards is not None:
            if forward_start_values is None:
                forward_start_values = np.empty((len(forwards),))
                for i in range(len(forwards)):
                    forward_start_values[i] = self.compute_fwd_value(start_value, forwards[i][0], forwards[i][1])
        
            result = np.full((timegrid.shape[0], rnd.shape[1], n_assets), np.nan)
            result[:,:,0] = X1 + X2  + self._f_grid[:,np.newaxis]
            dW1 = self.X1._volatility_grid[:1,np.newaxis]*rnd_[:,:,0]*self.X1._sqrt_delta_t[:,np.newaxis]
            dW2 = self._sigma2_grid[:1,np.newaxis]*rnd_[:,:,0]*self.X1._sqrt_delta_t[:,np.newaxis] 
            for fwd in range(len(forwards)):
                tt_T2 = forwards[fwd][1]-timegrid
                tt_T1 = forwards[fwd][0]-timegrid
                kappa = self.X1._speed_of_mean_reversion_grid
                tmp = -(np.exp(-kappa*tt_T2) - np.exp(-kappa*tt_T1))/(kappa*(forwards[fwd][1]-forwards[fwd][0]))           
                result[0,:,fwd+1] = forward_start_values[fwd]
                result[1:,:,fwd+1] = tmp[:1,np.newaxis]*dW1 + dW2
                result[:,:,fwd+1] = result[:,:,fwd+1].cumsum(axis=0)
        else:
            return X1 + X2  + self._f_grid[:,np.newaxis]
        return result

if __name__=='__main__':
    ls_model = LuciaSchwartz(rho=-.81, kappa = 0.077, sigma1 = 0.1, mu=-0.29, sigma2=0.1)
    n_sims = 10_000
    timegrid = np.linspace(0.0,1.0,365)
    rnd = np.random.normal(size=ls_model.rnd_shape(n_sims, timegrid.shape[0]))
    paths = ls_model.simulate(timegrid, start_value=np.array([0.0,0.0]), rnd=rnd, forwards=[(1.0,2.0)])