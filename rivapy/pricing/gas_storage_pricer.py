import numpy as np
import datetime as dt
from typing import Callable, Union

from rivapy.models.ornstein_uhlenbeck import OrnsteinUhlenbeck
from rivapy.models.gbm import GeometricBrownianMotion
from rivapy.instruments.gasstorage_specification import GasStorageSpecification

class _PolynomialRegressionFunction:
    def __init__(self, deg: int):
        self.deg = deg

    def fit(self, S: np.array, C: np.array):
        return np.polyfit(S, C, self.deg)

    def predict(self, R: np.array, S: np.array):
        return np.polyval(R, S)

class PricingParameter:
    def __init__(self, n_time_steps:int, n_actions: int, n_vol_levels: int, regression: object = _PolynomialRegressionFunction):
        """The pricing parameters needed for the gas storage valuation.

        Args:
            n_time_steps (int): number of time steps
            n_actions (int): number of possible actions
            n_vol_levels (int): number of discretized volume levels
            regression (object, optional): regression for continuous value. Defaults to _PolynomialRegressionFunction.
        """
        self.n_time_steps = n_time_steps    
        self.n_actions = n_actions         
        self.n_vol_levels = n_vol_levels
        self.regression = regression

def _payoff_func(S, delta_v, a1=0, a2=0, b1=0, b2=0, action=1):
    #payoff function h (with bid-ask spread & transaction cost)
    # -c*delta_v     inject         c=(1+a1)S +b1
    # 0              do nothing
    # -p*delta_v     withdraw       p=(1-a2)S -b2
    
    #if a1=a2=b1=b2=0 -> c=p=S
    if a1==0 and a2==0 and b1==0 and b2==0:
        return -S*delta_v
    else:
        if action == 1: #inject
            return -((1+a1)*S + b1) * delta_v
        elif action == -1: #withdraw
            return -((1-a2)*S - b2) * delta_v
        else: #do nothing
            return 0

def _penalty_func(S, v): #final condition
    def indicator_function(v, y):
        return np.array([0 if x == y else 1 for x in v])
    # v(T)=0 should be satisfied, else: penalty: S(T)*100_000
    return -100_000 * S * indicator_function(v, 0) 

def pricing_lsmc(storage: GasStorageSpecification, 
                pricing_parameters: PricingParameter,
                prices: np.ndarray, 
                nb_sims: int,
                penalty_func: Callable = _penalty_func) -> Union[float, np.ndarray]:
    """ Least-Squares Monte Carlo Method for Pricing the Gas Storage

    Args:
        storage (GasStorageSpecification): the specification 
        pricing_parameters (PricingParameter): the parameters
        prices (np.ndarray): the prices
        nb_sims (int): number of simulations
        penalty_func (Callable): the penalty function for the end. Defaults to _penalty_func.

    Returns:
        float: the optimal price
        np.ndarray: the optimal volume levels
    """

    #discretization of possible volume levels
    v = np.linspace(storage.min_level, storage.storage_capacity, pricing_parameters.n_vol_levels) 
    
    # Assign a value to the contract at maturity according to the final condition
    acc_cashflows = np.empty((len(prices), len(v), nb_sims)) # acc_cashflows[t, i,j]: expected accumulated cashflow (of optimal policy) for j-th path at t arriving at t at vollevel i 

    # For t=T+1: final condition
    for i in range(nb_sims):    
        acc_cashflows[-1,:,i] = penalty_func(prices[-1,i], v) #size v
    
    # Apply backward induction for t=T...1 
    # For each t, step over N allowed volume levels v(t,n) 
    regression = pricing_parameters.regression(deg=1)
    total_vol_levels = np.empty((len(prices), len(v), nb_sims), dtype=int) # total_vol_levels[t,i,j] gives the index of the next optimal volume level in t+1 starting from the i-th volume for the j-th path
    total_vol_levels[-1,0,0] = 0 # TODO 
    for t in range(len(prices)-2,-1,-1):
        for vol_t in range(len(v)):
            dec_func = np.empty((len(v), nb_sims))
            for vol_tplus1 in range(len(v)): #for all volumes by themselves
                # - Run an OLS regression to find an approx. of the continuation value
                if t > 0:
                    cont_val = acc_cashflows[t+1,vol_tplus1,:] # assumed cont. value for t+1, no disc. factor
                    cv_fit = regression.fit(prices[t,:], cont_val)
                    cv_pred = regression.predict(cv_fit, prices[t,:])
                else:
                    cont_val = acc_cashflows[t+1,vol_tplus1,:] # assumed cont. value for t+1, no disc. factor
                    cv_pred = np.full((nb_sims,), cont_val.mean())
                # - Combine the cont. values C into a decision rule for each volume level
                max_withdraw = max(storage.withdrawal_rate, -(v[vol_t]-storage.min_level))
                max_inject = min(storage.injection_rate, storage.storage_capacity - v[vol_t])
                
                dv = v[vol_tplus1] - v[vol_t]

                # check against constraints to find achievable actions:
                if dv >= max_withdraw and dv <= max_inject:
                    dec_func[vol_tplus1,:] = _payoff_func(prices[t,:],dv) + cv_pred
                else:
                    dec_func[vol_tplus1, :] = -1e12

            #argmax for decision rule for specified volume level at time step t+1
            ind = np.argmax(dec_func, axis=0)
            #dv_max = v[vol_t]*np.ones(nb_sims) - v[ind] #size nb_sims
            total_vol_levels[t, vol_t,:] = ind

            # - Calculate the accumulated future cash flows Y^b
            acc_cashflows[t,vol_t,:] = np.max(dec_func, axis=0) #no disc. factor
    
    #forward sweep for optimal path
    ind_level = np.empty((len(prices), nb_sims), dtype=int)
    total_volume = np.empty((len(prices), nb_sims))
    ind_level[0,:] = np.argmin(np.abs(v-storage.start_level)) # index of volume level that is closest to the start level
    total_volume[0,:] = v[ind_level[0,:]]
    for t in range(0,len(prices)-1):
        for m in range(nb_sims):
            ind_level[t+1,m] = total_vol_levels[t,ind_level[t,m],m]
            total_volume[t+1,m] = v[ind_level[t+1,m]]

    #price = acc_cashflows[0,0,0]
    return acc_cashflows, total_volume

if __name__=='__main__':

    def create_contract_dates(startdate: dt.datetime, enddate: dt.datetime, datestep:dt.timedelta)->list:
        dates=[startdate]
        while dates[-1] <= enddate-datestep:
            dates.append(dates[-1]+datestep)
        return dates

    ## Setting the parameters
    num_sims = 1000 #number of independent price paths simulated
    S0 = 1.0 #starting value
    sigma = 0.01
    mu = 0.2

    n_vol_levels = 11 #101
    min_level = 0.0
    storage_capacity = 10.0 #1_000
    start_level = 0.0 #100_000
    end_level = 0.0 #100_000
    max_withdrawal = -1.0 #-7500
    max_injection = 1.0 #2500

    startdate = dt.datetime.fromisoformat('2021-01-01')
    enddate = dt.datetime.fromisoformat('2021-12-31')
    dateStep = dt.timedelta(days=1) #daily nomination
    contractdates = create_contract_dates(startdate, enddate, dateStep)
    timegrid = np.array([(contractdates[i] - contractdates[0]).days/365.0 for i in range(len(contractdates))])
    rnd = np.random.normal(size=(timegrid.shape[0], num_sims))

    # Simulate M independent price paths S^b(1), S^b(T+1) for b = 1...M starting at S(0)
    gbm_sim = GeometricBrownianMotion(mu, sigma)
    gbm = gbm_sim.simulate(timegrid, S0, rnd) 
    ou_sim = OrnsteinUhlenbeck(speed_of_mean_reversion=0.1, volatility=sigma, mean_reversion_level=1.0)
    ou = ou_sim.simulate(timegrid=timegrid, start_value=S0, rnd=rnd)

    path = ou
    params = PricingParameter(n_time_steps = 0, n_actions = 0, n_vol_levels = n_vol_levels)#, regression = _PolynomialRegressionFunction)

    store = GasStorageSpecification(storage_capacity=storage_capacity, withdrawal_rate=max_withdrawal, injection_rate=max_injection, end_level=end_level, min_level=min_level, start_level=start_level)  


    cashflow, vol_levels = pricing_lsmc(store, params, path, num_sims)#, _penalty_func)
    
    price = cashflow[0,0,0]
    dispatch = np.mean(vol_levels, axis=1)    # 'mean timetable' of facility

    print(price)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(dispatch)
    plt.show()