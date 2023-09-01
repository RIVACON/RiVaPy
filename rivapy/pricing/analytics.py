import math
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.special import ndtr
import numpy as np


def compute_european_price_Buehler(strike:float, maturity:float, volatility:float, is_call: bool=True)->float:
    """Compute a call/put option price for the Buehler model (w.r.t. x-process), i.e. no dividends, rates etc.
    Args:
        strike (float): strike
        maturity (float): maturity
        volatility (float): volatility
        is_call (bool): if option is call or put
    Returns:
        float: Black-Scholes call price
    """
    if maturity < 1E-12:
	    return np.maximum(1.0 - strike, 0)
    sqrt_mat = math.sqrt(maturity)
    d1 = ( math.log( 1.0 / strike ) + 0.5*volatility*volatility*maturity ) / ( volatility *  sqrt_mat)
    d2 = d1 - volatility * sqrt_mat
    #print(d1,d2)
    if is_call:
        return  ndtr(d1) - strike * ndtr(d2)
    return -ndtr(-d1) + strike*ndtr(-d2)

def compute_implied_vol_Buehler(strike: float, maturity:float, price:float,
                                min_vol = 0.05, max_vol = 2.0, is_call=True, **kwargs)->float:
    """Computes the implied volatility for a given cal/putl price using brentq from scipy. It throws an exception if no implied vol can be determined.

    Args:
        strike (float): [description]
        maturity (float): [description]
        price (float): [description]
        min_vol (float, optional): [description]. Defaults to 0.05.
        max_vol (float, optional): [description]. Defaults to 2.0.
        is_call (bool): if option is call or put

    Returns:
        float: [description]
    """
    def error(vol:float):
        result = price - compute_european_price_Buehler(strike, maturity, vol, is_call= is_call)
        return result
    return brentq(error,a=min_vol, b=max_vol, **kwargs)