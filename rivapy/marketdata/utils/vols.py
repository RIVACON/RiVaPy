import numpy as np
from scipy.optimize import linprog



def fit_arbfree_callprices(bid_prices:np.ndarray, ask_prices:np.ndarray, c: np.ndarray,
                           return_optimizer_result: bool = False)->np.ndarray:
    """This method is designed to estimate a call option price surface that adheres to the principle 
    of arbitrage avoidance and falls within the provided bid and ask price ranges. It operates under 
    the assumption that bid and ask prices are provided on a consistent grid uniform in both strike price 
    and expiry date dimensions, eliminating the need to explicitly define these grids. The method 
    involves solving a linear programming problem with the objective function 
    defined by the parameter 'c.' The constraints in this problem are determined 
    by the bid-ask spread and the conditions necessary to prevent calendar and butterfly arbitrage.

    Args:
        bid_prices (np.ndarray): Call bid prices given on a grid where the first coordinate 
                                is the expiry and the second the strike direction. Note that strikes and expiries must be given on a uniform grid.
        ask_prices (np.ndarray): _description_
        c (np.ndarray): This defines the cost function and it can be used to control in which parts one wants to be close to the bid or the ask price.

    Returns:
        np.ndarray: The resulting surface of call option prices.
    """
    n_expiries = bid_prices.shape[0]
    n_strikes = bid_prices.shape[1]
    
    ## define the constraints for the LP
    # bid_ask_spread: bid < C(K,T) < ask
    bounds = [(bid_prices[i,j], ask_prices[i,j]) for i in range(n_expiries) for j in range(n_strikes)]
       
    def idx(shape1, shape2):
        def get_idx(idx1, idx2):
            return (idx1[0]*shape1[1] + idx1[1], idx2[0]*shape2[1] + idx2[1])
        return get_idx

    # butterfly_arbitrage: d_K^2(C) = C(K-h,T) - 2*C(K,T) + C(K+h,T) > 0
    butterfly_arbitrage = np.zeros((n_expiries*(n_strikes-2),n_expiries*n_strikes))
    butterfly_arbitrage_rhs = np.zeros((n_expiries*(n_strikes-2),))#*1E-6
    idx_ = idx((n_expiries, n_strikes-2), (n_expiries, n_strikes))
    for expiry in range(n_expiries):
        for strike in range(n_strikes-2):
            butterfly_arbitrage[idx_((expiry, strike), (expiry, strike))] = -1.0
            butterfly_arbitrage[idx_((expiry, strike), (expiry, strike+1))] = 2.0
            butterfly_arbitrage[idx_((expiry, strike), (expiry, strike+2))] = -1.0

    # calendar_arbitrage: C(K,T) <= C(K,T+1)
    calendar_arbitrage = np.zeros(((n_expiries-1)*n_strikes,n_expiries*n_strikes))
    calendar_arbitrage_rhs = np.zeros(((n_expiries-1)*n_strikes))
    idx_ = idx((n_expiries-1, n_strikes), (n_expiries, n_strikes))
    for expiry in range(n_expiries-1):
        for strike in range(n_strikes):
            calendar_arbitrage[idx_((expiry, strike), (expiry, strike))] = 1.0 #[i*(n_strikes)+j, i*(n_strikes)+j]
            calendar_arbitrage[idx_((expiry, strike), (expiry+1, strike))] = -1.0 #[i*(n_strikes)+j, (i+1)*(n_strikes)+j]

    A_ub = np.concatenate([butterfly_arbitrage, calendar_arbitrage])
    b_ub = np.concatenate([butterfly_arbitrage_rhs, calendar_arbitrage_rhs])

    ## Linear Programming: minimize c@x such that A_ub@x <= b_ub, A_eq@x = b_eq, lb<=x<=ub 
    result = linprog(c, A_ub, b_ub, bounds=bounds, method='highs')
    if not result.success:
        print('no success')
    call_prices = np.reshape(result.x, (n_expiries, n_strikes))
    if return_optimizer_result:
        return call_prices, result
    return call_prices

