import numpy as np
import datetime as dt


def spot_price_model(
    timestamp: dt.datetime,
    spot_price_level: float,
    peak_price_level: float,
    solar_price_level: float,
    weekend_price_level: float,
    winter_price_level: float,
    epsilon_mean: float = 0,
    epsilon_var: float = 1,
    seed: int = 42,
) -> float:
    """Dummy power spot price model.

    .. math::
        S(t) = S_0 +
        \\begin{cases}
            0, & 0 \leq h(t) < 8 \n
            P_p, & 8 \leq h(t) < 11 \n
            -P_{pv}, & 11 \leq h(t) < 16 \n
            P_p, & 16 \leq h(t) \leq 20 \n
            0, & 20 < h(t) \leq 23
        \\end{cases}
        +
        \\begin{cases}
            0, & 1\leq d(t) \leq 5 \n
            -P_{we}, & 6\leq d(t) \leq 7
        \\end{cases}
        +
        \\begin{cases}
            0, & m(t) \in \\{4,5,6,7,8,9\\} \n
            P_{W}, & m(t) \in \\{1,2,3,10,11,12\\}
        \\end{cases}
        + \\varepsilon

    .. math::
        \\begin{aligned}
            S_0 &\quad \\text{Spot price level} \n
            P_p &\quad \\text{Peak price level} \n
            P_{pv} &\quad \\text{Price level with regard to solar power} \n
            P_{we} &\quad \\text{Price level for weekends} \n
            P_W &\quad \\text{Price level for winter} \n
            h(t) &\quad \\text{Hour of the time step } t \n
            d(t) &\quad \\text{Weekday of the time step } t \n
            m(t) &\quad \\text{Month of the time step } t \n
            \\varepsilon &\sim \\mathcal{N}(\\mu, \\sigma^2)
        \\end{aligned}

    Args:
        timestamp (dt.datetime): Time stamp
        spot_price_level (float): Spot price level
        peak_price_level (float): Peak price level
        solar_price_level (float): Price level with regard to solar power
        weekend_price_level (float): Price level for weekends
        winter_price_level (float): Price level for winter
        epsilon_mean (float, optional): Additional additive noise mean. Defaults to 0.
        epsilon_var (float, optional): Additional additive noise standard deviation. Defaults to 1.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        float: spot price

    Example:

    .. highlight:: python
    .. code-block:: python

            parameter_dict = {
                'spot_price_level': 100,
                'peak_price_level': 10,
                'solar_price_level': 8,
                'weekend_price_level': 10,
                'winter_price_level': 20,
                'epsilon_mean': 0,
                'epsilon_var': 5
            }
            date_range = pd.date_range(start='1/1/2023', end='1/1/2025', freq='h', inclusive='left')
            spot_prices = list(map(lambda x: spot_price_model(x, **parameter_dict), date_range))
    """
    if seed is not None:
        np.random.seed(seed)
    spot_price = spot_price_level
    if (timestamp.hour >= 8 and timestamp.hour < 11) or (timestamp.hour >= 16 and timestamp.hour <= 20):
        spot_price += peak_price_level
    elif timestamp.hour >= 11 and timestamp.hour < 16:
        spot_price -= solar_price_level

    if timestamp.weekday() >= 5:
        spot_price -= weekend_price_level

    if timestamp.month in {1, 2, 3, 10, 11, 12}:
        spot_price += winter_price_level

    spot_price += np.random.normal(loc=epsilon_mean, scale=np.sqrt(epsilon_var))
    return spot_price
