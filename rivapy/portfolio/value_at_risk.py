"""
Defines a class for the value at risk. 
"""

import math

import numpy as np

import rivapy.portfolio.portfolio as pf

class VaRBase:

    def __init__(
        self,
        holding_period: int,
        historical_steps: int,
        confidence_level: float
    ):
        
        self.holding_period = holding_period
        self.historical_steps = historical_steps
        self.confidence_level = confidence_level

class VaRHistSim(VaRBase):

    def value_at_risk(
        self,
        timeseries: dict,
        portfolio: pf.Portfolio
    ):
        return self.__calc_var(timeseries, portfolio)
        
    def __calc_var(
        self,
        timeseries: dict,
        portfolio: pf.Portfolio
    ):
        
        # prune the market data to only keep the relevant returns needed for the VaR 
        num_steps = self.historical_steps + self.holding_period
        for k, v in timeseries.items():
            timeseries[k] = v[:num_steps]
            if len(v) < num_steps:
                raise ValueError(f"The timeseries for the product {k} does not have enough values. \
                                 {num_steps} are needed for the VaR.")

        # look up all values for all speecifications and timesteps
        portfolio_values = np.array([
            spec.value(np.array(timeseries[spec.symbol]))
            for spec in portfolio.constituents
        ])
        portfolio_values = np.sum(portfolio_values, axis=0)

        # calculate the portfolio returns and find the array index for the provided quantile
        portfolio_returns = portfolio_values[self.holding_period:] - portfolio_values[:-self.holding_period]
        portfolio_returns = np.sort(portfolio_returns)
        index = round(portfolio_returns.size * (1 - self.confidence_level))

        return portfolio_returns[index]


class VaRDeltaNormal(VaRBase):

    def __calc_var(
        self,
        timeseries: dict,
        portfolio: pf.Portfolio
    ):
        pass
    
class VaRDeltaGamma(VaRBase):

    def __calc_var(
        self,
        timeseries: dict,
        portfolio: pf.Portfolio
    ):
        pass

class Shapely:
    pass

# --------------------------------------------------------------------------------------
# todo: the following class is from another program and has to be translated into the new structure above
class VarAnalysis:
    """
    Performs a value at risk analysis for a given portfolio object.
    """

    def __init__(
            self,
            portfolio: pf.Portfolio,
            historical_timeseries: dict
    ):
        self.__hist_ts = historical_timeseries
        self.portfolio = portfolio
        self.__var_method = config.COMPANY_CONFIG["VAR"]["METHOD"]

    @staticmethod
    def __var_hist_sim(
            historical_timeseries: dict,
            holding_period: int,
            periods_back: int,
            confidence_level: float,
            portfolio: pf.Portfolio
    ) -> float:

        relevant_returns = np.array([timeseries \
                            for name, timeseries in historical_timeseries.items() \
                            if name in list(portfolio.names)
                            ]).T

        returns = relevant_returns[holding_period:] - relevant_returns[:(-holding_period)]
        returns = returns[:periods_back]

        portfolio_returns = np.sort(returns.dot(portfolio.volumes))
        index = round(portfolio_returns.size * (1 - confidence_level))

        return portfolio_returns[index]

    @staticmethod
    def __var_delta_normal(
            historical_timeseries: dict,        
            holding_period: int,
            periods_back: int,
            confidence_level: float,
            portfolio: pf.Portfolio
    ) -> float:
        # todo
        raise NotImplementedError("The delta normal method is not yet available.")

    var_types = {
        "DELTA_NORMAL": __var_delta_normal,
        "HIST_SIM": __var_hist_sim
    }

    def __calc_var(
            self,
            holding_period: int,
            periods_back: int,
            confidence_level: float,
            portfolio: pf.Portfolio,
            rounding_digits: int
    ) -> float:

        if portfolio.names.size == 0:
            return 0

        var = self.var_types[self.__var_method](
            self.__hist_ts, holding_period, periods_back, confidence_level, portfolio)
        return round(var, rounding_digits) * (-1)

    def value_at_risk(
            self,
            holding_period: int,
            periods_back: int,
            confidence_level: float,
            rounding_digits: int = 2
    ) -> float:
        """
        Computes the value at risk for the given portfolio.

        Args:
            holding_period (int): Holding period in number of periods the time series is given in.
            periods_back (int): Number of historical time periods to use for the analysis.
            confidence_level (float): Value at risk confidence level.
            rounding_digits (int, optional): Number of digits the result should be rounded to.
                Defaults to 2.

        Returns:
            float: Value at risk.
        """
        return self.__calc_var(
            holding_period,
            periods_back,
            confidence_level,
            self.portfolio,
            rounding_digits
        )

    def shapely(
        self,
        holding_period: int,
        periods_back: int,
        confidence_level: float,
        rounding_digits: int = 2) -> dict:
        """
        Calculates the shapely decomposition of the value at risk.

        Args:
            holding_period (int): Holding period in number of periods the time series is given in.
            periods_back (int): Number of historical time periods to use for the analysis.
            confidence_level (float): Value at risk confidence level.
            rounding_digits (int, optional): Number of digits the result should be rounded to.
                Defaults to 2.

        Returns:
            dict: Portfolio components and respective shapely values.
        """

        def var_function(portfolio_dict: dict):
            portfolio = pf.Portfolio(
                list(portfolio_dict.keys()),
                list(portfolio_dict.values()))

            return self.__calc_var(
                holding_period,
                periods_back,
                confidence_level,
                portfolio,
                rounding_digits
            )

        portfolio_dict = dict(zip(
            self.portfolio.names.tolist(),
            self.portfolio.volumes.tolist()))

        all_subsets = hf.powerset(portfolio_dict)
        all_subsets = [dict(subset, coalition=var_function(subset)) for subset in all_subsets]

        set_size = len(portfolio_dict)

        shapely_values = {}

        for name in portfolio_dict:

            subsets_with_id = [subset for subset in all_subsets if name in subset]
            subsets_without_id = [subset for subset in all_subsets if not name in subset]

            coalition_deltas = np.array([s["coalition"] for s in subsets_with_id]) \
                - np.array([s["coalition"] for s in subsets_without_id])

            subset_sizes = [len(s) - 1 for s in subsets_without_id]

            factors = np.array([math.factorial(sss) * math.factorial(set_size - 1 - sss) \
                                for sss in subset_sizes ]) \
                / math.factorial(set_size)

            shapely_values = dict(shapely_values, **{name: sum(coalition_deltas * factors)})

        return dict(shapely_values)

