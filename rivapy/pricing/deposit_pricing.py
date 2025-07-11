from datetime import datetime, date
from scipy.optimize import brentq
from rivapy.tools.interfaces import BaseDatedCurve, HasExpectedCashflows
from rivapy.marketdata import DiscountCurveParametrized, ConstantRate, DiscountCurve
from rivapy.pricing.pricing_request import PricingRequest
from rivapy.pricing._logger import logger
from rivapy.instruments.deposit_specifications import DepositSpecification
from typing import List as _List, Union as _Union, Tuple
from rivapy.tools.datetools import DayCounter


class DepositPricer:

    def __init__(
        self,
        val_date: _Union[date, datetime],
        deposit_spec: DepositSpecification,
        discount_curve: DiscountCurve,
        spread_curve: _Union[DiscountCurve, float] = 1.0,
    ):
        """_summary_

        Args:
            val_date (_Union[date, datetime]): _description_
            deposit_spec (DepositSpecification): _description_
            discount_curve (DiscountCurve): _description_
            spread_curve (_Union[DiscountCurve, float]): _description_
        """

        self._val_date = val_date
        self._deposit_spec = deposit_spec
        self._discount_curve = discount_curve
        self._spread_curve = spread_curve

    def impliedSimplyCompoundedRate(self):
        """Returns the fair rate such that the specification gives the contract a zero value.
        Assumption is that it is a simply compounded rate

        i.e. D(t) = 1 / ( 1+ rate(t) * t)

        Returns:
            float_: _description_
        """

        df = self._discount_curve.rivapy_valueFWD(self._val_date, self._deposit_spec.start_date, self._deposit_spec.maturity_date)

        if isinstance(self._spread_curve, DiscountCurve):
            spread_df = self._discount_curve.rivapy_valueFWD(self._val_date, self._deposit_spec.start_date, self._deposit_spec.maturity_date)
        else:
            spread_df = self._spread_curve

        # obtain time interval
        dcc = DayCounter(self._discount_curve.daycounter)  # use the curves or the specification? TODO: they should be the same though...
        dt = dcc.yf(self._deposit_spec.start_date, self._deposit_spec.maturity_date)

        return (1.0 / (spread_df * df) - 1.0) / dt

    def price(self):

        dcc = DayCounter(self._discount_curve.daycounter)  # use the curves or the specification? TODO: they should be the same though...
        dt = dcc.yf(self._deposit_spec.start_date, self._deposit_spec.maturity_date)

        value_d1 = self._deposit_spec.notional * (1 + self._deposit_spec.rate * dt) / (1 + self._deposit_spec.rate * dt)

        value_d1 = self._deposit_spec.notional  # as deposits are par-rate instrument?????
        df_val_d1 = self._discount_curve.rivapy_value(self._val_date, self._deposit_spec.start_date)
        PV = value_d1 * df_val_d1  # in the case that val_date is less that start date ... discount it from start datet to val date ...,

        return PV
