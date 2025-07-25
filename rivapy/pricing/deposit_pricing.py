from datetime import datetime, date
from scipy.optimize import brentq
from rivapy.tools.interfaces import BaseDatedCurve
from rivapy.instruments.specifications import HasExpectedCashflows
from rivapy.marketdata import DiscountCurveParametrized, ConstantRate, DiscountCurve
from rivapy.pricing.pricing_request import PricingRequest
from rivapy.pricing._logger import logger
from rivapy.instruments.deposit_specifications import DepositSpecification
from typing import List as _List, Union as _Union, Tuple
from rivapy.tools.datetools import DayCounter
from rivapy.tools._validators import _check_start_at_or_before_end
from rivapy.pricing.bond_pricing import SimpleCashflowPricer

from rivapy.tools.enums import DayCounterType


class DepositPricer:

    def __init__(
        self,
        val_date: _Union[date, datetime],
        deposit_spec: DepositSpecification,
        discount_curve: DiscountCurve,
        spread_curve: _Union[DiscountCurve, float] = 0.0,
    ):
        """_summary_

        Args:
            val_date (_Union[date, datetime]): specific date for which the value of the financial instrument is calculated.
            deposit_spec (DepositSpecification): Specification object with deposit specific parameters.
            discount_curve (DiscountCurve): Discount curve used for discounting.
            spread_curve (_Union[DiscountCurve, float]): Spread curve
        """

        self._val_date = val_date
        self._spec = deposit_spec
        self._discount_curve = discount_curve
        self._spread_curve = spread_curve
        self._validate_pricer_dates()

    def _validate_pricer_dates(self):
        """Validates consistency of valuation date, curve reference date, and deposit fixing date"""
        self._spec._fixing_date, self._val_date = _check_start_at_or_before_end(self._spec._fixing_date, self._val_date)
        self._discount_curve.refdate, self._val_date = _check_start_at_or_before_end(self._discount_curve.refdate, self._val_date)

        # def expected_cashflows(self) -> _List[Tuple[datetime, float]]:
        #     """Returns the expected cashflows of the deposit specification

        #     Returns:
        #         _List[Tuple[datetime, float]]: _description_
        #     """
        #     start_date = self._deposit_spec.start_date
        #     end_date = self._deposit_spec.maturity_date
        #     notional = self._deposit_spec.notional
        #     rate = self._deposit_spec.rate

        # if isinstance(self._spread_curve, DiscountCurve):
        #     spread_df = self._spread_curve.rivapy_valueFWD(self._val_date, start_date, end_date)
        # else:
        #     spread_df = self._spread_curve

        # dcc = DayCounter(self._discount_curve.daycounter)
        # dt = dcc.yf(start_date, end_date)
        # value_d1 = notional * (1 + rate * dt)

        # return [(end_date, value_d1)]

    def price(self):
        """Calculate the present value of the specified deposit given a discount curve and daycount convention

        Returns:
           float: present value of a deposit based on simple compounding
        """
        dc = self._discount_curve
        val_date = self._val_date
        spec = self._spec

        return SimpleCashflowPricer.pv_cashflows(val_date, spec, dc)

    # def impliedSimplyCompoundedRate(self):
    #     """Returns the fair rate such that the specification gives the contract a zero value.
    #     Assumption is that it is a simply compounded rate

    #     i.e. D(t) = 1 / ( 1+ rate(t) * t)

    #     Returns:
    #         float_: _description_
    #     """

    #     dc = self._discount_curve
    #     # self._discount_curve.rivapy_valueFWD(self._val_date, self._deposit_spec.start_date, self._deposit_spec.maturity_date)
    #     val_date = self._val_date
    #     start_date = self._spec.start_date
    #     maturity_date = self._spec.maturity_date

    #     if isinstance(dc, DiscountCurve):
    #         spread_df = dc.rivapy_valueFWD(val_date, start_date, maturity_date)
    #     else:
    #         raise ValueError("Discount curve must be of type DiscountCurve")

    #     # obtain time interval
    #     # dcc = DayCounter(self._deposit_spec._day_count_convention)  # use the curves or the specification? TODO: they should be the same though...
    #     # dt = dcc.yf(start_date, maturity_date)

    #     return spread_df  # (1.0 / (spread_df * df) - 1.0) / dt

    # At the moment, kept as a static method until pricing hierachy/architecture is decided upon
    @staticmethod
    def impliedSimplyCompoundedRate(
        val_date: _Union[date, datetime],
        start_date: _Union[date, datetime],
        end_date: _Union[date, datetime],
        dc: DiscountCurve,
        spread_curve: DiscountCurve = None,
    ):
        """Returns the fair rate such that the specification gives the contract a zero value.
        Assumption is that it is a simply compounded rate

        i.e. D(t) = 1 / ( 1+ rate(t) * t)

        Returns:
            float_: _description_
        """

        df = dc.rivapy_valueFWD(val_date, start_date, end_date)
        df_spread = 1.0

        if isinstance(spread_curve, DiscountCurve):
            df_spread = spread_curve.rivapy_valueFWD(val_date, start_date, end_date)
        else:
            raise ValueError("Discount curve must be of type DiscountCurve")

        dcc = DayCounter(DiscountCurve.daycounter)  # the DCC should be from the instrtument? or form the curve? should be the same if used totgether?
        dt = dcc.yf(start_date, end_date)  # should not be zero or will throw division by zero error
        rate = (1.0 / (df_spread * df) - 1.0) / dt

        return rate  # (1.0 / (spread_df * df) - 1.0) / dt
