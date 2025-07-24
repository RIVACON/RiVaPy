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
            val_date (_Union[date, datetime]): _description_
            deposit_spec (DepositSpecification): _description_
            discount_curve (DiscountCurve): _description_
            spread_curve (_Union[DiscountCurve, float]): _description_
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
        # """Returns the fair rate such that the specification gives the contract a zero value.
        # Assumption is that it is a simply compounded rate

        # i.e. D(t) = 1 / ( 1+ rate(t) * t)

        # Returns:
        #     float_: _description_
        # """

        # dc = self._discount_curve
        # # self._discount_curve.rivapy_valueFWD(self._val_date, self._deposit_spec.start_date, self._deposit_spec.maturity_date)
        # val_date = self._val_date
        # start_date = self._spec.start_date
        # maturity_date = self._spec.maturity_date

        # if isinstance(dc, DiscountCurve):
        #     spread_df = dc.rivapy_valueFWD(val_date, start_date, maturity_date)
        # else:
        #     raise ValueError("Discount curve must be of type DiscountCurve")

        # # obtain time interval
        # # dcc = DayCounter(self._deposit_spec._day_count_convention)  # use the curves or the specification? TODO: they should be the same though...
        # # dt = dcc.yf(start_date, maturity_date)

        # return spread_df  # (1.0 / (spread_df * df) - 1.0) / dt
