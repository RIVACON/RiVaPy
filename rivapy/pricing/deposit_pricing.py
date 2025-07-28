import math
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


class DepositPricer(SimpleCashflowPricer):

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
        self._discount_curve.refdate, self._spec._fixing_date = _check_start_at_or_before_end(self._discount_curve.refdate, self._spec._fixing_date)
        self._discount_curve.refdate, self._val_date = _check_start_at_or_before_end(self._discount_curve.refdate, self._val_date)

    @staticmethod
    def price(val_date: datetime, specification: HasExpectedCashflows, discount_curve: BaseDatedCurve):
        """Calculate the present value of the specified deposit given a discount curve and daycount convention

        Returns:
           float: present value of a deposit based on simple compounding
        """

        return SimpleCashflowPricer.pv_cashflows(val_date, specification, discount_curve)

    @staticmethod
    def implied_simply_compounded_rate(val_date: datetime, specification: HasExpectedCashflows, discount_curve: BaseDatedCurve):
        """Returns the fair rate such that the specification gives the contract a zero value.
        Assumption is that it is a simply compounded rate

        i.e. D(t) = 1 / ( 1+ simple_rate(t) * t)

        Calculation requires to convert a continous rate for the deposit term into a simple rate:
        i. e. simple_rate = ((1 / (D(t)) - 1) /t

        Returns:
            float_: _description_
        """

        # self._discount_curve.rivapy_valueFWD(self._val_date, self._deposit_spec.start_date, self._deposit_spec.maturity_date)
        start_date = specification.start_date
        maturity_date = specification.maturity_date
        daycountconvention = specification.day_count_convention

        if isinstance(discount_curve, DiscountCurve):
            cont_df = discount_curve.rivapy_valueFWD(val_date, start_date, maturity_date)
        else:
            raise ValueError("Discount curve must be of type DiscountCurve")

        # obtain time interval
        dcc = DayCounter(daycountconvention)  # use the curves or the specification? TODO: they should be the same though...
        dt = dcc.yf(start_date, maturity_date)
        simple_rate = ((1 / cont_df) - 1) / dt

        return simple_rate
