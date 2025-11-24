from datetime import datetime, date
from scipy.optimize import brentq
from rivapy.tools.interfaces import BaseDatedCurve
from rivapy.instruments.bond_specifications import DeterministicCashflowBondSpecification
from rivapy.marketdata import DiscountCurveParametrized, ConstantRate, DiscountCurve
from rivapy.pricing.bond_pricing import DeterministicCashflowPricer
from rivapy.pricing.pricing_request import PricingRequest
from rivapy.pricing._logger import logger
from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.instruments.fra_specifications import ForwardRateAgreementSpecification
from typing import List as _List, Union as _Union, Tuple
from rivapy.tools.datetools import DayCounter, roll_day


class ForwardRateAgreementPricer:

    def __init__(
        self,
        val_date: _Union[date, datetime],
        fra_spec: ForwardRateAgreementSpecification,
        discount_curve: DiscountCurve,
        forward_curve: DiscountCurve = None,
    ):
        """Initializes the FRA pricer with valuation date, FRA specification, discount curve and forward curve.

        Args:
            val_date (_Union[date, datetime]): specific date for which the value of the financial instrument is calculated.
            fra_spec (ForwardRateAgreementSpecification): Specification object with FRA specific parameters.
            discount_curve (DiscountCurve): Discount curve used for discounting.
            forward_curve(): from underlying index...

        """

        self._val_date = val_date
        self._fra_spec = fra_spec
        self._discount_curve = discount_curve

        if forward_curve == None:
            # generate forward curve from given discount curve?
            self._forward_curve = discount_curve  # TODO implement functionality
        else:
            self._forward_curve = forward_curve

    @staticmethod
    def get_expected_cashflows(
        specification: ForwardRateAgreementSpecification,
        val_date: _Union[datetime.date, datetime],
        fwdcurve: DiscountCurve,
    ) -> _List[Tuple[datetime, float]]:
        """Calculate expected cashflows for the FRA specification based on the valuation date and discount curve.

        Args:
            specification (ForwardRateAgreementSpecification): The FRA specification.
            val_date (_Union[datetime.date, datetime]): The data as of which the cashflows are calculated.
            fwdcurve (_Union[DiscountCurve, None]): The forward curve.

        Returns:
            List[Tuple[datetime, float]]: List of tuples containing payment dates and amounts.
        """

        cashflows = []
        # using curve daycount convention to get fwd-rate data
        dcc_rate = DayCounter(fwdcurve.daycounter)
        fwdrateDF = fwdcurve.value_fwd(val_date, specification._rate_start_date, specification._rate_end_date)
        dt_rate = dcc_rate.yf(specification._rate_start_date, specification._rate_end_date)
        fwdrate = (1.0 / fwdrateDF - 1) / dt_rate
        print(f"Day count fraction (yf): {dt_rate}, Forward rate: {fwdrate}")

        # using instrument daycount convention to calculate delta t for cf amount calculation and discouting
        dcc = DayCounter(specification.day_count_convention)
        dt = dcc.yf(specification._start_date, specification._end_date)
        amount = specification._notional * (fwdrate - specification._rate) * dt
        print(f"dt: {dt}, Specification_Rate: {specification._rate}, Amount: {amount}")
        cf = amount / (1 + fwdrate * dt)
        print(f"Cashflow: {cf}")

        payment_date = roll_day(
            specification._start_date, specification._calendar, specification._business_day_convention, settle_days=specification._payment_days
        )
        cashflows.append((payment_date, cf))

        return cashflows

    def expected_cashflows(self):
        """Calculate expected cashflows for the FRA specification.

        Returns:
            List[Tuple[datetime, float]]: List of tuples containing payment dates and amounts.
        """
        return ForwardRateAgreementPricer.get_expected_cashflows(self._fra_spec, self._val_date, self._forward_curve)

    @staticmethod
    def get_price(
        val_date: _Union[datetime.date, datetime],
        specification: ForwardRateAgreementSpecification,
        discount_curve: DiscountCurve,
        forward_curve: _Union[DiscountCurve, None] = None,
    ) -> float:
        """Calculate the present value of the specified FRA given a discount curve and forward curve

        Args:
            val_date (_Union[datetime.date, datetime]): The valuation date.
            specification (ForwardRateAgreementSpecification): The FRA specification.
            discount_curve (DiscountCurve): The discount curve.
            forward_curve (_Union[DiscountCurve, None]): The forward curve.

        Returns:
            float: The present value of the FRA.
        """
        expected_cashflows = ForwardRateAgreementPricer.get_expected_cashflows(specification, val_date, forward_curve)
        price = discount_curve.value(val_date, expected_cashflows[0][0]) * expected_cashflows[0][1]
        # DeterministicCashflowPricer.get_pv_cashflows(val_date, specification, discount_curve, expected_cashflows)
        return price

    def price(self):
        """Calculate the present value of the specified FRA given a discount curve and forward curve

        Returns:
           float: present value of a deposit based on simple compounding
        """
        price = ForwardRateAgreementPricer.get_price(self._val_date, self._fra_spec, self._discount_curve, self._forward_curve)

        return price

    @staticmethod
    def compute_fair_rate(
        val_date: _Union[datetime, date],
        specification: ForwardRateAgreementSpecification,
        discount_curve: DiscountCurve,
    ):
        """Computes the fair rate such that the when used in the specification of the FRA gives a net value of zero.
        A discount curve is given, from which the Forward Rate is determined between the two dates.
        Assuming simple compounding
        Forward rate = (DF_1/DF_2 -1 )/ time_interval
                     = (1 /FWD_DF -1 )/ time_interval

        Args:
            val_date (_Union[datetime, date]): specific date as of which the value of the financial instrument is calculated.
            forward_curve (DiscountCurve): Forward curve used for projecting rates
            rate_start_date (_Union[datetime, date]): start date for the forward period
            rate_end_date (_Union[datetime, date]): end date for the forward period

        Returns:
            float: _description_
        """

        rate_start_date = specification._rate_start_date
        rate_end_date = specification._rate_end_date

        dcc = DayCounter(discount_curve.daycounter)
        yf = dcc.yf(rate_start_date, rate_end_date)
        fwd_df = discount_curve.value_fwd(val_date, rate_start_date, rate_end_date)  # REF DATE is =

        fair_rate = (1.0 / fwd_df - 1) / yf

        return fair_rate
