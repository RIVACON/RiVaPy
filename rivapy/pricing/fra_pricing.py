from datetime import datetime, date
from scipy.optimize import brentq
from rivapy.tools.interfaces import BaseDatedCurve
from rivapy.instruments.specifications import HasExpectedCashflows
from rivapy.marketdata import DiscountCurveParametrized, ConstantRate, DiscountCurve
from rivapy.pricing.bond_pricing import SimpleCashflowPricer
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
        """_summary_

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

    # ToDo:
    # -- add: expected cashflows
    # -- add: @static method price and ensure that non-static method can be called
    @staticmethod
    def expected_FRA_cashflows(
        specification: ForwardRateAgreementSpecification,
        val_date: _Union[datetime.date, datetime],
        curve: DiscountCurve,
        fwdcurve: _Union[DiscountCurve, None] = None,
    ) -> _List[Tuple[datetime, float]]:
        """Calculate expected cashflows for the FRA specification based on the valuation date and discount curve.

        Args:
            specification (ForwardRateAgreementSpecification): The FRA specification.
            val_date (_Union[datetime.date, datetime]): The data as of which the cashflows are calculated.
            curve (DiscountCurve): The discount curve.
            fwdcurve (_Union[DiscountCurve, None]): The forward curve.

        Returns:
            List[Tuple[datetime, float]]: List of tuples containing payment dates and amounts.
        """

        if fwdcurve is None:
            fwdcurve = curve

        dcc = DayCounter(specification.day_count_convention)
        cashflows = []
        payment_date = roll_day(
            specification._start_date, specification._calendar, specification._business_day_convention, settle_days=specification._settlement_days
        )
        fwdrate = fwdcurve.rivapy_valueFWD(val_date, specification._rate_start_date, specification._rate_end_date)
        dscrate = curve.rivapy_valueFWD(val_date, specification._start_date, specification._end_date)
        dt = dcc.yf(specification._start_date, specification._end_date, specification._end_date)
        amount = specification._notional * (fwdrate - specification._rate) * dt
        cf = amount / (1 + dscrate * dt)
        cashflows.append((payment_date, cf))

        return cashflows

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
        expected_cashflows = ForwardRateAgreementPricer.expected_FRA_cashflows(specification, val_date, discount_curve, forward_curve)
        price = SimpleCashflowPricer(val_date, specification, discount_curve, expected_cashflows)
        return price

    def price(self):
        """Calculate the present value of the specified FRA given a discount curve and forward curve

        Returns:
           float: present value of a deposit based on simple compounding
        """
        # dcc = DayCounter(self._fra_spec.day_count_convention)

        # #        roll convention??
        # time_delta = dcc.yf(self._fra_spec.start_date, self._fra_spec.end_date)  # as yearfrac
        # fwd_rate = ForwardRateAgreementPricer.compute_fair_rate(
        #     self._val_date, self._forward_curve, self._fra_spec._rate_start_date, self._fra_spec._rate_end_date
        # )  # 1.00  # self._fra_spec. # need forward curve
        # fra_rate = self._fra_spec._rate
        # pay_off = self._fra_spec.notional * time_delta * (fwd_rate - fra_rate)
        # pay_off_discounted_to_start_date = pay_off / (1 + fwd_rate * time_delta)
        # df_val_date = self._discount_curve.rivapy_value(
        #     self._val_date, self._fra_spec.start_date
        # )  # discounted from payment datet, usually start-date to val_date
        # PV = pay_off_discounted_to_start_date * df_val_date
        price = get_price(self._val_date, self._fra_spec, self._discount_curve, self._forward_curve)

        return price

    # {
    # 	double fwdRate = ForwardRateAgreementPricer::computeFairRate(valDate, spec, forwardCurve);
    # 	double yf = spec->getDc()->yf(spec->getStartDate(), spec->getEndDate());
    # 	double pv = (fwdRate - spec->getRate()) / (1. + yf * fwdRate) * spec->getNotional() * yf *
    # 		discountCurve->value(valDate, spec->getStartDate());
    # 	if (!spec->isBuyer()) #TODO determine if is buyer or seller in specification
    # 		pv = -pv;
    # 	return pv;
    # }

    @staticmethod
    def compute_fair_rate(
        val_date: _Union[datetime, date],
        specification: ForwardRateAgreementSpecification,
        forward_curve: DiscountCurve,
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

        rate_start_date: specification.rate_start_date
        rate_end_date: specification.rate_end_date

        dcc = DayCounter(forward_curve.daycounter)
        yf = dcc.yf(rate_start_date, rate_end_date)
        fwd_df = forward_curve.rivapy_valueFWD(val_date, rate_start_date, rate_end_date)  # REF DATE is =

        fair_rate = (1.0 / fwd_df - 1) / yf

        return fair_rate


# fair rate:
# 	double yf = spec->getDc()->yf(spec->getRateStartDate(), spec->getRateEndDate());
# 			return  (1.0 / forwardCurve->valueFwd(refDate, spec->getRateStartDate(), spec->getRateEndDate()) - 1.) / yf;
