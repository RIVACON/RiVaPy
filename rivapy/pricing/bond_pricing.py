from datetime import datetime, date
from typing import List, Tuple, Union as _Union
from scipy.optimize import brentq
from rivapy.tools.enums import InterestRateIndex
from rivapy.tools.interfaces import BaseDatedCurve
from rivapy.instruments.bond_specifications import DeterministicCashflowBondSpecification
from rivapy.marketdata.curves import DiscountCurveComposition
from rivapy.marketdata import DiscountCurveParametrized, ConstantRate
from rivapy.pricing.pricing_request import PricingRequest
from rivapy.pricing._logger import logger
from rivapy.marketdata.curves import DiscountCurve
from rivapy.tools.datetools import Period, _date_to_datetime, _term_to_period, _string_to_calendar, DayCounter, Schedule, roll_day, calc_start_day
from typing import Tuple, Union as _Union, List as _List


class DeterministicCashflowPricer:
    """
    DeterministicCashflowPricer - A static class for pricing cashflow-based instruments
    """

    def __init__(
        self,
        val_date: datetime,
        spec: DeterministicCashflowBondSpecification,
        discount_curve: DiscountCurve,
        fwd_curve: _Union[DiscountCurve, None] = None,
    ):
        """
        Initialize the DeterministicCashflowPricer.

        Args:
            val_date (datetime): The valuation date.
            spec (DeterministicCashflowBondSpecification): The specification of the cashflow instrument.
            discount_curve (DiscountCurve): The discount curve to use for pricing.
            fwd_curve (Union[DiscountCurve, None], optional): The forward curve to use for pricing. Defaults to None.
        """
        self._val_date = _date_to_datetime(val_date)
        self._spec = spec
        self._discount_curve = discount_curve
        self._fwd_curve = fwd_curve

    def expected_cashflows(self) -> _List[Tuple[datetime, float]]:
        """Get the expected cashflows of the instrument.

        Returns:
            List[Tuple[datetime, float]]: The expected cashflows of the instrument.
        """
        return DeterministicCashflowPricer.get_expected_cashflows(self, self._val_date, self._fwd_curve)

    @staticmethod
    def get_expected_cashflows(
        spec: DeterministicCashflowBondSpecification,
        val_date: _Union[datetime.date, datetime, None] = None,
        fwd_curve: _Union[DiscountCurve, None] = None,
    ) -> List[Tuple[datetime, float]]:
        """
        Calculate the expected cashflows for a deterministic cashflow instrument.

        Args:
            spec (DeterministicCashflowBondSpecification): The instrument specification containing schedule, notional, coupon type, etc.
            val_date (datetime.date or datetime, optional): The valuation date, required for floating rate calculation. Defaults to None.
            curve (DiscountCurve, optional): The forward curve used for floating rate calculation. Defaults to None.

        Returns:
            List[Tuple[datetime, float]]: A sorted list of tuples, each containing the payment date and cashflow amount.
        """
        cashflows = []
        if spec._coupon_type != "zero":
            schedule = spec.get_schedule()
            # schedule for accrual periods rolled out
            dates = schedule._roll_out(from_=spec._start_date, to_=spec._end_date, term=_term_to_period(spec._frequency))
            dcc = DayCounter(spec.day_count_convention)
            for d1, d2 in zip(dates[:-1], dates[1:]):
                payment_date = roll_day(d2, spec._calendar, spec._business_day_convention, settle_days=spec._payment_days)
                if spec._coupon_type == "float":
                    if val_date is None or fwd_curve is None:
                        raise ValueError("val_date and fwd_curve must be provided for floating rate cashflow calculation.")
                    rate = DeterministicCashflowPricer.get_float_rate(spec, val_date, d1, d2, fwd_curve)
                else:
                    rate = spec._coupon
                amount = spec._notional * rate * dcc.yf(d1, d2)
                cashflows.append((payment_date, amount))
        if spec._notional_exchange:
            # add notional exchange at start and end date
            not_init = spec._issue_price if spec._issue_price is not None else spec._notional
            cashflows.append((spec._start_date, not_init * (-1)))
            cashflows.append(
                (roll_day(spec._maturity_date, spec._calendar, spec._business_day_convention, settle_days=spec._payment_days), spec._notional)
            )
        cashflows = sorted(cashflows)
        return cashflows

    # ToDo: consider rate/index period shorter than coupon period
    @staticmethod
    def get_float_rate(
        specification: DeterministicCashflowBondSpecification,
        val_date: _Union[datetime.date, datetime, None] = None,
        d1: _Union[datetime.date, datetime, None] = None,
        d2: _Union[datetime.date, datetime, None] = None,
        curve: _Union[DiscountCurve, None] = None,
    ) -> float:
        """
        Get the floating rate for a given period.

        Args:
            specification (DeterministicCashflowBondSpecification): The bond or instrument specification containing index, margin, calendar, and conventions.
            val_date (datetime.date or datetime, optional): The valuation date as of which forward rates are calculated. Defaults to None.
            d1 (datetime.date or datetime, optional): The start date of the interest period. Defaults to None.
            d2 (datetime.date or datetime, optional): The end date of the interest period. Defaults to None.
            curve (DiscountCurve, optional): The forward curve used for forward rate calculation. Defaults to None.
        Returns:
            float: The floating rate for the given period, including margin.
        """
        if specification._index is not None:  # For the first period, check if we have a fixing rate or if d1 is before curve date
            spot_days = InterestRateIndex(specification._index).value.spot_days
        else:
            spot_days = specification._spot_days
        fixing_date = calc_start_day(
            d1,
            f"{spot_days}D",
            business_day_convention=specification._business_day_convention,
            calendar=specification._calendar,
        )
        if fixing_date <= curve.refdate:
            try:
                rate = specification._fixings.get_fixing(specification._index, fixing_date)
            except Exception as e:
                logger.warning(f"No fixing found for {specification._index} on {fixing_date}. Using 0.0 as fixed rate. Error: {e}")
                rate = 0.0
        else:
            # For other periods use forward rate from curve
            rate = curve.value_fwd(val_date, d1, d2) if curve is not None else 0.0
        rate += specification._margin / 10000.0  # add margin
        return rate

    @staticmethod
    def get_accrued_interest(
        specification: DeterministicCashflowBondSpecification,
        trade_date: _Union[date, datetime, None] = None,
    ) -> float:
        """
        Get the accrued interest for a given instrument specification.

        Args:
            specification (DeterministicCashflowBondSpecification): The bond specification.
            val_date (datetime.date or datetime, optional): The valuation date. Defaults to None.

        Returns:
            float: The accrued interest.
        """
        if trade_date is None:
            raise ValueError("trade_date must be provided.")
        if specification._coupon_type == "zero":
            return 0.0
        else:
            schedule = specification.get_schedule()
            # schedule for payment periods rolled out
            dates = schedule._roll_out(from_=specification._start_date, to_=specification._end_date, term=_term_to_period(specification._frequency))
            dates = sorted(dates)
            # find the last coupon date before or on trade_date
            last_coupon_date = None
            next_coupon_date = None
            for d in dates:
                if d <= trade_date:
                    last_coupon_date = d
                elif d > trade_date and next_coupon_date is None:
                    next_coupon_date = d
                    break
            if last_coupon_date is None or next_coupon_date is None:
                return 0.0  # No accrued interest if trade_date is before first coupon or after last coupon

            dcc = DayCounter(specification.day_count_convention)
            # Calculate the fraction of the coupon period that has accrued
            accrual_fraction = dcc.yf(last_coupon_date, trade_date) / dcc.yf(last_coupon_date, next_coupon_date)
            # Calculate the accrued interest
            if specification._coupon_type == "float":
                rate = DeterministicCashflowPricer.get_float_rate(specification, trade_date, last_coupon_date, next_coupon_date)
            else:
                rate = specification._coupon
            accrued_interest = specification._notional * rate * accrual_fraction * dcc.yf(last_coupon_date, next_coupon_date)
            return accrued_interest

        return accrued_interest

    def pv_cashflows(self) -> float:
        """Get the present value of the cashflows.

        Returns:
            float: The present value of the cashflows.
        """
        return DeterministicCashflowPricer.get_pv_cashflows(self._val_date, self._spec, self._discount_curve, self._fwd_curve)

    @staticmethod
    def get_pv_cashflows(
        val_date: datetime,
        specification: DeterministicCashflowBondSpecification,
        discount_curve: DiscountCurve,
        fwd_curve: _Union[DiscountCurve, None] = None,
        cashflows: _Union[List[Tuple[datetime, float]], None] = None,
    ) -> float:
        # logger.info('Start computing pv cashflows for bond ' + specification.obj_id)

        if cashflows is None:
            cashflows = DeterministicCashflowPricer.get_expected_cashflows(
                specification, val_date=val_date, fwd_curve=fwd_curve
            )  # get only cashflows

        pv_cashflows = 0.0
        for c in cashflows:
            if c[0] > val_date:
                df = discount_curve.value(val_date, c[0])
                logger.debug("Cashflow " + str(c[1]) + ", date: " + str(c[0]) + ", df: " + str(df))
                pv_cashflows += df * c[1]
        # logger.info('Finished computing pv cashflows for bond ' + specification.obj_id + ', pv_cashflows: '+ str(pv_cashflows) )
        return pv_cashflows

    @staticmethod
    def get_dirty_price(
        val_date: datetime,
        specification: DeterministicCashflowBondSpecification,
        discount_curve: DiscountCurve,
        fwd_curve: _Union[DiscountCurve, None] = None,
    ) -> float:
        logger.info("Start computing dirty price for bond " + specification.obj_id)
        pv_cashflows = DeterministicCashflowPricer.get_pv_cashflows(val_date, specification, discount_curve, fwd_curve)
        logger.info("Finished computing dirty price for bond " + specification.obj_id + ", dirty_price: " + str(pv_cashflows))
        return pv_cashflows

    def dirty_price(self) -> float:
        """Get the dirty price of the bond.

        Returns:
            float: The dirty price of the bond.
        """
        return DeterministicCashflowPricer.get_dirty_price(self._val_date, self._spec, self._discount_curve, self._fwd_curve)

    @staticmethod
    def clean_price(
        val_date: datetime,
        specification: DeterministicCashflowBondSpecification,
        discount_curve: DiscountCurve,
        fwd_curve: _Union[DiscountCurve, None] = None,
    ) -> float:
        dirty_price = DeterministicCashflowPricer.get_dirty_price(val_date, specification, discount_curve, fwd_curve)
        accrued_interest = DeterministicCashflowPricer.get_accrued_interest(specification, val_date)
        return dirty_price - accrued_interest

    def clean_price(self) -> float:
        """Get the clean price of the bond.

        Returns:
            float: The clean price of the bond.
        """
        return self.dirty_price() - self.accrued_interest()

    def compute_yield(self, target_dirty_price: float) -> float:
        """Compute the yield of the bond.

        Args:
            target_dirty_price (float): The target dirty price.

        Returns:
            float: The computed yield.
        """
        return DeterministicCashflowPricer.get_compute_yield(target_dirty_price, self._val_date, self._spec)

    # TODO: add accrued interest
    @staticmethod
    def get_compute_yield(
        target_dirty_price: float, val_date: datetime, specification: DeterministicCashflowBondSpecification, discount_curve: DiscountCurve
    ) -> float:
        logger.info("Start computing bond z-spread for bond " + specification.obj_id + ", dirty price: " + str(target_dirty_price))

        def target_function(r: float) -> float:
            dc = DiscountCurveParametrized(discount_curve, 1.0, ConstantRate(r))
            price = DeterministicCashflowPricer.pv_cashflows(val_date, specification, dc)
            logger.debug("Target function called with r: " + str(r) + ", price: " + str(price) + ", target_dirty_price: " + str(target_dirty_price))
            return price - target_dirty_price

        result = brentq(target_function, -0.2, 1.5, full_output=False)
        logger.info("Finished computing bond z-spread")
        return result

    @staticmethod
    def get_z_spread(target_dirty_price: float, val_date: datetime, specification: DeterministicCashflowBondSpecification) -> float:
        # logger.info('Start computing bond yield for bond ' + specification.obj_id + ', dirty price: ' + str(target_dirty_price))
        def target_function(r: float) -> float:
            dc = DiscountCurveComposition()
            price = DeterministicCashflowPricer.pv_cashflows(val_date, specification, dc)
            logger.debug("Target function called with r: " + str(r) + ", price: " + str(price) + ", target_dirty_price: " + str(target_dirty_price))
            return price - target_dirty_price

        result = brentq(target_function, -0.2, 1.5, full_output=False)
        # logger.info('Finished computing bond yield')
        return result

    def z_spread(self, target_dirty_price: float) -> float:
        """Compute the z-spread of the bond.

        Args:
            target_dirty_price (float): The target dirty price.
        Returns:
            float: The computed z-spread.
        """
        return DeterministicCashflowPricer.get_z_spread(target_dirty_price, self._val_date, self._spec, self._discount_curve)
