from datetime import datetime, date
from typing import List, Tuple, Union as _Union, Optional as _Optional
from scipy.optimize import brentq
from rivapy.tools.enums import DayCounterType, InterestRateIndex
from rivapy.tools.interfaces import BaseDatedCurve
from rivapy.instruments.bond_specifications import DeterministicCashflowBondSpecification, FloatingRateBondSpecification
from rivapy.marketdata.curves import DiscountCurveComposition
from rivapy.marketdata import DiscountCurveParametrized, ConstantRate
from rivapy.pricing.pricing_request import PricingRequest
from rivapy.instruments._logger import logger
from rivapy.marketdata.curves import DiscountCurve
from rivapy.tools.datetools import (
    Period,
    _date_to_datetime,
    _term_to_period,
    _string_to_calendar,
    DayCounter,
    Schedule,
    roll_day,
    calc_start_day,
    _period_to_string,
)
from typing import Tuple, Union as _Union, List as _List


class DeterministicCashflowPricer:
    """Deterministic cashflow pricer utilities.

    This class provides static and instance helpers for pricing deterministic cashflow
    instruments (fixed- and floating-rate bonds, deposits, zero-coupon instruments).
    It contains methods to generate expected cashflows, discount them, compute PVs,
    yields, z-spreads and duration measures.
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
        self._cashflows = None

    @property
    def cashflows(self) -> _List[Tuple[datetime, float]]:
        """Get the cashflows of the instrument.

        Returns:
            Cashflow: The cashflows of the instrument.
        """
        if self._cashflows is None:
            self._cashflows = DeterministicCashflowPricer.get_expected_cashflows(self._spec, self._val_date, self._fwd_curve)
        return self._cashflows

    @cashflows.setter
    def cashflows(self, value: _List[Tuple[datetime, float]]):
        """Set the cashflows of the instrument.

        Returns:
            Cashflow: The cashflows of the instrument.
        """
        self._cashflows = value

    def expected_cashflows(self) -> _List[Tuple[datetime, float]]:
        """Get the expected cashflows of the instrument.

        Returns:
            List[Tuple[datetime, float]]: The expected cashflows of the instrument.
        """
        return DeterministicCashflowPricer.get_expected_cashflows(self._spec, self._val_date, self._fwd_curve)

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
            # schedule = spec.get_schedule()
            # # schedule for accrual periods rolled out
            # dates = schedule._roll_out(
            #     from_=spec._start_date if not spec._backwards else spec._end_date,
            #     to_=spec._end_date if not spec._backwards else spec._start_date,
            #     term=_term_to_period(spec._frequency),
            #     long_stub=spec.stub_type_is_Long,
            #     backwards=spec.backwards,
            # )
            dates = spec.dates
            # print(spec._notional.get_amortization_schedule())
            dcc = DayCounter(spec.day_count_convention)
            for d1, d2 in zip(dates[:-1], dates[1:]):

                if spec.coupon_type == "float":
                    if val_date is None or fwd_curve is None:
                        raise ValueError("val_date and fwd_curve must be provided for floating rate cashflow calculation.")
                    rate = DeterministicCashflowPricer.get_float_rate(spec, val_date, d1, d2, fwd_curve)
                else:
                    rate = spec.coupon
                # normalize day count convention to canonical string before comparison
                if spec._adjust_accruals is False:
                    d1_adj = d1
                    d2_adj = d2
                else:
                    d1_adj = roll_day(d1, spec.calendar, spec.business_day_convention)
                    d2_adj = roll_day(d2, spec.calendar, spec.business_day_convention)
                if DayCounterType.to_string(spec.day_count_convention) == DayCounterType.ActActICMA.value:
                    nr = spec.get_nr_annual_payments()
                    if nr == 0:
                        raise ValueError("Number of annual payments is zero. Please check the frequency setting in the bond specification.")
                    dcv = dcc.yf(d1_adj, d2_adj, dates, nr)
                else:
                    dcv = dcc.yf(d1_adj, d2_adj)
                amount = spec._notional.get_amount_per_date(d1_adj) * rate * dcv
                payment_date = roll_day(d2_adj, spec.calendar, spec.business_day_convention, settle_days=spec.payment_days)
                cashflows.append((payment_date, amount))
        # add notional amortizations to cashflow list
        cashflows.extend(spec._notional.get_amortization_schedule())
        # add notional exchanges at start and end date if applicable (and only remaining notionals after amortizations)
        if spec._notional_exchange:
            # add notional exchange at start and end date
            not_init = spec._issue_price if spec._issue_price is not None else spec._notional.get_amount(0)
            cashflows.append((spec._start_date, not_init * (-1)))
            rem_amount = spec._notional.get_amount(0) - spec._amortization_scheme.get_total_amortization()
            if rem_amount > 0:
                cashflows.append(
                    (
                        roll_day(spec._maturity_date, spec._calendar, spec._business_day_convention, settle_days=spec._payment_days),
                        rem_amount,
                    )
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
        if specification._ir_index is not None:  # For the first period, check if we have a fixing rate or if d1 is before curve date
            spot_days = InterestRateIndex(specification._ir_index).value.spot_days
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
                print(specification._ir_index)
                fix_name = (
                    InterestRateIndex(specification._ir_index).value.name
                    if specification._ir_index is not None and isinstance(InterestRateIndex(specification._ir_index), InterestRateIndex)
                    else _period_to_string(specification._frequency)
                )
                rate = specification._fixings.get_fixing(fix_name, fixing_date)
            except Exception as e:
                logger.warning(f"No fixing found for {specification._index} on {fixing_date}. Using 0.0 as fixed rate. Error: {e}")
                rate = 0.0
        else:
            # For other periods use forward rate from curve
            rate = curve.value_fwd_rate(val_date, d1, d2) if curve is not None else 0.0
        rate += specification._margin / 10000.0  # add margin
        return rate

    @staticmethod
    def get_accrued_interest(
        specification: DeterministicCashflowBondSpecification,
        trade_date: _Union[date, datetime, None] = None,
        fwd_curve: _Union[DiscountCurve, None] = None,
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
            # schedule = specification.get_schedule()
            # # schedule for payment periods rolled out
            # dates = schedule._roll_out(from_=specification._start_date, to_=specification._end_date, term=_term_to_period(specification._frequency))
            dates = specification.accrual_dates
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
            if isinstance(specification, FloatingRateBondSpecification):
                accrual_fraction = dcc.yf(last_coupon_date, trade_date, specification.dates, specification.get_nr_annual_payments()) / dcc.yf(
                    last_coupon_date, next_coupon_date, specification.dates, specification.get_nr_annual_payments()
                )
                yf = dcc.yf(last_coupon_date, next_coupon_date, specification.dates, specification.get_nr_annual_payments())
            else:
                accrual_fraction = dcc.yf(last_coupon_date, trade_date) / dcc.yf(last_coupon_date, next_coupon_date)
                yf = dcc.yf(last_coupon_date, next_coupon_date)
            # Calculate the accrued interest
            if specification._coupon_type == "float":
                rate = DeterministicCashflowPricer.get_float_rate(specification, trade_date, last_coupon_date, next_coupon_date, fwd_curve)
            else:
                rate = specification._coupon
            accrued_interest = specification._notional.get_amount_per_date(trade_date) * rate * accrual_fraction * yf
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
        """Discount and sum cashflows to obtain present value.

        Args:
            val_date (date | datetime): Valuation date used for discounting.
            specification (DeterministicCashflowBondSpecification): Instrument specification.
            discount_curve (DiscountCurve): Curve used to obtain discount factors.
            fwd_curve (DiscountCurve, optional): Forward curve used for floating-rate cashflows.
            cashflows (List[(date, amount)], optional): Precomputed cashflows; if not
                provided they will be generated from the specification.

        Returns:
            float: Present value obtained by discounting future cashflows occurring after val_date.
        """

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
        """Compute the dirty price (present value including accrued interest) of a bond.

        Args:
            val_date (date | datetime): Valuation date.
            specification (DeterministicCashflowBondSpecification): Instrument specification.
            discount_curve (DiscountCurve): Curve used for discounting.
            fwd_curve (DiscountCurve, optional): Forward curve for floating-rate cashflows.

        Returns:
            float: Dirty price (PV of future cashflows after val_date).
        """
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
        """Compute the clean price of a bond (dirty price minus accrued interest).

        Args:
            val_date (date | datetime): Valuation date.
            specification (DeterministicCashflowBondSpecification): Instrument specification.
            discount_curve (DiscountCurve): Discount curve used for discounting.
            fwd_curve (DiscountCurve, optional): Forward curve for floating-rate cashflows.

        Returns:
            float: Clean price (dirty price less accrued interest).
        """
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
        return DeterministicCashflowPricer.get_compute_yield(target_dirty_price, self._val_date, self._spec, cashflows=self.cashflows)

    # TODO: add accrued interest
    @staticmethod
    def get_compute_yield(
        target_dirty_price: float,
        val_date: datetime,
        specification: DeterministicCashflowBondSpecification,
        cashflows: _Union[List[Tuple[datetime, float]], None] = None,
        fwd_curve: _Union[DiscountCurve, None] = None,
    ) -> float:
        logger.info("Start computing bond z-spread for bond " + specification.obj_id + ", dirty price: " + str(target_dirty_price))
        if cashflows is None:
            cashflows = DeterministicCashflowPricer.get_expected_cashflows(specification, val_date, fwd_curve=fwd_curve)

        def target_function(r: float) -> float:
            dc = ConstantRate(r)
            price = DeterministicCashflowPricer.get_pv_cashflows(val_date, specification, dc, cashflows=cashflows)
            logger.debug("Target function called with r: " + str(r) + ", price: " + str(price) + ", target_dirty_price: " + str(target_dirty_price))
            return price - target_dirty_price

        result = brentq(target_function, -0.2, 1.5, full_output=False)
        logger.info("Finished computing bond z-spread")
        return result

    @staticmethod
    def get_z_spread(
        target_dirty_price: float,
        val_date: datetime,
        specification: DeterministicCashflowBondSpecification,
        discount_curve: DiscountCurve,
        cashflows: _Union[List[Tuple[datetime, float]], None] = None,
        fwd_curve: _Union[DiscountCurve, None] = None,
    ) -> float:
        logger.info("Start computing z-spread for bond " + specification.obj_id + ", dirty price: " + str(target_dirty_price))
        if cashflows is None and fwd_curve is not None:
            cashflows = DeterministicCashflowPricer.get_expected_cashflows(specification, val_date, fwd_curve=fwd_curve)
        else:
            logger.error("To compute z-spread with floating rate bonds, cashflows, or fwd_curve must be provided to calculate cashflows.")

        def target_function(r: float) -> float:
            dc = DiscountCurveComposition(discount_curve, 1.0, r)
            price = DeterministicCashflowPricer.get_pv_cashflows(val_date, specification, dc, cashflows=cashflows)
            logger.debug("Target function called with r: " + str(r) + ", price: " + str(price) + ", target_dirty_price: " + str(target_dirty_price))
            return price - target_dirty_price

        result = brentq(target_function, -0.2, 1.5, full_output=False)
        logger.info("Finished computing z-spread.")
        return result

    def z_spread(self, target_dirty_price: float) -> float:
        """Compute the z-spread of the bond.

        Args:
            target_dirty_price (float): The target dirty price.
        Returns:
            float: The computed z-spread.
        """
        return DeterministicCashflowPricer.get_z_spread(target_dirty_price, self._val_date, self._spec, self._discount_curve, self._cashflows)

    ############################# PRICER ONLY METHODS BELOW #####################################

    def macaulay_duration(self) -> float:
        """Compute the Macaulay duration for the instrument.

        Macaulay duration is the weighted average time until cashflows are received,
        weighted by the present value of the cashflows.

        Returns:
            float: Macaulay duration expressed in the same time units used by day count.
        """
        logger.info("Start computing macaulay duration for bond " + self._spec.obj_id)
        cashflows = self.expected_cashflows()
        pv_cashflows = DeterministicCashflowPricer.get_pv_cashflows(
            self._val_date, self._spec, self._discount_curve, self._fwd_curve, cashflows=cashflows
        )
        # print(pv_cashflows)
        macaulay_duration = 0.0
        dcc = DayCounter(self._spec.day_count_convention)  # , self._spec._get_coupon_frequency if self._spec._coupon_type != "zero" else None)
        for c in cashflows:
            if c[0] > self._val_date:
                df = self._discount_curve.value(self._val_date, c[0])
                t = dcc.yf(self._val_date, c[0], self._spec.dates, self._spec.nr_annual_payments)
                macaulay_duration += t * df * c[1]
        if pv_cashflows > 0:
            macaulay_duration /= pv_cashflows
        logger.info("Finished computing macaulay duration for bond " + self._spec.obj_id + ", macaulay_duration: " + str(macaulay_duration))
        return macaulay_duration

    def modified_duration(
        self,
        target_dirty_price: float = 100.0,
    ) -> float:
        """Compute the modified duration for the instrument.

        Modified duration approximates the percentage price change for a unit change
        in yield and is computed from the Macaulay duration and yield.

        Args:
            target_dirty_price (float, optional): Dirty price used for yield inversion.
                Defaults to 100.0.

        Returns:
            float: Modified duration.
        """
        logger.info("Start computing modified duration for bond " + self._spec.obj_id)
        macaulay_duration = self.macaulay_duration()
        yld = self.compute_yield(target_dirty_price)
        modified_duration = macaulay_duration / (1 + yld)
        logger.info("Finished computing modified duration for bond " + self._spec.obj_id + ", modified_duration: " + str(modified_duration))
        return modified_duration
