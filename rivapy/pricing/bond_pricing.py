from datetime import datetime
from typing import List, Tuple, Union as _Union
from scipy.optimize import brentq
from rivapy.tools.interfaces import BaseDatedCurve
from rivapy.instruments.specifications import HasExpectedCashflows
from rivapy.marketdata import DiscountCurveParametrized, ConstantRate
from rivapy.pricing.pricing_request import PricingRequest
from rivapy.pricing._logger import logger
from rivapy.marketdata.curves import DiscountCurve
from rivapy.tools.datetools import Period, _date_to_datetime, _term_to_period, _string_to_calendar, DayCounter, Schedule, roll_day
from typing import Tuple, Union as _Union, List as _List


class SimpleCashflowPricer:
    """
    SimpleCashflowPricer - A static class for pricing cashflow-based instruments
    """

    def expected_cashflows(self) -> _List[Tuple[datetime, float]]:
        """Get the expected cashflows of the instrument.

        Returns:
            List[Tuple[datetime, float]]: The expected cashflows of the instrument.
        """
        return SimpleCashflowPricer.get_expected_cashflows(self, self._val_date)

    @staticmethod
    def get_expected_cashflows(
        specification: HasExpectedCashflows, val_date: _Union[datetime.date, datetime, None] = None, curve: _Union[DiscountCurve, None] = None
    ) -> List[Tuple[datetime, float]]:
        schedule = specification.get_schedule()
        dates = schedule._roll_out(
            from_=specification._start_date,
            to_=specification._end_date,
            term=_term_to_period(specification._tenor),
            ref_date=val_date,  # restrict schedule to the current and future periods
        )
        dcc = DayCounter(specification.day_count_convention)
        if specification._coupon_type == "float":
            if val_date is None:
                raise ValueError("val_date must be provided for floating rate cashflow calculation.")
            cashflows = []
            for d1, d2 in zip(dates[:-1], dates[1:]):
                payment_date = roll_day(d2, specification._calendar, specification._business_day_convention, settle_days=specification._payment_days)
                # For the first period, check if we have a fixing rate or if d1 is before curve date
                if len(cashflows) == 0 and (specification.last_fixing is not None or (curve is not None and d1 >= curve.refdate)):
                    rate = specification.last_fixing if specification.last_fixing is not None else curve.rivapy_valueFWD(val_date, d1, d2)
                else:
                    # For other periods use forward rate from curve
                    rate = curve.rivapy_valueFWD(val_date, d1, d2) if curve is not None else specification._coupon

                amount = specification._notional * rate * dcc.yf(d1, d2)
                cashflows.append((payment_date, amount))
        else:
            cashflows = [
                (
                    roll_day(d2, specification._calendar, specification._business_day_convention, settle_days=specification._payment_days),
                    specification._notional * specification._coupon * dcc.yf(d1, d2),
                )
                for d1, d2 in zip(dates[:-1], dates[1:])
            ]
        if specification._notional_exchange:
            cashflows.append(
                (
                    specification._start_date,
                    specification._notional * (-1),
                )
            )
            cashflows.append(
                (
                    roll_day(
                        specification._maturity_date,
                        specification._calendar,
                        specification._business_day_convention,
                        settle_days=specification._payment_days,
                    ),
                    specification._notional,
                )
            )
        return cashflows

    def pv_cashflows(self) -> float:
        """Get the present value of the cashflows.

        Returns:
            float: The present value of the cashflows.
        """
        return SimpleCashflowPricer.get_pv_cashflows(self._val_date, self._spec, self._discount_curve)

    @staticmethod
    def get_pv_cashflows(
        val_date: datetime,
        specification: HasExpectedCashflows,
        discount_curve: DiscountCurve,
        cashflows: _Union[List[Tuple[datetime, float]], None] = None,
    ) -> float:
        # logger.info('Start computing pv cashflows for bond ' + specification.obj_id)

        if cashflows is None:
            cashflows = SimpleCashflowPricer.get_expected_cashflows(specification, val_date=val_date)  # get only cashflows

        pv_cashflows = 0.0
        for c in cashflows:
            if c[0] > val_date:
                df = discount_curve.rivapy_value(val_date, c[0])
                logger.debug("Cashflow " + str(c[1]) + ", date: " + str(c[0]) + ", df: " + str(df))
                pv_cashflows += df * c[1]
        # logger.info('Finished computing pv cashflows for bond ' + specification.obj_id + ', pv_cashflows: '+ str(pv_cashflows) )
        return pv_cashflows

    def compute_yield(self, target_dirty_price: float) -> float:
        """Compute the yield of the bond.

        Args:
            target_dirty_price (float): The target dirty price.

        Returns:
            float: The computed yield.
        """
        return SimpleCashflowPricer.get_compute_yield(target_dirty_price, self._val_date, self._spec)

    # TODO: add accrued interest
    @staticmethod
    def get_compute_yield(target_dirty_price: float, val_date: datetime, specification: HasExpectedCashflows) -> float:
        # logger.info('Start computing bond yield for bond ' + specification.obj_id + ', dirty price: ' + str(target_dirty_price))
        def target_function(r: float) -> float:
            dc = DiscountCurveParametrized("", val_date, ConstantRate(r))
            price = SimpleCashflowPricer.pv_cashflows(val_date, specification, dc)
            logger.debug("Target function called with r: " + str(r) + ", price: " + str(price) + ", target_dirty_price: " + str(target_dirty_price))
            return price - target_dirty_price

        result = brentq(target_function, -0.2, 1.5, full_output=False)
        # logger.info('Finished computing bond yield')
        return result
