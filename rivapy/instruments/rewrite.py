import abc
from datetime import datetime, date
from typing import List, Any, Optional, Union, Tuple, Dict
from holidays import HolidayBase
from holidays.financial import ECB
from collections import defaultdict
import rivapy.tools.interfaces as interfaces
from dateutil.relativedelta import relativedelta
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from rivapy.tools.enums import Currency, Rating, SecuritizationLevel, RollConvention
from rivapy.tools.datetools import _date_to_datetime, Schedule, Period, DayCounterType, DayCounter
from rivapy.tools._validators import _check_positivity, _check_start_before_end, _string_to_calendar
from rivapy.tools.interfaces import BaseDatedCurve
from scipy.optimize import brentq
import numpy as np


class BondBaseSpecification(interfaces.FactoryObject):
    """Abstract base class for bond specifications."""

    def __init__(
        self,
        obj_id: str,
        schedule: Schedule,
        notional: float,
        currency: Union[Currency, str],
        issue_date: Union[date, datetime],
        issuer: Optional[str] = None,
        securitization_level: Optional[Union[SecuritizationLevel, str]] = SecuritizationLevel.NONE,
        rating: Optional[Union[Rating, str]] = Rating.NONE,
    ):
        """
        Initializes the base bond specification.

        Args:
            obj_id (str): A unique identifier for the bond, e.g., ISIN.
            schedule (Schedule): The payment schedule of the bond.
            notional (float): The face value of the bond.
            currency (Union[Currency, str]): The currency of the bond.
            issue_date (Union[date, datetime]): The date the bond was issued.
            issuer (Optional[str], optional): The issuer of the bond. Defaults to None.
            securitization_level (Optional[Union[SecuritizationLevel, str]], optional): The securitization level. Defaults to SecuritizationLevel.NONE.
            rating (Optional[Union[Rating, str]], optional): The credit rating of the bond. Defaults to Rating.NONE.
        """

        self.obj_id = obj_id

        if not isinstance(schedule, Schedule):
            raise TypeError("schedule must be an instance of rivapy.tools.datetools.Schedule.")

        self._schedule = schedule
        self.notional = notional
        self.currency = currency
        self.issue_date = issue_date
        self.issuer = issuer
        self.securitization_level = securitization_level
        self.rating = rating

        _check_start_before_end(self.issue_date, self.maturity_date)

    def _to_dict(self) -> Dict:
        # TODO: further addtion to the dictionary like Schedule
        return_dict = {
            "obj_id": self.obj_id,
            "issuer": self.issuer,
            "securitization_level": self.securitization_level,
            "issue_date": self.issue_date,
            "maturity_date": self.maturity_date,
            "currency": self.currency,
            "notional": self.notional,
            "rating": self.rating,
        }
        return return_dict

    @property
    def schedule(self) -> Schedule:
        """The bond's schedule of payments."""
        return self._schedule

    @property
    def issue_date(self) -> datetime:
        """The bond's issue date as a datetime object."""
        return self._issue_date

    @issue_date.setter
    def issue_date(self, value: Union[date, datetime]):
        self._issue_date = _date_to_datetime(value)

    @property
    def maturity_date(self) -> datetime:
        """The bond's maturity date, derived from the schedule's end date."""
        return _date_to_datetime(self._schedule.end_day)

    @property
    def notional(self) -> float:
        """The bond's notional amount (face value)."""
        return self._notional

    @notional.setter
    def notional(self, value: float):
        self._notional = _check_positivity(value)

    @property
    def currency(self) -> str:
        """The bond's currency as a string."""
        return self._currency

    @currency.setter
    def currency(self, value: Union[Currency, str]):
        self._currency = Currency.to_string(value)

    @property
    def issuer(self) -> Optional[str]:
        """The bond's issuer."""
        return self._issuer

    @issuer.setter
    def issuer(self, value: Optional[str]):
        self._issuer = value

    @property
    def securitization_level(self) -> str:
        """The bond's securitization level as a string."""
        return self._securitization_level

    @securitization_level.setter
    def securitization_level(self, value: Union[SecuritizationLevel, str]):
        self._securitization_level = SecuritizationLevel.to_string(value)

    @property
    def rating(self) -> str:
        """The bond's credit rating as a string."""
        return self._rating

    @rating.setter
    def rating(self, value: Union[Rating, str]):
        self._rating = Rating.to_string(value)

    @abc.abstractmethod
    def expected_cashflows(self) -> List[Tuple[datetime, float]]:
        """
        Computes all expected cashflows of the bond.

        Returns:
            List[Tuple[datetime, float]]: A list of tuples, where each tuple contains
                                          the payment date and the cashflow amount.
        """
        pass

    @abc.abstractmethod
    def compute_dirty_price(self, discount_curve: "DiscountCurve") -> float:
        """
        Computes the dirty price of the bond.
        The dirty price is the price of a bond including any accrued interest.

        Args:
            discount_curve (DiscountCurve): The curve used to discount future cashflows.

        Returns:
            float: The calculated dirty price.
        """
        pass

    @abc.abstractmethod
    def compute_clean_price(self, discount_curve: "DiscountCurve") -> float:
        """
        Computes the clean price of the bond by discounting all future cashflows.
        The clean price is the price of a bond including any accrued interest.

        Args:
            discount_curve (DiscountCurve): The curve used to discount future cashflows.

        Returns:
            float: The calculated clean price.
        """
        pass

    @abc.abstractmethod
    def compute_yield(self, price: float, val_date: datetime) -> float:
        """
        Computes the yield-to-maturity (YTM) of the bond.

        Args:
            price (float): The dirty price of the bond.
            val_date (datetime): The valuation date.

        Returns:
            float: The computed yield-to-maturity.
        """
        pass


class FixedRateBond(BondBaseSpecification):
    """
    Represents a fixed-rate bond with regular coupon payments.
    """

    def __init__(
        self,
        obj_id: str,
        schedule: Schedule,
        notional: float,
        currency: Union[Currency, str],
        issue_date: Union[date, datetime],
        coupon_rate: float,
        issuer: Optional[str] = None,
        securitization_level: Optional[Union[SecuritizationLevel, str]] = SecuritizationLevel.NONE,
        rating: Optional[Union[Rating, str]] = Rating.NONE,
        accrual_day_counter_type: DayCounterType = DayCounterType.ActActICMA,
    ):
        """
        Initializes a fixed-rate bond.

        Args:
            obj_id (str): A unique identifier for the bond.
            schedule (Schedule): The payment schedule of the bond.
            notional (float): The face value of the bond.
            currency (Union[Currency, str]): The currency of the bond.
            issue_date (Union[date, datetime]): The date the bond was issued.
            coupon_rate (float): The annual coupon rate (e.g., 0.05 for 5%).
            issuer (Optional[str], optional): The issuer of the bond. Defaults to None.
            securitization_level (Optional[Union[SecuritizationLevel, str]], optional): The securitization level. Defaults to SecuritizationLevel.NONE.
            rating (Optional[Union[Rating, str]], optional): The credit rating of the bond. Defaults to Rating.NONE.
            accrual_day_counter_type (DayCounterType, optional): The day count convention for accrual calculations. Defaults to DayCounterType.ActActICMA.
        """
        super().__init__(obj_id, schedule, notional, currency, issue_date, issuer, securitization_level, rating)
        if coupon_rate < 0:
            raise ValueError("Coupon rate must be non-negative.")
        self.coupon_rate = coupon_rate
        self._accrual_day_counter = DayCounter(accrual_day_counter_type)
        self.accrual_day_counter_type = accrual_day_counter_type

        self.coupon_freq = self._schedule.time_period

        if self.accrual_day_counter_type == DayCounterType.ActActICMA:
            _check = False
            for cp_freq_str in ["1Y", "6M", "3M"]:
                if self.coupon_freq == Period.from_string(cp_freq_str):
                    _check = True
                    break
            if _check == False:
                raise ValueError("For the Act/Act ICMA only a coupon frequency of 1Y, 6M or 3M is supported!")

        self.cashflows = self.expected_cashflows()

    def _to_dict(self):
        return super()._to_dict()

    @staticmethod
    def _create_sample(n_samples: int, seed: int = None):
        result = []
        if seed is not None:
            np.random.seed(seed)

        issue_date = datetime(2025, 1, 1)
        maturity_date = datetime(2027, 1, 1)
        notional = 100.0
        currency = Currency.EUR
        securitization_level = SecuritizationLevel.SUBORDINATED
        daycounter = DayCounterType.ACT_ACT
        for i in range(n_samples):
            coupon_rate = np.random.choice([0.0, 0.01, 0.03, 0.05])
            period = np.random.choice(["1Y", "6M", "3M"])
            schedule = Schedule(
                start_day=issue_date, end_day=maturity_date, time_period=Period.from_string(period), business_day_convention=RollConvention.UNADJUSTED
            )
            result.append(
                {
                    "obj_id": f"ID_{i}",
                    "schedule": schedule,
                    "notional": notional,
                    "currency": currency,
                    "issue_date": issue_date,
                    "coupon_rate": coupon_rate,
                    "securitization_level": securitization_level,
                    "accrual_day_counter_type": daycounter,
                }
            )

    def _get_coupon_frequency(self):
        if self.coupon_freq.years > 0:
            coupon_frequency = 1.0 / self.coupon_freq.years
        else:
            coupon_frequency = 12.0 / self.coupon_freq.months

        return coupon_frequency

    def expected_cashflows(self) -> List[Tuple[datetime, float]]:
        """
        Computes all expected cashflows (coupons and notional) of the bond.

        Returns:
            List[Tuple[datetime, float]]: A sorted list of (date, amount) tuples for each cashflow.
        """
        # Generate all schedule dates, which are already business-day adjusted
        all_schedule_dates = self._schedule.generate_dates(ends_only=False)

        cashflows = []
        # Get frequency information needed for Act/Act ICMA calculation
        coupon_freq = self._get_coupon_frequency()

        # --- Iterate over all coupon periods ---
        for i in range(len(all_schedule_dates) - 1):
            period_start_dt = _date_to_datetime(all_schedule_dates[i])
            payment_date_dt = _date_to_datetime(all_schedule_dates[i + 1])

            # Skip cashflows that are paid out before or on the issue date
            if payment_date_dt <= self.issue_date:
                continue

            # The accrual for the coupon calculation always uses the full period from the schedule
            # to correctly handle stub periods (short/long first or last coupons).
            year_fraction_for_coupon = self._accrual_day_counter.yf(
                period_start_dt, payment_date_dt, coupon_schedule=all_schedule_dates, coupon_frequency=coupon_freq  # coupon_frequency_int
            )

            coupon_amount = self.notional * self.coupon_rate * year_fraction_for_coupon
            if coupon_amount > 0.0:
                cashflows.append((payment_date_dt, coupon_amount))

        # Add notional at maturity date (which is the last date in the schedule)
        maturity_payment_date = _date_to_datetime(all_schedule_dates[-1])
        if maturity_payment_date >= self.issue_date:
            cashflows.append((maturity_payment_date, self.notional))

        # Use a dictionary to sum amounts for cashflows on the same date (e.g., last coupon + notional)
        combined_cashflows = defaultdict(float)
        for cf_date, amount in cashflows:
            # Normalize datetime to date to ensure correct grouping if time components differ
            normalized_date = cf_date.replace(hour=0, minute=0, second=0, microsecond=0)
            combined_cashflows[normalized_date] += amount

        # Convert back to list of tuples and sort by date
        return sorted(combined_cashflows.items(), key=lambda x: x[0])

    def compute_accrued_interest(self, valuation_date: Union[date, datetime]) -> float:
        """
        Computes the accrued interest of the bond on a given valuation date.

        Args:
            valuation_date (Union[date, datetime]): The date for which to calculate the accrued interest.

        Returns:
            float: The amount of accrued interest. Returns 0.0 if the valuation date is
                   outside the bond's life (before issue or on/after maturity).
        """
        val_date_dt = _date_to_datetime(valuation_date)

        # For zero-coupon bonds, accrued interest is always zero. This also prevents division by zero errors.
        if self.coupon_rate == 0.0:
            return 0.0

        # No accrued interest if valuation is outside the bond's life
        if val_date_dt >= self.maturity_date or val_date_dt < self.issue_date:
            return 0.0

        all_coupon_schedule_dates = self._schedule.generate_dates(ends_only=False)

        current_accrual_start = None
        current_accrual_end = None

        if len(all_coupon_schedule_dates) < 2:
            return 0.0

        # Find the coupon period that contains the valuation date
        for i in range(len(all_coupon_schedule_dates) - 1):
            p_start_dt = _date_to_datetime(all_coupon_schedule_dates[i])
            p_end_dt = _date_to_datetime(all_coupon_schedule_dates[i + 1])

            if p_start_dt <= val_date_dt < p_end_dt:
                current_accrual_start = p_start_dt
                current_accrual_end = p_end_dt
                break

        if current_accrual_start is None or current_accrual_end is None:
            return 0.0

        # Get frequency information needed for Act/Act ICMA
        coupon_frequency = self._get_coupon_frequency()

        # Calculate accrued interest year fraction for the period [current_accrual_start, val_date_dt]
        accrued_year_fraction = self._accrual_day_counter.yf(current_accrual_start, val_date_dt, all_coupon_schedule_dates, coupon_frequency)
        return self.notional * self.coupon_rate * accrued_year_fraction

    def compute_clean_price(self, discount_curve: "DiscountCurve") -> float:
        """
        Computes the clean price of the bond by discounting all future cashflows.
        The clean price is the price of a bond including any accrued interest.

        Args:
            discount_curve (DiscountCurve): The curve used to discount future cashflows.

        Returns:
            float: The calculated clean price.
        """
        val_date_dt = _date_to_datetime(discount_curve.valuation_date)
        # cashflows = self.cashflows#self.expected_cashflows()
        pv_cashflows = 0.0
        for c in self.cashflows:
            if c[0] > val_date_dt:
                df = discount_curve.value(val_date_dt, c[0])
                pv_cashflows += df * c[1]
        return pv_cashflows

    def compute_dirty_price(self, discount_curve: "DiscountCurve") -> float:
        """
        Computes the dirty price of the bond.
        Dirty Price = Clean Price + Accrued Interest.

        Args:
            discount_curve (DiscountCurve): The curve used to discount future cashflows.

        Returns:
            float: The dirty price of the bond.
        """
        clean_price = self.compute_clean_price(discount_curve)
        accrued = self.compute_accrued_interest(discount_curve.valuation_date)
        return clean_price + accrued

    def compute_yield(
        self, dirty_price: float, val_date: datetime, yield_search_lower_bound: float = -0.2, yield_search_upper_bound: float = 1.5
    ) -> float:
        """
        Computes the yield-to-maturity (YTM) for a given dirty price.
        This method uses the brentq root-finding algorithm to find the yield that
        equates the present value of future cashflows to the given dirty price.

        Args:
            dirty_price (float): The dirty price of the bond.
            val_date (datetime): The valuation date.
            yield_search_lower_bound (float, optional): The lower bound for the yield search. Defaults to -0.2.
            yield_search_upper_bound (float, optional): The upper bound for the yield search. Defaults to 1.5.

        Returns:
            float: The calculated yield-to-maturity (annually compounded).
        """
        valuation_datetime = _date_to_datetime(val_date)
        # cashflows = self.expected_cashflows()

        # For ActActICMA, we need the schedule and frequency for the day counter
        all_schedule_dates = [_date for _date, cpn in self.cashflows]
        coupon_freq = self._get_coupon_frequency()

        def target_function(r: float) -> float:
            """
            Calculates the difference between the PV of cashflows (for a given yield r) and the dirty price.
            The root of this function is the desired YTM.
            """
            # Calculate the dirty price for a given yield 'r' without creating a full DiscountCurve object. This ensures we use the bond's specific day counter for the yield calculation, matching QuantLib.
            pv_cashflows = 0.0
            for cf_date, amount in self.cashflows:
                if cf_date > valuation_datetime:
                    # Calculate year fraction for discounting using the bond's accrual day counter
                    yf = self._accrual_day_counter.yf(valuation_datetime, cf_date, coupon_schedule=all_schedule_dates, coupon_frequency=coupon_freq)

                    # Discount the cashflow (annually compounded, matching QL's default)
                    df = 1.0 / ((1.0 + r) ** yf)
                    pv_cashflows += df * amount

            return pv_cashflows - dirty_price

        # Use brentq to find the root of the target function (i.e., the yield)
        result = brentq(target_function, yield_search_lower_bound, yield_search_upper_bound, full_output=False)
        return result


class DiscountCurve(BaseDatedCurve):
    """
    A simple discount curve implementation based on a single flat interest rate.
    """

    def __init__(
        self,
        valuation_date: Union[date, datetime],
        flat_rate: Optional[float] = 0.05,
        curve_data: Any = None,
        day_counter_type: DayCounterType = DayCounterType.Act365Fixed,
    ):
        """
        Initializes the flat discount curve.

        Args:
            valuation_date (Union[date, datetime]): The valuation date of the curve.
            flat_rate (Optional[float], optional): The flat interest rate used for discounting. Defaults to 0.05.
            curve_data (Any, optional): Placeholder for more complex curve data (not used in this implementation). Defaults to None.
            day_counter_type (DayCounterType, optional): The day count convention for calculating year fractions. Defaults to DayCounterType.Act365Fixed.
        """
        self.valuation_date = valuation_date
        self._flat_rate = flat_rate
        self._curve_data = curve_data  # Placeholder for more complex curve data
        self._day_counter = DayCounter(day_counter_type)

    @property
    def valuation_date(self) -> datetime:
        """The valuation date of the curve as a datetime object."""
        return self._valuation_date

    @valuation_date.setter
    def valuation_date(self, value: Union[date, datetime]):
        self._valuation_date = _date_to_datetime(value)

    def get_discount_factor(self, target_date: Union[date, datetime]) -> float:
        """
        Calculates the discount factor from the valuation date to a target date.

        Args:
            target_date (Union[date, datetime]): The date to which to discount.

        Returns:
            float: The discount factor. Returns 0.0 if the target date is before the valuation date.
        """
        val_date_dt = _date_to_datetime(self.valuation_date)
        target_date_dt = _date_to_datetime(target_date)

        if target_date_dt < val_date_dt:
            return 0.0
        time_to_maturity_years = self._day_counter.yf(val_date_dt, target_date_dt)
        rate_to_use = self._flat_rate if self._flat_rate is not None else 0.02  # Fallback if flat_rate is None
        return 1 / ((1 + rate_to_use) ** time_to_maturity_years)

    def value(self, ref_date: datetime, target_date: datetime) -> float:
        """
        Returns the discount factor from a reference date to a target date.
        For this simple implementation, the reference date must be the curve's valuation date.

        Args:
            ref_date (datetime): The reference date (must match the curve's valuation date).
            target_date (datetime): The date to which to discount.

        Raises:
            ValueError: If the reference date does not match the curve's valuation date.

        Returns:
            float: The discount factor.
        """
        # Ensure ref_date matches the curve's valuation_date for this simple implementation
        if _date_to_datetime(ref_date).date() != self.valuation_date.date():
            raise ValueError(f"Reference date {ref_date} does not match DiscountCurve valuation date {self.valuation_date}")
        return self.get_discount_factor(target_date)


if __name__ == "__main__":
    obj_id = "DE000CZ40NT7"
    issue_date_dt = date(2019, 2, 1)
    maturity_date_dt = date(2021, 2, 1)
    coupon_rate_val = 0.03
    tenor_period = Period(years=1)
    notional_val = 100.0
    currency_val = "EUR"
    issuer_val = "Commerzbank"
    securitisation_level_val = SecuritizationLevel.NON_PREFERRED_SENIOR
    valuation_date_dt = date(2020, 2, 1)
    flat_rate_val = 0.02
    coupon_freq = Period(months=6)

    # --- RiVaPy Setup ---
    # Create a payment schedule for the bond
    rivapy_schedule = Schedule(
        start_day=issue_date_dt,
        end_day=maturity_date_dt,
        time_period=tenor_period,
        backwards=True,
        stub=True,
        business_day_convention=RollConvention.FOLLOWING,
        calendar=ECB(years=range(issue_date_dt.year, maturity_date_dt.year + 2)),
    )

    # Create the RiVaPy FixedRateBond instance
    rivapy_bond = FixedRateBond(
        obj_id=obj_id,
        schedule=rivapy_schedule,
        notional=notional_val,
        currency=currency_val,
        issue_date=issue_date_dt,
        coupon_rate=coupon_rate_val,
        coupon_freq=coupon_freq,
        issuer=issuer_val,
        securitization_level=securitisation_level_val,
        accrual_day_counter_type=DayCounterType.ACT_ACT,
    )
    valuation_date = date(2020, 4, 1)
    curve = DiscountCurve(valuation_date=valuation_date, flat_rate=0.03, day_counter_type=DayCounterType.ACT_ACT)

    rivapy_bond.expected_cashflows()
    rivapy_bond.compute_accrued_interest(valuation_date=valuation_date)
    rivapy_bond.compute_clean_price(discount_curve=curve)
    dirty_price = rivapy_bond.compute_dirty_price(discount_curve=curve)
    rivapy_bond.compute_yield(dirty_price=dirty_price, val_date=valuation_date)
    print()
