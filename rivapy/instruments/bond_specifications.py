import abc
import numpy as np
import rivapy.tools.interfaces as interfaces
from scipy.optimize import brentq
from collections import defaultdict
from typing import List, Union, Optional, Dict, Tuple
from rivapy.tools.enums import Currency, Rating, SecuritizationLevel, RollConvention
from rivapy.tools.datetools import _date_to_datetime, Schedule, Period, DayCounterType, DayCounter, roll_day
from rivapy.tools._validators import _check_positivity, _check_start_before_end, _string_to_calendar
from datetime import datetime, date
from holidays import HolidayBase as _HolidayBase, ECB as _ECB
from rivapy.marketdata.curves import DiscountCurve


class BondBaseSpecification(interfaces.FactoryObject):
    """Abstract base class for bond specifications."""

    def __init__(
        self,
        obj_id: str,
        schedule: Schedule,
        notional: float,
        currency: Union[Currency, str],
        issue_date: Union[date, datetime],
        maturity_date: Union[date, datetime],
        issuer: Optional[str] = None,
        securitization_level: Optional[Union[SecuritizationLevel, str]] = SecuritizationLevel.NONE,
        rating: Optional[Union[Rating, str]] = Rating.NONE,
        accrual_day_counter_type: DayCounterType = DayCounterType.ActActICMA,
    ):
        """
        Initializes the base bond specification.

        Args:
            obj_id (str): A unique identifier for the bond, e.g., ISIN.
            schedule (Schedule): The payment schedule of the bond.
            notional (float): The face value of the bond.
            currency (Union[Currency, str]): The currency of the bond.
            issue_date (Union[date, datetime]): The date the bond was issued.
            maturity_date (Union[date, datetime]): Maturity date of the bond.
            coupon_rate (float): The annual coupon rate (e.g., 0.05 for 5%).
            issuer (Optional[str], optional): The issuer of the bond. Defaults to None.
            securitization_level (Optional[Union[SecuritizationLevel, str]], optional): The securitization level. Defaults to SecuritizationLevel.NONE.
            rating (Optional[Union[Rating, str]], optional): The credit rating of the bond. Defaults to Rating.NONE.
            accrual_day_counter_type (DayCounterType, optional): The day count convention for accrual calculations. Defaults to DayCounterType.ActActICMA.
        """

        self.obj_id = obj_id

        if not isinstance(schedule, Schedule):
            raise TypeError("schedule must be an instance of rivapy.tools.datetools.Schedule.")

        self._schedule = schedule
        self._notional = notional
        self._currency = currency
        self._issue_date = _date_to_datetime(issue_date)
        self._maturity_date = _date_to_datetime(maturity_date)
        self._issuer = issuer
        self._securitization_level = securitization_level
        self._rating = rating

        _check_start_before_end(self._issue_date, self._maturity_date)

        self._coupon_rates: List[Tuple[datetime, datetime, float]] = []

        self._accrual_day_counter = DayCounter(accrual_day_counter_type)
        self._accrual_day_counter_type = accrual_day_counter_type

        self._coupon_freq = self._schedule.time_period

        if self._accrual_day_counter_type == DayCounterType.ActActICMA:
            _check = False
            for cp_freq_str in ["1Y", "6M", "3M"]:
                if self._coupon_freq == Period.from_string(cp_freq_str):
                    _check = True
                    break
            if _check == False:
                raise ValueError("For the Act/Act ICMA day count convention only a coupon frequency of 1Y, 6M or 3M is supported!")

        self._schedule_dates = self._schedule.generate_dates(ends_only=False)

    def _to_dict(self) -> Dict:
        # TODO: further addtion to the dictionary like Schedule
        return_dict = {
            "obj_id": self.obj_id,
            "issuer": self._issuer,
            "securitization_level": self._securitization_level,
            "issue_date": self._issue_date,
            "maturity_date": self._maturity_date,
            "currency": self._currency,
            "notional": self._notional,
            "rating": self._rating,
            "accrual_day_counter_type": self._accrual_day_counter_type,
        }
        return return_dict

    @property
    def cashflows(self) -> List[Tuple[datetime, float]]:
        return self._cashflows

    @property
    def coupon_rates(self) -> List[Tuple[datetime, datetime, float]]:
        return self._coupon_rates

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
        """The bond's issue date as a datetime object."""
        return self._maturity_date

    @maturity_date.setter
    def maturity_date(self, value: Union[date, datetime]):
        self._maturity_date = _date_to_datetime(value)

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

    @property
    def accrual_day_counter_type(self) -> str:
        return self._accrual_day_counter_type

    @accrual_day_counter_type.setter
    def accrual_day_counter_type(self, value: Union[DayCounterType, str]):
        self._accrual_day_counter_type = DayCounterType.to_string(value)

    def _get_coupon_frequency(self):
        if self._coupon_freq.years > 0:
            coupon_frequency = 1.0 / self._coupon_freq.years
        else:
            coupon_frequency = 12.0 / self._coupon_freq.months

        return coupon_frequency

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

        if (val_date_dt < self._coupon_rates[0][0]) or (val_date_dt > self._coupon_rates[-1][1]):
            raise ValueError(f" Valuation date {val_date_dt} is out of bonds {self._coupon_rates[0][0]} / {self._coupon_rates[-1][1]}")

        coupon_frequency = self._get_coupon_frequency()

        for dt_start, dt_end, coupon_rate in self._coupon_rates:
            if dt_start <= val_date_dt < dt_end:
                accrued_year_fraction = self._accrual_day_counter.yf(dt_start, val_date_dt, self._schedule_dates, coupon_frequency)
                return self._notional * coupon_rate * accrued_year_fraction

    @abc.abstractmethod
    def expected_cashflows(self) -> List[Tuple[datetime, float]]:
        """
        Computes all expected cashflows of the bond.

        Returns:
            List[Tuple[datetime, float]]: A list of tuples, where each tuple contains
                                          the payment date and the cashflow amount.
        """
        pass

    def compute_dirty_price(self, value_date: datetime, discount_curve: DiscountCurve, spread: float = 0.0) -> float:
        """
        Computes the dirty price of the bond.
        Dirty Price = Clean Price + Accrued Interest.
        The dirty price is the price of a bond including any accrued interest.

        Args:
            discount_curve (DiscountCurve): The curve used to discount future cashflows.

        Returns:
            float: The dirty price of the bond.
        """
        # val_date_dt = _date_to_datetime(discount_curve.valuation_date)
        # cashflows = self.cashflows#self.expected_cashflows()
        ref_date = discount_curve.refdate
        coupon_freq = self._get_coupon_frequency()

        pv_cashflows = 0.0
        for c in self._cashflows:
            if c[0] > value_date:
                rate = discount_curve.rivapy_value(refdate=ref_date, d=c[0], payment_dates=self._schedule_dates, annual_payment_frequency=coupon_freq)
                yf = self._accrual_day_counter.yf(d1=value_date, d2=c[0], coupon_schedule=self._schedule_dates, coupon_frequency=coupon_freq)
                df = 1 / ((1 + rate + spread) ** yf)
                pv_cashflows += df * c[1]
        return pv_cashflows

    def compute_clean_price(self, value_date: datetime, discount_curve: DiscountCurve, spread: float = 0.0) -> float:
        """
        Computes the clean price of the bond by discounting all future cashflows.

        Args:
            discount_curve (DiscountCurve): The curve used to discount future cashflows.

        Returns:
            float: The calculated clean price.
        """
        dirty_price = self.compute_dirty_price(value_date, discount_curve, spread)
        accrued_coupon = self.compute_accrued_interest(value_date)

        ref_date = discount_curve.refdate
        coupon_freq = self._get_coupon_frequency()

        accrued_coupon_pv = 0.0
        for c in self._cashflows:
            if c[0] > value_date:
                rate = discount_curve.rivapy_value(refdate=ref_date, d=c[0], payment_dates=self._schedule_dates, annual_payment_frequency=coupon_freq)
                yf = self._accrual_day_counter.yf(d1=value_date, d2=c[0], coupon_schedule=self._schedule_dates, coupon_frequency=coupon_freq)
                df = 1 / ((1 + rate + spread) ** yf)
                accrued_coupon_pv += df * accrued_coupon
                break

        return dirty_price - accrued_coupon_pv

    def compute_yield(
        self,
        dirty_price: float,
        val_date: datetime,
        spread: float = 0.0,
        yield_search_lower_bound: float = -0.2,
        yield_search_upper_bound: float = 1.5,
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
        all_schedule_dates = [_date for _date, cpn in self._cashflows]
        coupon_freq = self._get_coupon_frequency()

        def target_function(r: float) -> float:
            """
            Calculates the difference between the PV of cashflows (for a given yield r) and the dirty price.
            The root of this function is the desired YTM.
            """
            # Calculate the dirty price for a given yield 'r' without creating a full DiscountCurve object. This ensures we use the bond's specific day counter for the yield calculation, matching QuantLib.
            pv_cashflows = 0.0
            for cf_date, amount in self._cashflows:
                if cf_date > valuation_datetime:
                    # Calculate year fraction for discounting using the bond's accrual day counter
                    yf = self._accrual_day_counter.yf(valuation_datetime, cf_date, coupon_schedule=all_schedule_dates, coupon_frequency=coupon_freq)

                    # Discount the cashflow (annually compounded, matching QL's default)
                    df = 1.0 / ((1.0 + r + spread) ** yf)
                    pv_cashflows += df * amount

            return pv_cashflows - dirty_price

        # Use brentq to find the root of the target function (i.e., the yield)
        result = brentq(target_function, yield_search_lower_bound, yield_search_upper_bound, full_output=False)
        return result


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
        maturity_date: Union[date, datetime],
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
            maturity_date (Union[date, datetime]): Maturity date of the bond.
            coupon_rate (float): The annual coupon rate (e.g., 0.05 for 5%).
            issuer (Optional[str], optional): The issuer of the bond. Defaults to None.
            securitization_level (Optional[Union[SecuritizationLevel, str]], optional): The securitization level. Defaults to SecuritizationLevel.NONE.
            rating (Optional[Union[Rating, str]], optional): The credit rating of the bond. Defaults to Rating.NONE.
            accrual_day_counter_type (DayCounterType, optional): The day count convention for accrual calculations. Defaults to DayCounterType.ActActICMA.
        """
        super().__init__(
            obj_id,
            schedule,
            notional,
            currency,
            issue_date,
            maturity_date,
            issuer,
            securitization_level,
            rating,
            accrual_day_counter_type,
        )
        if coupon_rate < 0:
            raise ValueError("Coupon rate must be non-negative.")

        self._coupon_rate = coupon_rate

        self._cashflows = self.expected_cashflows()

    @property
    def coupon_rate(self) -> float:
        """The bond's annual coupon rate"""
        return self._coupon_rate

    @coupon_rate.setter
    def coupon_rate(self, value: float):
        self._coupon_rate = _check_positivity(value)

    def _to_dict(self):
        _dict = super()._to_dict()
        _dict["coupon_rate"] = self._coupon_rate
        return _dict

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

    def expected_cashflows(self) -> List[Tuple[datetime, float]]:
        """
        Computes all expected cashflows (coupons and notional) of the bond.

        Returns:
            List[Tuple[datetime, float]]: A sorted list of (date, amount) tuples for each cashflow.
        """
        # Generate all schedule dates, which are already business-day adjusted

        cashflows = []
        # Get frequency information needed for Act/Act ICMA calculation
        coupon_freq = self._get_coupon_frequency()

        # --- Iterate over all coupon periods ---
        for i in range(len(self._schedule_dates) - 1):
            period_start_dt = _date_to_datetime(self._schedule_dates[i])
            payment_date_dt = _date_to_datetime(self._schedule_dates[i + 1])

            # Skip cashflows that are paid out before or on the issue date
            if payment_date_dt <= self.issue_date:
                continue

            # The accrual for the coupon calculation always uses the full period from the schedule
            # to correctly handle stub periods (short/long first or last coupons).
            year_fraction_for_coupon = self._accrual_day_counter.yf(
                period_start_dt, payment_date_dt, coupon_schedule=self._schedule_dates, coupon_frequency=coupon_freq  # coupon_frequency_int
            )

            coupon_amount = self._notional * self._coupon_rate * year_fraction_for_coupon
            if coupon_amount > 0.0:
                cashflows.append((payment_date_dt, coupon_amount))

            self._coupon_rates.append((period_start_dt, payment_date_dt, self._coupon_rate))

        # Add notional at maturity date (which is the last date in the schedule)
        maturity_payment_date = roll_day(
            _date_to_datetime(self._maturity_date), calendar=self._schedule.calendar, business_day_convention=self._schedule.business_day_convention
        )
        if maturity_payment_date >= self._issue_date:
            cashflows.append((maturity_payment_date, self._notional))

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
        return super().compute_accrued_interest(valuation_date=valuation_date)


class FloatingRateBond(BondBaseSpecification):
    def __init__(
        self,
        obj_id: str,
        schedule: Schedule,
        notional: float,
        currency: Union[Currency, str],
        issue_date: Union[date, datetime],
        maturity_date: Union[date, datetime],
        coupon_ref_curve: DiscountCurve,
        margin: float,
        fixing_date: Union[date, datetime],
        fixing_coupon_rate: float,
        issuer: Optional[str] = None,
        securitization_level: Optional[Union[SecuritizationLevel, str]] = SecuritizationLevel.NONE,
        rating: Optional[Union[Rating, str]] = Rating.NONE,
        accrual_day_counter_type: DayCounterType = DayCounterType.ActActICMA,
    ):
        super().__init__(
            obj_id,
            schedule,
            notional,
            currency,
            issue_date,
            maturity_date,
            issuer,
            securitization_level,
            rating,
            accrual_day_counter_type,
        )
        self._coupon_ref_curve = coupon_ref_curve
        self._margin = margin
        self._fixing_date = _date_to_datetime(fixing_date)
        self._fixing_coupon_rate = fixing_coupon_rate

        _check_start_before_end(self._fixing_date, self._maturity_date)

        self._cashflows = self.expected_cashflows()

    @property
    def coupon_ref_curve(self) -> DiscountCurve:
        return self._coupon_ref_curve

    @property
    def margin(self) -> float:
        return self._margin

    @margin.setter
    def margin(self, value: float):
        self._margin = _check_positivity(value=value)

    @property
    def fixing_date(self) -> datetime:
        return self._fixing_date

    @fixing_date.setter
    def fixing_date(self, value: Union[datetime, date]):
        self._fixing_date = _date_to_datetime(value)

    @property
    def fixing_coupon_rate(self) -> float:
        return self._fixing_coupon_rate

    @fixing_coupon_rate.setter
    def fixing_coupon_rate(self, value: float):
        self._fixing_coupon_rate = _check_positivity(value=value)

    def _to_dict(self):
        _dict = {"coupon_rate_spread": self._margin, "fixing_date": self._fixing_date, "fixing_coupon_rate": self._fixing_coupon_rate}
        return {**super()._to_dict(), **_dict}

    def expected_cashflows(self) -> List[Tuple[datetime, float]]:
        """
        Computes all expected cashflows (coupons and notional) of the bond.

        Returns:
            List[Tuple[datetime, float]]: A sorted list of (date, amount) tuples for each cashflow.
        """

        cashflows = []
        # Get frequency information needed for Act/Act ICMA calculation
        coupon_freq = self._get_coupon_frequency()

        _fixed_coupon = False
        _curve_ref_date = self._coupon_ref_curve.refdate

        for i in range(len(self._schedule_dates) - 1):
            period_start_dt = _date_to_datetime(self._schedule_dates[i])
            payment_date_dt = _date_to_datetime(self._schedule_dates[i + 1])

            if payment_date_dt < self._fixing_date:
                continue

            if (not _fixed_coupon) and (self._fixing_date <= payment_date_dt):
                # first coupon payment after the fixing date has a fixed coupon rate
                coupon_rate = self._fixing_coupon_rate
                _fixed_coupon = True

            elif (
                (_fixed_coupon)
                and (payment_date_dt < _date_to_datetime(self._coupon_ref_curve.get_dates()[0]))
                and (self._fixing_date < payment_date_dt)
            ):
                # coupon payment falls between fixing date (without being the first coupon payment after the fixing date) and the start of the coupon rate curve
                raise ValueError(
                    f"Coupon payment {payment_date_dt} is not the first coupon payment after fixing and happens before the start date of the coupon reference curve {_date_to_datetime(self._coupon_ref_curve.get_dates()[0])}"
                )
            else:
                coupon_rate = (
                    self._coupon_ref_curve.rivapy_value(
                        refdate=_curve_ref_date, d=payment_date_dt, payment_dates=self._schedule_dates, annual_payment_frequency=coupon_freq
                    )
                    + self._margin
                )

            year_fraction_for_coupon = self._accrual_day_counter.yf(
                period_start_dt, payment_date_dt, coupon_schedule=self._schedule_dates, coupon_frequency=coupon_freq  # coupon_frequency_int
            )

            coupon = self._notional * coupon_rate * year_fraction_for_coupon
            if coupon > 0.0:
                cashflows.append((payment_date_dt, coupon))

            self._coupon_rates.append((period_start_dt, payment_date_dt, coupon_rate))

        # Add notional at maturity date (which is the last date in the schedule)
        maturity_payment_date = roll_day(
            _date_to_datetime(self._maturity_date), calendar=self._schedule.calendar, business_day_convention=self._schedule.business_day_convention
        )
        if maturity_payment_date >= self._issue_date:
            cashflows.append((maturity_payment_date, self._notional))

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
        return super().compute_accrued_interest(valuation_date=valuation_date)


# class BondBaseSpecification(interfaces.FactoryObject):

#     def __init__(self,
#                  obj_id: str,
#                  issue_date: _Union[date, datetime],
#                  maturity_date: _Union[date, datetime],
#                  currency: _Union[Currency, str] = 'EUR',
#                  notional: float = 100.0,
#                  issuer: str = None,
#                  securitization_level: _Union[SecuritizationLevel, str] = SecuritizationLevel.NONE,
#                  rating: _Union[Rating, str] = Rating.NONE):
#         """Base bond specification.

#         Args:
#             obj_id (str): (Preferably) Unique label of the bond, e.g. ISIN.
#             issue_date (_Union[date, datetime]): Date of bond issuance.
#             maturity_date (_Union[date, datetime]): Bond's maturity/expiry date. Must lie after the issue_date.
#             currency (str, optional): Currency as alphabetic, Defaults to 'EUR'.
#             notional (float, optional): Bond's notional/face value. Must be positive. Defaults to 100.0.
#             issuer (str, optional): Name/id of issuer. Defaults to None.
#             securitization_level (_Union[SecuritizationLevel, str], optional): Securitization level. Defaults to None.
#             rating (_Union[Rating, str]): Paper rating.
#         """
#         self.obj_id = obj_id
#         if issuer is not None:
#             self.issuer = issuer
#         if securitization_level is not None:
#             self.securitization_level = securitization_level
#         self.issue_date = issue_date
#         self.maturity_date = maturity_date
#         self.currency = currency
#         self.notional = notional
#         self.rating = Rating.to_string(rating)
#         # validate dates
#         self._validate_derived_issued_instrument()

#     @staticmethod
#     def _create_sample(n_samples: int, seed: int = None, ref_date = None,
#                     issuers: _List[str]= None,
#                     sec_levels: _List[str]=None,
#                     currencies: _List[str]= None)->_List[dict]:
#         if seed is not None:
#             np.random.seed(seed)
#         if ref_date is None:
#             ref_date = datetime.now()
#         else:
#             ref_date = _date_to_datetime(ref_date)
#         if issuers is None:
#             issuers = ['Issuer_'+str(i) for i in range(int(n_samples/2))]
#         result = []
#         if currencies is None:
#             currencies = list(Currency)
#         if sec_levels is None:
#             sec_levels = list(SecuritizationLevel)
#         for _ in range(n_samples):
#             days = int(15.0*365.0*np.random.beta(2.0,2.0)) + 1
#             issue_date = ref_date + timedelta(days=np.random.randint(low=-365, high=0))
#             result.append(
#                 {
#                 'issue_date': issue_date,
#                 'maturity_date': ref_date + timedelta(days=days),
#                 'currency':np.random.choice(currencies),
#                 'notional': np.random.choice([100.0, 1000.0, 10_000.0, 100_0000.0]),
#                 'issuer': np.random.choice(issuers),
#                 'securitization_level': np.random.choice(sec_levels)
#                 }
#             )
#         return result

#     def _validate_derived_issued_instrument(self):
#         self.__issue_date, self.__maturity_date = _check_start_before_end(self.__issue_date, self.__maturity_date)

#     def _to_dict(self)->dict:
#         result = {
#             'obj_id': self.obj_id, 'issuer':self.issuer,
#             'securitization_level': self.securitization_level,
#             'issue_date': self.issue_date, 'maturity_date':self.maturity_date,
#             'currency': self.currency, 'notional': self.notional,
#             'rating': self.rating
#         }
#         return result

#     #region properties

#     @property
#     def issuer(self) -> str:
#         """
#         Getter for instrument's issuer.

#         Returns:
#             str: Instrument's issuer.
#         """
#         return self.__issuer

#     @issuer.setter
#     def issuer(self, issuer: str):
#         """
#         Setter for instrument's issuer.

#         Args:
#             issuer(str): Issuer of the instrument.
#         """
#         self.__issuer = issuer

#     @property
#     def rating(self)->str:
#         return self.__rating

#     @rating.setter
#     def rating(self, rating:_Union[Rating, str])->str:
#         self.__rating = Rating.to_string(rating)

#     @property
#     def securitization_level(self) -> str:
#         """
#         Getter for instrument's securitisation level.

#         Returns:
#             str: Instrument's securitisation level.
#         """
#         return self.__securitization_level

#     @securitization_level.setter
#     def securitization_level(self, securitisation_level:  _Union[SecuritizationLevel, str]):
#         self.__securitization_level = SecuritizationLevel.to_string(securitisation_level)

#     @property
#     def issue_date(self) -> date:
#         """
#         Getter for bond's issue date.

#         Returns:
#             date: Bond's issue date.
#         """
#         return self.__issue_date

#     @issue_date.setter
#     def issue_date(self, issue_date: _Union[datetime, date]):
#         """
#         Setter for bond's issue date.

#         Args:
#             issue_date (Union[datetime, date]): Bond's issue date.
#         """
#         self.__issue_date = _date_to_datetime(issue_date)

#     @property
#     def maturity_date(self) -> date:
#         """
#         Getter for bond's maturity date.

#         Returns:
#             date: Bond's maturity date.
#         """
#         return self.__maturity_date

#     @maturity_date.setter
#     def maturity_date(self, maturity_date: _Union[datetime, date]):
#         """
#         Setter for bond's maturity date.

#         Args:
#             maturity_date (Union[datetime, date]): Bond's maturity date.
#         """
#         self.__maturity_date = _date_to_datetime(maturity_date)

#     @property
#     def currency(self) -> str:
#         """
#         Getter for bond's currency.

#         Returns:
#             str: Bond's ISO 4217 currency code
#         """
#         return self.__currency

#     @currency.setter
#     def currency(self, currency:str):
#         self.__currency = Currency.to_string(currency)

#     @property
#     def notional(self) -> float:
#         """
#         Getter for bond's face value.

#         Returns:
#             float: Bond's face value.
#         """
#         return self.__notional

#     @notional.setter
#     def notional(self, notional):
#         self.__notional = _check_positivity(notional)

#     #endregion

# class ZeroCouponBondSpecification(BondBaseSpecification):
#     def __init__(self,
#                  obj_id: str,
#                  issue_date: _Union[date, datetime],
#                  maturity_date: _Union[date, datetime],
#                  currency: str = 'EUR',
#                  notional: float = 100.0,
#                  issuer: str = None,
#                 securitization_level: _Union[SecuritizationLevel, str] = None,
#                  rating: _Union[Rating, str] = Rating.NONE):
#         """Zero coupon bond specification.

#         Args:
#             obj_id (str): (Preferably) Unique label of the bond, e.g. ISIN.
#             issue_date (_Union[date, datetime]): Date of bond issuance.
#             maturity_date (_Union[date, datetime]): Bond's maturity/expiry date. Must lie after the issue_date.
#             currency (str, optional): Currency as alphabetic, Defaults to 'EUR'.
#             notional (float, optional): Bond's notional/face value. Must be positive. Defaults to 100.0.
#             issuer (str, optional): Name/id of issuer. Defaults to None.
#             securitization_level (_Union[SecuritizationLevel, str], optional): Securitization level. Defaults to None.
#             rating (_Union[Rating, str]): Paper rating.
#         """
#         super().__init__(obj_id,
#                          issue_date,
#                          maturity_date,
#                          currency,
#                          notional,
#                          issuer,
#                          securitization_level)

#     @staticmethod
#     def _create_sample(n_samples: int, seed: int = None, ref_date = None,
#                         issuers: _List[str]= None, sec_levels: _List[str]=None,
#                     currencies: _List[str]= None):
#         specs = BondBaseSpecification._create_sample(**locals())
#         result = []
#         for i, b in enumerate(specs):
#             result.append(ZeroCouponBondSpecification('ZC_BND_'+str(i), **b))
#         return result

#     def _validate_derived_bond(self):
#         pass

#     def _validate_derived_issued_instrument(self):
#         pass

#     def expected_cashflows(self)->_List[Tuple[datetime, float]]:
#         """Return a list of all expected cashflows (here only the final notional) together with their payment date.

#         Returns:
#             _List[Tuple[datetime, float]]: The resulting list of all cashflows.
#         """
#         return [(self.maturity_date, self.notional)]


# class PlainVanillaCouponBondSpecification(BondBaseSpecification):
#     def __init__(self,
#                  obj_id: str,
#                  issue_date: _Union[date, datetime],
#                  maturity_date: _Union[date, datetime],
#                  accrual_start: _Union[date, datetime],
#                  coupon_freq: str,
#                  coupon: float,
#                  currency: str = 'EUR',
#                  notional: float = 100.0,
#                  issuer: str = None,
#                   securitization_level: _Union[SecuritizationLevel, str] = None,
#                   stub: bool = True,
#                   rating: _Union[Rating, str] = Rating.NONE):
#         """PlainVanillaCouponBond specification.

#         Args:
#             obj_id (str): (Preferably) Unique label of the bond, e.g. ISIN.
#             issue_date (_Union[date, datetime]): Date of bond issuance.
#             maturity_date (_Union[date, datetime]): Bond's maturity/expiry date. Must lie after the issue_date.
#             first_coupondate (_Union[date, datetime]): The first coupon date.
#             coupon_freq (str): Frequency of coupons. Defaults to '1Y' for yearly. Internally, the method :func:`rivapy.tools.Period.from_string` is used, see the definition of valid strings there.
#             coupon (float): Coupon as relative number (multiplied internaly by notional to get absolute cashflow).
#             currency (str, optional): Currency as alphabetic, Defaults to 'EUR'.
#             notional (float, optional): Bond's notional/face value. Must be positive. Defaults to 100.0.
#             issuer (str, optional): Name/id of issuer. Defaults to None.
#             securitization_level (_Union[SecuritizationLevel, str], optional): Securitization level. Defaults to None.
#             rating ( _Union[Rating, str]): Paper rating.
#         """
#         super().__init__(obj_id,
#                          issue_date,
#                          maturity_date,
#                          currency,
#                          notional,
#                          issuer,
#                          securitization_level,
#                          rating)

#         self.accrual_start = accrual_start
#         self.coupon_freq = coupon_freq
#         self.coupon = coupon
#         self.stub = stub

#     def expected_cashflows(self)->_List[Tuple[datetime, float]]:
#         """Return a list of all expected cashflows (final notional and coupons) together with their payment date.

#         Returns:
#             _List[Tuple[datetime, float]]: The resulting list of all cashflows.
#         """
#         #if self.coupon_freq != 'Y':
#         #    raise Exception('Cannot calc cashflows for other than yearly coupons. Missing transformation from yearly coupon to .... ')
#         period = Period.from_string(self.coupon_freq)
#         coupon_multiplier = 1.0
#         if period.years > 0:
#             coupon_multiplier = period.years
#         elif period.months > 0:
#             coupon_multiplier = period.months/12.0
#         elif period.days > 0:
#             coupon_multiplier = period.days/365.0
#         schedule = Schedule(self.accrual_start, self.maturity_date, period, stub=self.stub).generate_dates(ends_only=True)
#         result = [(d, self.coupon*coupon_multiplier*self.notional) for d in schedule]
#         result.insert(0, (self.accrual_start, 0.0))# the first entry of this schedule is the accrual start which has a cashflow of zero and is just used for accrual calculation
#         result.append((self.maturity_date, self.notional))
#         return result

#     def _to_dict(self) -> dict:
#         result = {
#             'accrual_start': self.accrual_start,
#             'coupon_freq' : self.coupon_freq,
#             'coupon' : self.coupon,
#             }
#         result.update(super(PlainVanillaCouponBondSpecification, self)._to_dict())
#         return result

#     @staticmethod
#     def _create_sample(n_samples: int, seed: int = None, ref_date = None,
#                         issuers: _List[str]= None, sec_levels: _List[str]=None,
#                     currencies: _List[str]= None):
#         specs = BondBaseSpecification._create_sample(**locals())
#         result = []
#         coupons = np.arange(0.0, 0.09, 0.0025)
#         for i, b in enumerate(specs):
#             b['coupon_freq'] = np.random.choice(['3M', '6M', '9M', '1Y'], p=[0.1,0.4,0.1,0.4])
#             issue_date = b['issue_date']
#             b['accrual_start'] = issue_date + timedelta(days=np.random.randint(low=0, high=10))
#             b['coupon'] = np.random.choice(coupons)
#             result.append(PlainVanillaCouponBondSpecification('BND_PV_'+str(i), **b))
#         return result

# class FixedRateBondSpecification(BondBaseSpecification):
#     def __init__(self,
#                  obj_id: str,
#                  issue_date: _Union[date, datetime],
#                  maturity_date: _Union[date, datetime],
#                  coupon_payment_dates: _List[_Union[date, datetime]],
#                  coupons: _List[float],
#                 currency: str = 'EUR',
#                  notional: float = 100.0,
#                  issuer: str = None,
#                  securitization_level: _Union[SecuritizationLevel, str] = None,
#                  rating: _Union[Rating, str] = Rating.NONE):
#         """
#         Fixed rate bond specification by providing coupons and coupon payment dates directly.

#         Args:
#             coupon_payment_dates (List[Union[date, datetime]]): List of annualised coupon payment dates.
#             coupons (List[float]): List of annualised coupon amounts as fraction of notional.
#         """
#         super().__init__(obj_id, issue_date, maturity_date, currency, notional, issuer, securitization_level, rating)
#         self.__coupon_payment_dates = coupon_payment_dates
#         self.__coupons = coupons
#         # validation of dates' consistency
#         if not _is_ascending_date_list(issue_date, coupon_payment_dates, maturity_date):
#             raise Exception("Inconsistent combination of issue date '" + str(issue_date)
#                             + "', payment dates '" + str(coupon_payment_dates)
#                             + "', and maturity date '" + str(maturity_date) + "'.")
#             # TODO: Clarify if inconsistency should be shown explicitly.
#         if len(coupon_payment_dates) == len(coupons):
#             self.__coupons = coupons
#         else:
#             raise Exception('Number of coupons ' + str(coupons) +
#                             ' is not equal to number of coupon payment dates ' + str(coupon_payment_dates))


#     @staticmethod
#     def _create_sample(n_samples: int, seed: int = None, ref_date = None, issuers: _List[str]= None):
#         specs = BondBaseSpecification._create_sample(**locals())
#         result = []
#         coupons = np.arange(0.01, 0.09, 0.005)
#         for i, b in enumerate(specs):
#             issue_date = b['issue_date']
#             n_coupons = np.random.randint(low=1, high=20)
#             days_coupon_period = np.random.choice([90.0,180.0,365.0], p=[0.2,0.2,0.6])
#             b['coupon_payment_dates'] = [issue_date+timedelta(days=(i+1)*days_coupon_period) for i in range(n_coupons)]
#             coupon = np.random.choice(coupons)
#             b['coupons'] = [coupon]*n_coupons
#             b['maturity_date'] = b['coupon_payment_dates'][-1]
#             result.append(FixedRateBondSpecification('BND_FR_'+str(i), **b))
#         return result

#     def _validate_derived_bond(self):
#         self.__coupon_payment_dates = _datetime_to_date_list(self.__coupon_payment_dates)
#         # validation of dates' consistency
#         if not _is_ascending_date_list(self.__issue_date, self.__coupon_payment_dates, self.__maturity_date):
#             raise Exception("Inconsistent combination of issue date '" + str(self.__issue_date)
#                             + "', payment dates '" + str(self.__coupon_payment_dates)
#                             + "', and maturity date '" + str(self.__maturity_date) + "'.")
#             # TODO: Clarify if inconsistency should be shown explicitly.
#         if len(self.__coupon_payment_dates) != len(self.__coupons):
#             raise Exception('Number of coupons ' + str(self.__coupons) +
#                             ' is not equal to number of coupon payment dates ' + str(self.__coupon_payment_dates))

#     def _validate_derived_issued_instrument(self):
#         pass


#     def _to_dict(self)->dict:
#         result = {
#             'coupon_payment_dates': self.__coupon_payment_dates,
#             'coupons' : self.__coupons
#             }
#         result.update(super(FixedRateBondSpecification, self)._to_dict())
#         return result

#     @classmethod
#     def from_master_data(cls,
#                          obj_id: str,
#                          issue_date: _Union[date, datetime],
#                          maturity_date: _Union[date, datetime],
#                          coupon: float,
#                          tenor: _Union[Period, str],
#                          backwards: bool = True,
#                          stub: bool = False,
#                          business_day_convention: _Union[RollConvention, str] = RollConvention.FOLLOWING,
#                          calendar: _Union[_HolidayBase, str] = None,
#                          currency: str = 'EUR',
#                          notional: float = 100.0,
#                          issuer: str = None,
#                          securitisation_level: _Union[SecuritizationLevel, str] = None):
#         """
#         Fixed rate bond specification based on bond's master data.

#         Args:
#             # TODO: How can we avoid repeating ourselves here?
#             obj_id (str): (Preferably) Unique label of the bond, e.g. ISIN.
#             issue_date (Union[date, datetime]): Date of bond issuance.
#             maturity_date (Union[date, datetime]): Bond's maturity/expiry date. Must lie after the issue_date.

#             coupon (float): Annualised coupon amount as fraction of notional, e.g. 0.0125 for fixed rate coupon of
#                             1.25%.
#             tenor: (Union[period, str]): Time distance between two coupon payment dates.
#             backwards (bool, optional): Defines direction for rolling out the schedule. True means the schedule will be
#                                         rolled out (backwards) from maturity date to issue date. Defaults to True.
#             stub (bool, optional): Defines if the first/last period is accepted (True), even though it is shorter than
#                                    the others, or if it remaining days are added to the neighbouring period (False).
#                                    Defaults to True.
#             business_day_convention (Union[RollConvention, str], optional): Set of rules defining the adjustment of
#                                                                             days to ensure each date being a business
#                                                                             day with respect to a given holiday
#                                                                             calendar. Defaults to
#                                                                             RollConvention.FOLLOWING
#             calendar (Union[HolidayBase, str], optional): Holiday calendar defining the bank holidays of a country or
#                                                            province (but not all non-business days as for example
#                                                            Saturdays and Sundays).
#                                                            Defaults (through constructor) to holidays.ECB
#                                                            (= Target2 calendar) between start_day and end_day.
#             # TODO: How can we avoid repeating ourselves here?
#             currency (str, optional): Currency as alphabetic  according to iso
#                                                             currency code ISO 4217
#                                                             (cf. https://www.iso.org/iso-4217-currency-codes.html).
#                                                             Defaults to 'EUR'.
#             notional (float, optional): Bond's notional/face value. Must be positive. Defaults to 100.0.
#             issuer (str, optional): Issuer of the instrument. Defaults to None.
#             securitisation_level (Union[SecuritizationLevel, str], optional): Securitisation level of the instrument.
#                                                                               Defaults to None.

#         Returns:
#             FixedRateBond: Corresponding fixed rate bond with already generated schedule for coupon payments.
#         """
#         coupon = _check_positivity(coupon)
#         tenor = _term_to_period(tenor)
#         business_day_convention = RollConvention.to_string(business_day_convention)
#         if calendar is None:
#             calendar = _ECB(years=range(issue_date.year, maturity_date.year + 1))
#         else:
#             calendar = _string_to_calendar(calendar)
#         schedule = Schedule(issue_date, maturity_date, tenor, backwards, stub, business_day_convention, calendar)
#         coupon_payment_dates = schedule.generate_dates(True)
#         coupons = [coupon] * len(coupon_payment_dates)
#         securitisation_level = SecuritizationLevel.to_string(securitisation_level)
#         return FixedRateBondSpecification(obj_id, issue_date, maturity_date, coupon_payment_dates, coupons, currency, notional,
#                              issuer, securitisation_level)

#     @property
#     def coupon_payment_dates(self) -> _List[date]:
#         """
#         Getter for payment dates for fixed coupons.

#         Returns:
#             List[date]: List of dates for fixed coupon payments.
#         """
#         return self.__coupon_payment_dates

#     @property
#     def coupons(self) -> _List[float]:
#         """
#         Getter for fixed coupon payments.

#         Returns:
#             List[float]: List of coupon amounts expressed as annualised fractions of bond's face value.
#         """
#         return self.__coupons

# class FloatingRateNoteSpecification(BondBaseSpecification):
#     def __init__(self,
#                  obj_id: str,
#                  issue_date: _Union[date, datetime],
#                  maturity_date: _Union[date, datetime],
#                  coupon_period_dates: _List[_Union[date, datetime]],
#                  day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
#                  spreads: _List[float] = None,
#                  reference_index: str = 'dummy_curve',
#                  currency: str = 'EUR',
#                  notional: float = 100.0,
#                  issuer: str = None,
#                  securitisation_level: _Union[SecuritizationLevel, str] = None):
#         """
#         Floating rate note specification by providing coupon periods directly.

#         Args:
#             coupon_period_dates (List[_Union[date, datetime]): Floating rate note's coupon periods, i.e. beginning and
#                                                                ends of the accrual periods for the floating rate coupon
#                                                                payments.
#             day_count_convention (Union[DayCounter, str], optional): Day count convention for determining period
#                                                                      length. Defaults to DayCounter.ThirtyU360.
#             spreads (List[float], optional): List of spreads added to the floating rates derived from fixing the
#                                              reference curve as fraction of notional. Defaults to None.
#             reference_index (str, optional): Floating rate note underlying reference curve used for fixing the floating
#                                              rate coupon amounts. Defaults to 'dummy_curve'.
#                                              Note: A reference curve could also be provided later at the pricing stage.
#         """
#         #super().__init__(obj_id, issue_date, maturity_date, currency, notional, issuer, securitisation_level)
#         BondBaseSpecification.__init__(self, obj_id, issue_date, maturity_date, currency, notional, issuer, securitisation_level)
#         self.__coupon_period_dates = _datetime_to_date_list(coupon_period_dates)
#         # validation of dates' consistency
#         if not _is_ascending_date_list(issue_date, coupon_period_dates, maturity_date, False):
#             raise Exception("Inconsistent combination of issue date '" + str(issue_date)
#                             + "', payment dates '" + str(coupon_period_dates)
#                             + "', and maturity date '" + str(maturity_date) + "'.")
#             # TODO: Clarify if inconsistency should be shown explicitly.
#         self.__day_count_convention = DayCounterType.to_string(day_count_convention)
#         if spreads is None:
#             self.__spreads = [0.0] * (len(coupon_period_dates) - 1)
#         elif len(spreads) == len(coupon_period_dates) - 1:
#             self.__spreads = spreads
#         else:
#             raise Exception('Number of spreads ' + str(spreads) +
#                             ' does not fit to number of coupon periods ' + str(coupon_period_dates))
#         if reference_index == '':
#             # do not leave reference curve empty as this causes pricer to ignore floating rate coupons!
#             self.__reference_index = 'dummy_curve'
#         else:
#             self.__reference_index = reference_index

#     @classmethod
#     def from_master_data(cls,
#                          obj_id: str,
#                          issue_date: _Union[date, datetime],
#                          maturity_date: _Union[date, datetime],
#                          tenor: _Union[Period, str],
#                          backwards: bool = True,
#                          stub: bool = False,
#                          business_day_convention: _Union[RollConvention, str] = RollConvention.FOLLOWING,
#                          calendar: _Union[_HolidayBase, str] = None,
#                          day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
#                          spread: float = 0.0,
#                          reference_index: str = 'dummy_curve',
#                          currency: str = 'EUR',
#                          notional: float = 100.0,
#                          issuer: str = None,
#                          securitisation_level: _Union[SecuritizationLevel, str] = None):
#         """
#         Floating rate note specification based on master data.

#         Args:
#             # TODO: How can we avoid repeating ourselves here?
#             obj_id (str): (Preferably) Unique label of the bond, e.g. ISIN.
#             issue_date (Union[date, datetime]): Date of bond issuance.
#             maturity_date (Union[date, datetime]): Bond's maturity/expiry date. Must lie after the issue_date.

#             tenor: (Union[period, str]): Time distance between two coupon payment dates.
#             backwards (bool, optional): Defines direction for rolling out the schedule. True means the schedule will be
#                                         rolled out (backwards) from maturity date to issue date. Defaults to True.
#             stub (bool, optional): Defines if the first/last period is accepted (True), even though it is shorter than
#                                    the others, or if it remaining days are added to the neighbouring period (False).
#                                    Defaults to True.
#             business_day_convention (Union[RollConvention, str], optional): Set of rules defining the adjustment of
#                                                                             days to ensure each date being a business
#                                                                             day with respect to a given holiday
#                                                                             calendar. Defaults to
#                                                                             RollConvention.FOLLOWING
#             calendar (Union[HolidayBase, str], optional): Holiday calendar defining the bank holidays of a country or
#                                                           province (but not all non-business days as for example
#                                                           Saturdays and Sundays).
#                                                           Defaults (through constructor) to holidays.ECB
#                                                           (= Target2 calendar) between start_day and end_day.
#             # TODO: How can we avoid repeating ourselves here?
#             day_count_convention (Union[DayCounter, str], optional): Day count convention for determining period
#                                                                      length. Defaults to DayCounter.ThirtyU360.
#             spread (float, optional): Spread added to floating rate derived from fixing the reference curve as fraction
#                                       of notional, i.e. 0.0025 for 25 basis points. Defaults to 0.0.
#             reference_index (str, optional): Floating rate note underlying reference curve used for fixing the floating
#                                              rate coupon amounts. Defaults to 'dummy_curve'.
#                                              Note: A reference curve could also be provided later at the pricing stage.
#             currency (str, optional): Currency as alphabetic code according to iso
#                                                             currency code ISO 4217
#                                                             (cf. https://www.iso.org/iso-4217-currency-codes.html).
#                                                             Defaults to 'EUR'.
#             notional (float, optional): Bond's notional/face value. Must be positive. Defaults to 100.0.
#             issuer (str, optional): Issuer of the instrument. Defaults to None.
#             securitisation_level (Union[SecuritizationLevel, str], optional): Securitisation level of the instrument.
#                                                                              Defaults to None.
#         Returns:
#             FloatingRateNote: Corresponding floating rate note with already generated schedule for coupon payments.
#         """
#         tenor = _term_to_period(tenor)
#         business_day_convention = RollConvention.to_string(business_day_convention)
#         if calendar is None:
#             calendar = _ECB(years=range(issue_date.year, maturity_date.year + 1))
#         else:
#             calendar = _string_to_calendar(calendar)
#         schedule = Schedule(issue_date, maturity_date, tenor, backwards, stub, business_day_convention, calendar)
#         coupon_period_dates = schedule.generate_dates(False)
#         spreads = [spread] * (len(coupon_period_dates) - 1)
#         return FloatingRateNoteSpecification(obj_id, issue_date, maturity_date, coupon_period_dates, day_count_convention, spreads,
#                                 reference_index, currency, notional, issuer, securitisation_level)

#     @property
#     def coupon_period_dates(self) -> _List[date]:
#         """
#         Getter for accrual periods for floating rate coupons.

#         Returns:
#             List[date]: List of accrual periods for floating rate coupons.
#         """
#         return self.__coupon_period_dates

#     @property
#     def daycount_convention(self) -> str:
#         """
#         Getter for bond's day count convention.

#         Returns:
#             str: Bond's day count convention.
#         """
#         return self.__day_count_convention

#     @daycount_convention.setter
#     def daycount_convention(self, day_count_convention: _Union[DayCounterType, str])-> str:
#         self.__day_count_convention = DayCounterType.to_string(day_count_convention)

#     @property
#     def spreads(self) -> _List[float]:
#         """
#         Getter for spreads added to the floating rates determined by fixing of reference index.

#         Returns:
#             List[float]: List of spreads added to the floating rates determined by fixing of reference index.
#         """
#         return self.__spreads

#     @property
#     def reference_index(self) -> str:
#         """
#         Getter for reference index for fixing floating rates.

#         Returns:
#             str: Reference index for fixing floating rates.
#         """
#         return self.__reference_index


# class FixedToFloatingRateNoteSpecification(FixedRateBondSpecification, FloatingRateNoteSpecification):
#     def __init__(self,
#                  obj_id: str,
#                  issue_date: _Union[date, datetime],
#                  maturity_date: _Union[date, datetime],
#                  coupon_payment_dates: _List[_Union[date, datetime]],
#                  coupons: _List[float],
#                  coupon_period_dates: _List[_Union[date, datetime]],
#                  day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
#                  spreads: _List[float] = None,
#                  reference_index: str = 'dummy_curve',
#                  currency: str = 'EUR',
#                  notional: float = 100.0,
#                  issuer: str = None,
#                  securitisation_level: _Union[SecuritizationLevel, str] = None):
#         """
#         Fixed-to-floating rate note specification by providing fixed rate coupons and fixed rate coupon payment dates
#         as well as floating rate coupon periods directly.
#         """
#         # TODO FIX THIS CLASS!!!!!!!!!!!!!!!!
#         raise Exception('Not working properly, @Stefan: Please fix me!!!!')
#         FixedRateBondSpecification.__init__(self, obj_id, issue_date, maturity_date, coupon_payment_dates, coupons,
#                                currency, notional, issuer, securitisation_level)

#         FloatingRateNoteSpecification.__init__(self, obj_id, issue_date, maturity_date, coupon_period_dates,
#                                   day_count_convention, spreads, reference_index, currency, notional, issuer,
#                                   securitisation_level)

#     @classmethod
#     def from_master_data(cls, obj_id: str,
#                          issue_date: _Union[date, datetime],
#                          fixed_to_float_date: _Union[date, datetime],
#                          maturity_date: _Union[date, datetime],
#                          coupon: float,
#                          tenor_fixed: _Union[Period, str],
#                          tenor_float: _Union[Period, str],
#                          backwards_fixed: bool = True,
#                          backwards_float: bool = True,
#                          stub_fixed: bool = False,
#                          stub_float: bool = False,
#                          business_day_convention_fixed: _Union[RollConvention, str] = RollConvention.FOLLOWING,
#                          business_day_convention_float: _Union[RollConvention, str] = RollConvention.FOLLOWING,
#                          calendar_fixed: _Union[_HolidayBase, str] = None,
#                          calendar_float: _Union[_HolidayBase, str] = None,
#                          day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
#                          spread: float = 0.0,
#                          reference_index: str = 'dummy_curve',
#                          currency: _Union[str, int] = 'EUR',
#                          notional: float = 100.0,
#                          issuer: str = None,
#                          securitisation_level: _Union[SecuritizationLevel, str] = None):
#         """
#         Fixed-to-floating rate note specification based on master data.

#         Args:
#             # TODO: How can we avoid repeating ourselves here?
#             obj_id (str): (Preferably) Unique label of the bond, e.g. ISIN.
#             issue_date (_Union[date, datetime]): Date of bond issuance.
#             fixed_to_float_date (_Union[date, datetime]): Date where fixed schedule changes into floating one.
#             maturity_date (_Union[date, datetime]): Bond's maturity/expiry date. Must lie after the issue_date.
#             coupon (float): Annualised coupon amount as fraction of notional, e.g. 0.0125 for fixed rate coupon of
#                             1.25%.
#             tenor_fixed (_Union[period, str]): Time distance between two fixed rate coupon payment dates.
#             tenor_float (_Union[period, str]): Time distance between two floating rate coupon payment dates.
#             backwards_fixed (bool, optional): Defines direction for rolling out the schedule for the fixed rate part.
#                                               True means the schedule will be rolled out (backwards) from maturity date
#                                               to issue date. Defaults to True.
#             backwards_float (bool, optional): Defines direction for rolling out the schedule for the floating rate part.
#                                               True means the schedule will be rolled out (backwards) from maturity date
#                                               to issue date. Defaults to True.
#             stub_fixed (bool, optional): Defines if the first/last period is accepted (True) in the fixed rate schedule,
#                                          even though it is shorter than the others, or if it remaining days are added to
#                                          the neighbouring period (False). Defaults to True.
#             stub_float (bool, optional): Defines if the first/last period is accepted (True) in the float rate schedule,
#                                          even though it is shorter than the others, or if it remaining days are added to
#                                          the neighbouring period (False). Defaults to True.
#             business_day_convention_fixed (_Union[RollConvention, str], optional): Set of rules defining the adjustment
#                                                                                    of days to ensure each date in the
#                                                                                    fixed rate schedule being a business
#                                                                                    day with respect to a given holiday
#                                                                                    calendar. Defaults to
#                                                                                    RollConvention.FOLLOWING
#             business_day_convention_float (_Union[RollConvention, str], optional): Set of rules defining the adjustment
#                                                                                    of days to ensure each date in the
#                                                                                    float rate schedule being a business
#                                                                                    day with respect to a given holiday
#                                                                                    calendar. Defaults to
#                                                                                    RollConvention.FOLLOWING
#             calendar_fixed (_Union[__HolidayBase, str], optional): Holiday calendar defining the bank holidays of a
#                                                                   country or province (but not all non-business days as
#                                                                   for example Saturdays and Sundays).
#                                                                   Defaults (through constructor) to holidays.ECB
#                                                                   (= Target2 calendar) between start_day and end_day.
#             calendar_float (_Union[__HolidayBase, str], optional): Holiday calendar defining the bank holidays of a
#                                                                   country or province (but not all non-business days as
#                                                                   for example Saturdays and Sundays).
#                                                                   Defaults (through constructor) to holidays.ECB
#                                                                   (= Target2 calendar) between start_day and end_day.
#             day_count_convention (_Union[DayCounter, str], optional): Day count convention for determining period
#                                                                       length.Defaults to DayCounter.ThirtyU360.
#             spread (float, optional): Spread added to floating rate derived from fixing the reference curve as fraction
#                                       of notional, i.e. 0.0025 for 25 basis points. Defaults to 0.0.
#             reference_index (str, optional): Floating rate note underlying reference curve used for fixing the floating
#                                              rate coupon amounts. Defaults to 'dummy_curve'.
#                                              Note: A reference curve could also be provided later at the pricing stage.
#             currency (str, optional): Currency as alphabeticcode according to iso currency code
#                                                    ISO 4217 (cf. https://www.iso.org/iso-4217-currency-codes.html).
#                                                    Defaults to 'EUR'.
#             notional (float, optional): Bond's notional/face value. Must be positive. Defaults to 100.0.
#             issuer (str, optional): Issuer of the instrument. Defaults to None.
#             securitisation_level (_Union[SecuritizationLevel, str], optional): Securitisation level of the instrument.
#                                                                                Defaults to None.

#         Returns:
#             FixedToFloatingRateNote: Corresponding fixed-to-floating rate note with already generated schedules for
#                                      fixed rate and floating rate coupon payments.
#         """
#         fixed_rate_part = FixedRateBondSpecification.from_master_data(obj_id, issue_date, fixed_to_float_date, coupon, tenor_fixed,
#                                                          backwards_fixed, stub_fixed, business_day_convention_fixed,
#                                                          calendar_fixed, currency, notional, issuer,
#                                                          securitisation_level)
#         floating_rate_part = FloatingRateNoteSpecification.from_master_data(obj_id, fixed_to_float_date, maturity_date, tenor_float,
#                                                                backwards_float, stub_float,
#                                                                business_day_convention_float, calendar_float,
#                                                                day_count_convention, spread, reference_index, currency,
#                                                                notional, issuer, securitisation_level)
#         return FixedToFloatingRateNoteSpecification(obj_id, issue_date, maturity_date, fixed_rate_part.coupon_payment_dates,
#                                        fixed_rate_part.coupons, floating_rate_part.coupon_period_dates,
#                                        day_count_convention, floating_rate_part.spreads, reference_index, currency,
#                                        notional, issuer, securitisation_level)
