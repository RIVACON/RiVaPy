import abc
import calendar
from turtle import fd
from flask import logging
import numpy as np
from rivapy.instruments._logger import logger
from rivapy.instruments.components import (
    AmortizationScheme,
    ZeroAmortizationScheme,
    Issuer,
    LinearAmortizationScheme,
    LinearNotionalStructure,
    VariableAmortizationScheme,
)
from rivapy.marketdata.fixing_table import FixingTable
import rivapy.tools.interfaces as interfaces
from scipy.optimize import brentq

from collections import defaultdict
from typing import Optional, Dict, Tuple, List as _List, Union as _Union, Optional as _Optional
from dateutil.relativedelta import relativedelta
from rivapy.tools.enums import Currency, Rating, SecuritizationLevel, RollConvention, InterestRateIndex, get_index_by_alias
from rivapy.tools.datetools import _date_to_datetime, Schedule, Period, DayCounterType, DayCounter, _string_to_period
from rivapy.instruments.components import NotionalStructure, ConstNotionalStructure, VariableNotionalStructure  # , ResettingNotionalStructure
from rivapy.tools._validators import (
    _check_positivity,
    _check_start_before_end,
    _string_to_calendar,
    _check_start_at_or_before_end,
    _check_non_negativity,
    _is_ascending_date_list,
)
from datetime import datetime, date, timedelta
from rivapy.tools.datetools import (
    _term_to_period,
    calc_end_day,
    calc_start_day,
    roll_day,
    next_or_previous_business_day,
    is_business_day,
    RollRule,
    serialize_date,
)
from holidays import HolidayBase as _HolidayBase
from holidays import ECB as _ECB

# placeholder
from rivapy.marketdata.curves import DiscountCurve


class BondBaseSpecification(interfaces.FactoryObject):
    # ToDo: amend setter and property to handle float vs notional structure upon initialization, focus on Const and Linear, and variable with provided %-vector (as in FBG)
    # ToDo: amend setter and property to handle amortization scheme upon initialization
    # ToDo: adjust getCashFlows methods in derived classes accordingly
    def __init__(
        self,
        obj_id: str,
        issue_date: _Union[date, datetime],
        maturity_date: _Union[date, datetime],
        currency: _Union[Currency, str] = "EUR",
        notional: _Union[NotionalStructure, float] = 100.0,
        amortization_scheme: _Optional[_Union[str, AmortizationScheme]] = None,
        issuer: str = None,
        securitization_level: _Union[SecuritizationLevel, str] = "NONE",
        rating: _Union[Rating, str] = "NONE",
        day_count_convention: _Union[DayCounterType, str] = "ACT360",
        business_day_convention: _Union[RollConvention, str] = "ModifiedFollowing",
        roll_convention: _Union[RollRule, str] = "NONE",
        calendar: _Union[_HolidayBase, str] = _ECB(),
    ):
        """Base bond specification.

        Args:
            obj_id (str): (Preferably) Unique label of the bond, e.g. ISIN.
            issue_date (_Union[date, datetime]): Date of bond issuance.
            maturity_date (_Union[date, datetime]): Bond's maturity/expiry date. Must lie after the issue_date.
            currency (str, optional): Currency as alphabetic, Defaults to 'EUR'.
            notional (float, optional): Bond's notional/face value. Must be positive. Defaults to 100.0.
            issuer (str, optional): Name/id of issuer. Defaults to None.
            securitization_level (_Union[SecuritizationLevel, str], optional): Securitization level. Defaults to None.
            rating (_Union[Rating, str]): Paper rating.
        """
        self.obj_id = obj_id
        if issuer is not None:
            self._issuer = issuer
        else:
            self._issuer = "Unknown"
        if securitization_level is not None:
            self._securitization_level = securitization_level
        self._issue_date = issue_date
        self._maturity_date = maturity_date
        self._currency = currency
        self._amortization_scheme = self.set_amortization_scheme(amortization_scheme)
        # pass the resolved amortization scheme (object) to set_notional_structure
        self._notional = self.set_notional_structure(notional, self._amortization_scheme)
        self._rating = Rating.to_string(rating)
        self._day_count_convention = day_count_convention
        self._business_day_convention = business_day_convention
        self._roll_convention = roll_convention
        self._calendar = calendar
        # validate dates
        self._validate_derived_issued_instrument()

    def set_amortization_scheme(self, amortization_scheme) -> AmortizationScheme:
        if amortization_scheme is None:
            return ZeroAmortizationScheme()
        elif isinstance(amortization_scheme, str):
            return AmortizationScheme._from_string(amortization_scheme.lower())
        elif isinstance(amortization_scheme, AmortizationScheme):
            return amortization_scheme
        else:
            raise ValueError("Invalid amortization scheme provided.")

    def set_notional_structure(self, notional, amortization_scheme) -> NotionalStructure:
        if amortization_scheme is None:
            if isinstance(notional, _Union[int, float]):
                return ConstNotionalStructure(_check_positivity(notional))
            elif isinstance(notional, NotionalStructure):
                return notional
            raise ValueError("Invalid notional structure provided.")
        elif isinstance(amortization_scheme, ZeroAmortizationScheme):
            # accept ints and floats for numeric notional values
            if isinstance(notional, (_Union[int, float])):
                return ConstNotionalStructure(_check_positivity(notional))
            elif isinstance(notional, ConstNotionalStructure):
                return notional
            else:
                logger.warning("Amortization scheme is Const but notional is not ConstNotionalStructure. Using provided notional structure.")
                return notional
        elif isinstance(amortization_scheme, LinearAmortizationScheme):
            # accept ints and floats for numeric notional values
            if isinstance(notional, (_Union[int, float])):
                return LinearNotionalStructure(_check_positivity(notional))
            elif isinstance(notional, ConstNotionalStructure):
                logger.warning("Amortization scheme is Linear but notional is ConstNotionalStructure. Converting to LinearNotionalStructure.")
                return LinearNotionalStructure(notional.get_amount(0))
            elif isinstance(notional, (LinearNotionalStructure, VariableNotionalStructure)):
                logger.warning(
                    "Amortization scheme is Linear but notional is alredy LinearNotionalStructure or VariableNotionalStructure. Using provided notional structure."
                )
                return notional
        elif isinstance(amortization_scheme, VariableAmortizationScheme):
            logger.warning("Variable amortization scheme is not implemented. Retuning notional as is.")
            if isinstance(notional, (_Union[int, float])):
                return ConstNotionalStructure(_check_positivity(notional))
            else:
                return notional

    @staticmethod
    def _create_sample(
        n_samples: int, seed: int = None, ref_date=None, issuers: _List[str] = None, sec_levels: _List[str] = None, currencies: _List[str] = None
    ) -> _List[dict]:
        if seed is not None:
            np.random.seed(seed)
        if ref_date is None:
            ref_date = datetime.now()
        else:
            ref_date = _date_to_datetime(ref_date)
        if issuers is None:
            issuers = ["Issuer_" + str(i) for i in range(int(n_samples / 2))]
        result = []
        if currencies is None:
            currencies = list(Currency)
        if sec_levels is None:
            sec_levels = list(SecuritizationLevel)
        for _ in range(n_samples):
            days = int(15.0 * 365.0 * np.random.beta(2.0, 2.0)) + 1
            issue_date = ref_date + timedelta(days=np.random.randint(low=-365, high=0))
            result.append(
                {
                    "issue_date": issue_date,
                    "maturity_date": ref_date + timedelta(days=days),
                    "currency": np.random.choice(currencies),
                    "notional": np.random.choice([100.0, 1000.0, 10_000.0, 100_0000.0]),
                    "issuer": np.random.choice(issuers),
                    "securitization_level": np.random.choice(sec_levels),
                }
            )
        return result

    def _validate_derived_issued_instrument(self):
        self._issue_date, self._maturity_date = _check_start_before_end(self._issue_date, self._maturity_date)

    def _to_dict(self) -> dict:
        result = {
            "obj_id": self.obj_id,
            "issuer": self.issuer,
            "securitization_level": self.securitization_level,
            "issue_date": serialize_date(self.issue_date),
            "maturity_date": serialize_date(self.maturity_date),
            "currency": self.currency,
            "notional": self.notional,
            "rating": self.rating,
        }
        return result

    # region properties

    @property
    def issuer(self) -> str:
        """
        Getter for instrument's issuer.

        Returns:
            str: Instrument's issuer.
        """
        return self._issuer

    @issuer.setter
    def issuer(self, issuer: str):
        """
        Setter for instrument's issuer.

        Args:
            issuer(str): Issuer of the instrument.
        """
        self._issuer = issuer

    @property
    def rating(self) -> str:
        return self._rating

    @rating.setter
    def rating(self, rating: _Union[Rating, str]) -> str:
        self._rating = Rating.to_string(rating)

    @property
    def securitization_level(self) -> str:
        """
        Getter for instrument's securitisation level.

        Returns:
            str: Instrument's securitisation level.
        """
        return self._securitization_level.value

    @securitization_level.setter
    def securitization_level(self, securitisation_level: _Union[SecuritizationLevel, str]):
        self._securitization_level = SecuritizationLevel.to_string(securitisation_level)

    @property
    def issue_date(self) -> date:
        """
        Getter for bond's issue date.

        Returns:
            date: Bond's issue date.
        """
        return self._issue_date

    @issue_date.setter
    def issue_date(self, issue_date: _Union[datetime, date]):
        """
        Setter for bond's issue date.

        Args:
            issue_date (Union[datetime, date]): Bond's issue date.
        """
        self._issue_date = _date_to_datetime(issue_date)

    @property
    def maturity_date(self) -> date:
        """
        Getter for bond's maturity date.

        Returns:
            date: Bond's maturity date.
        """
        return self._maturity_date

    @maturity_date.setter
    def maturity_date(self, maturity_date: _Union[datetime, date]):
        """
        Setter for bond's maturity date.

        Args:
            maturity_date (Union[datetime, date]): Bond's maturity date.
        """
        self._maturity_date = _date_to_datetime(maturity_date)

    @property
    def currency(self) -> str:
        """
        Getter for bond's currency.

        Returns:
            str: Bond's ISO 4217 currency code
        """
        return self._currency

    @currency.setter
    def currency(self, currency: str):
        self._currency = Currency.to_string(currency)

    @property
    def notional(self) -> NotionalStructure:
        """
        Getter for bond's face value.

        Returns:
            float: Bond's face value.
        """
        return self._notional

    @notional.setter
    def notional(self, notional):
        if isinstance(notional, NotionalStructure):
            self._notional = notional
        else:
            self._notional = ConstNotionalStructure(_check_positivity(notional))

    @property
    def day_count_convention(self) -> str:
        """
        Getter for instruments's day count convention.

        Returns:
            str: instruments's day count convention.
        """
        return self._day_count_convention

    @day_count_convention.setter
    def day_count_convention(self, dcc: _Union[DayCounterType, str]):
        self._day_count_convention = DayCounterType.to_string(dcc)

    @property
    def business_day_convention(self) -> str:
        """
        Getter for FRA's day count convention.

        Returns:
            str: FRA's day count convention.
        """
        return self._business_day_convention

    @business_day_convention.setter
    def business_day_convention(self, business_day_convention: _Union[RollConvention, str]):
        self._business_day_convention = DayCounterType.to_string(business_day_convention)

    @property
    def roll_convention(self) -> str:
        """
        Getter for the roll convention used for business day adjustment.

        Returns:
            str: The roll convention used for business day adjustment.
        """
        return self._roll_convention

    @roll_convention.setter
    def roll_convention(self, roll_convention: _Union[RollRule, str]):
        """
        Setter for the roll convention used for business day adjustment.

        Args:
            roll_convention (_Union[RollRule, str]): The roll convention used for business day adjustment.
        """
        self._roll_convention = RollRule.to_string(roll_convention)

    @property
    def calendar(self):
        """
        Getter for the calendar used for business day adjustment.

        Returns:
            The calendar used for business day adjustment.
        """
        return self._calendar

    @calendar.setter
    def calendar(self, calendar: _Union[_HolidayBase, str]):
        """
        Setter for the calendar used for business day adjustment.

        Args:
            calendar (_Union[_HolidayBase, str]): The calendar used for business day adjustment.
        """
        if isinstance(calendar, str) and calendar.upper() == "TARGET":
            self._calendar = _ECB()
        else:
            self._calendar = _string_to_calendar(calendar)

    # endregion

    def notional_amount(self, index: _Union[date, datetime, int] = None) -> float:
        """Get the notional amount at a specific date.

        Args:
            index (_Union[date, datetime, int]): The index for which to get the notional amount, may be a date or an integer index. If None, returns the full notional structure.

        Returns:
            float: The notional amount at the specified index.
        """
        if index is not None:
            if isinstance(index, int):
                return self._notional.get_amount(index)
            else:
                return self._notional.get_amount_per_date(_date_to_datetime(index))
        else:
            return self._notional.get_amount(index)


class DeterministicCashflowBondSpecification(BondBaseSpecification):

    def __init__(
        self,
        obj_id: str,
        issue_date: _Union[date, datetime],
        start_date: _Union[date, datetime],
        end_date: _Union[date, datetime],
        maturity_date: _Union[date, datetime],
        notional: _Union[NotionalStructure, float] = 100.0,
        frequency: _Optional[_Union[Period, str]] = None,
        first_fixing_date: _Optional[_Union[date, datetime]] = None,
        issue_price: _Optional[float] = None,
        ir_index: _Union[InterestRateIndex, str] = None,
        index: _Optional[_Union[InterestRateIndex, str]] = None,
        currency: _Union[Currency, str] = Currency.EUR,
        notional_exchange: bool = True,
        coupon: float = 0.0,
        margin: float = 0.0,
        amortization_scheme: _Optional[_Union[str, AmortizationScheme]] = None,
        day_count_convention: _Union[DayCounterType, str] = "ACT360",
        business_day_convention: _Union[RollConvention, str] = "ModifiedFollowing",
        roll_convention: _Union[RollRule, str] = "NONE",
        calendar: _Union[_HolidayBase, str] = _ECB(),
        coupon_type: str = "fix",
        payment_days: int = 0,
        spot_days: int = 2,
        pays_in_arrears: bool = True,
        issuer: _Optional[_Union[Issuer, str]] = None,
        rating: _Union[Rating, str] = Rating.NONE,
        securitization_level: _Union[SecuritizationLevel, str] = SecuritizationLevel.NONE,
        backwards=True,
        stub_type_is_Long=True,
        last_fixing: _Optional[float] = None,
        fixings: _Optional[FixingTable] = None,
    ):
        """Initializes the DeterministicCashflowBondSpecification object.

        Args:
            obj_id (str): Unique identifier for the object.
            first_fixing_date (_Union[date, datetime]): Date of the first fixing.
            start_date (_Union[date, datetime]): Start date of the first accrual period.
            end_date (_Union[date, datetime]): End of the last accrual period. Not necessarily a good business day.
            maturity_date (_Union[date, datetime]): Adjusted end date of the last accrual period. Is a good business day.
            notional (float): Notional amount of the instrument.
            coupon (float): Fixed coupon rate .
            margin (float): Spread added to the floating rate coupon.
            tenor (_Union[Period, str]): Tenor of the underlying floating index of the instrument.
            frequency (_Union[Period, str]): Payment frequency of the instrument.
            day_count_convention (_Union[DayCounterType, str], optional): Day count convention. Defaults to DayCounterType.ACT360.
            business_day_convention (_Union[RollConvention, str], optional): Business day convention. Defaults to RollConvention.MODIFIED_FOLLOWING.
            roll_convention (_Union[RollRule, str], optional): Roll convention. Defaults to RollRule.EOM.
            calendar (_Union[_HolidayBase, str], optional): Holiday calendar. Defaults to _ECB().
            payment_days (int, optional): Number of payment days that pass between accrual end or maturity to payment. Defaults to 0.
            notional_exchange (bool, optional): Indicates if notional is exchanged at maturity. Defaults
            pays_in_arrears (bool, optional): Indicates if the instrument pays in arrears. Defaults to True.
            fwd_curve (_Optional[DiscountCurve], optional): Forward curve used for pricing. Defaults to None.
            last_fixing (_Optional[float], optional): Last known fixing rate. Defaults to None.
            fixings (_Optional[FixingTable], optional): Fixing table containing historical fixings. Defaults to None.
        """
        super().__init__(
            obj_id,
            issue_date,
            _date_to_datetime(maturity_date),
            Currency.to_string(currency),
            notional,
            amortization_scheme,
            issuer,
            securitization_level,
            Rating.to_string(rating),
            day_count_convention,
            business_day_convention,
            roll_convention,
            calendar,
        )
        if first_fixing_date is not None:
            self._first_fixing_date = _date_to_datetime(first_fixing_date)
        else:
            self._first_fixing_date = _date_to_datetime(start_date)
        self._start_date = _date_to_datetime(start_date)
        self._end_date = _date_to_datetime(end_date)
        if issue_price is not None:
            self._issue_price = _check_non_negativity(issue_price)
        else:
            self._issue_price = None
        self._coupon = coupon
        self._margin = margin
        self._frequency = frequency
        self._ir_index = ir_index
        self._index = index
        self._coupon_type = coupon_type
        self._notional_exchange = notional_exchange
        self._payment_days = payment_days
        self._spot_days = spot_days
        self._pays_in_arrears = pays_in_arrears
        self._backwards = backwards
        self._stub_type_is_Long = stub_type_is_Long
        self._validate()
        self._last_fixing = last_fixing
        self._fixings = fixings
        self._schedule = None
        self._nr_annual_payments = None
        self._dates = None

    # region properties

    @property
    def coupon(self) -> float:
        """
        Getter for instrument's coupon.

        Returns:
            float: Instrument's coupon.
        """
        return self._coupon

    @coupon.setter
    def coupon(self, rate: float):
        """
        Setter for instrument's rate.

        Args:
            rate(float): interest rate of the instrument.
        """
        self._rate = rate

    @property
    def start_date(self) -> datetime.date:
        """
        Getter for deposit's start date.

        Returns:
            date: deposit's start date.
        """
        return self._start_date

    @start_date.setter
    def start_date(self, start_date: _Union[date, datetime]):
        """
        Setter for deposit's start date.

        Args:
            start_date (Union[datetime, date]): deposit's start date.
        """
        self._start_date = _date_to_datetime(start_date)

    @property
    def end_date(self) -> datetime:
        """
        Getter for deposit's end date.

        Returns:
            date: deposit's end date.
        """
        return self._end_date

    @end_date.setter
    def end_date(self, end_date: _Union[date, datetime]):
        """
        Setter for deposit's end date.

        Args:
            end_date (Union[datetime, date]): deposit's end date.
        """
        if not isinstance(end_date, (date, datetime)):
            raise TypeError("end_date must be a datetime or date object.")
        self._end_date = _date_to_datetime(end_date)

    @property
    def frequency(self) -> Period:
        """
        Getter for instrument's payment frequency.

        Returns:
            Period: instrument's payment frequency.
        """
        return self._frequency

    @frequency.setter
    def frequency(self, frequency: _Union[Period, str]):
        """
        Setter for instrument's payment frequency.

        Args:
            frequency (Union[Period, str]): instrument's payment frequency.
        """
        self._frequency = _term_to_period(frequency)

    @property
    def issue_price(self) -> _Optional[float]:
        """The bond's issue price as a float."""
        return getattr(self, "_issue_price", None)

    @issue_price.setter
    def issue_price(self, issue_price: _Union[float, str]):
        self._issue_price = _check_non_negativity(issue_price)

    @property
    def notional_exchange(self):
        return self._notional_exchange

    @notional_exchange.setter
    def notional_exchange(self, notional_exchange: bool):
        self._notional_exchange = notional_exchange

    @property
    def payment_days(self) -> int:
        """
        Getter for the number of payment days.

        Returns:
            int: Number of payment days.
        """
        return self._payment_days

    @payment_days.setter
    def payment_days(self, payment_days: int):
        """
        Setter for the number of payment days.

        Args:
            payment_days (int): Number of payment days.
        """
        if not isinstance(payment_days, int) or payment_days < 0:
            raise ValueError("payment days must be a non-negative integer.")
        self._payment_days = payment_days

    @property
    def pays_in_arrears(self) -> bool:
        """
        Getter for the pays_in_arrears flag.

        Returns:
            bool: True if the instrument pays in arrears, False otherwise.
        """
        return self._pays_in_arrears

    @pays_in_arrears.setter
    def pays_in_arrears(self, pays_in_arrears: bool):
        """
        Setter for the pays_in_arrears flag.

        Args:
            pays_in_arrears (bool): True if the instrument pays in arrears, False otherwise.
        """
        if not isinstance(pays_in_arrears, bool):
            raise ValueError("pays_in_arrears must be a boolean value.")
        self._pays_in_arrears = pays_in_arrears

    @property
    def coupon_type(self) -> str:
        """
        Getter for the coupon type of the instrument.

        Returns:
            str: The coupon type of the instrument.
        """
        return self._coupon_type

    @coupon_type.setter
    def coupon_type(self, coupon_type: str):
        """
        Setter for the coupon type of the instrument.

        Args:
            coupon_type (str): The coupon type of the instrument.
        """
        if not isinstance(coupon_type, str):
            raise ValueError("Coupon type must be a string.")
        self._coupon_type = coupon_type

    @property
    def first_fixing_date(self) -> datetime:
        """
        Getter for the first fixing date of the instrument.

        Returns:
            datetime: The first fixing date.
        """
        return self._first_fixing_date

    @first_fixing_date.setter
    def first_fixing_date(self, first_fixing_date: _Union[date, datetime]):
        """
        Setter for the first fixing date of the instrument.

        Args:
            first_fixing_date (_Union[date,datetime]): The first fixing date.
        """
        self._first_fixing_date = _date_to_datetime(first_fixing_date)

    @property
    def backwards(self) -> bool:
        """
        Getter for the backwards flag.

        Returns:
            bool: True if the schedule is generated backwards, False otherwise.
        """
        return self._backwards

    @backwards.setter
    def backwards(self, backwards: bool):
        """
        Setter for the backwards flag.

        Args:
            backwards (bool): True if the schedule is generated backwards, False otherwise.
        """
        if not isinstance(backwards, bool):
            raise ValueError("Backwards must be a boolean value.")
        self._backwards = backwards

    @property
    def stub_type_is_Long(self) -> bool:
        """
        Getter for the stub type flag.

        Returns:
            bool: True if the stub type is long, False otherwise.
        """
        return self._stub_type_is_Long

    @stub_type_is_Long.setter
    def stub_type_is_Long(self, stub_type_is_Long: bool):
        """
        Setter for the stub type flag.

        Args:
            stub_type_is_Long (bool): True if the stub type is long, False otherwise.
        """
        if not isinstance(stub_type_is_Long, bool):
            raise ValueError("Stub type must be a boolean value.")
        self._stub_type_is_Long = stub_type_is_Long

    @property
    def spot_days(self) -> int:
        """
        Getter for the number of spot days.

        Returns:
            int: Number of spot days.
        """
        return self._spot_days

    @spot_days.setter
    def spot_days(self, spot_days: int):
        """
        Setter for the number of spot days.

        Args:
            spot_days (int): Number of spot days.
        """
        if not isinstance(spot_days, int) or spot_days < 0:
            raise ValueError("Spot days must be a non-negative integer.")
        self._spot_days = spot_days

    @property
    def last_fixing(self) -> _Optional[float]:
        """
        Getter for the last fixing value.

        Returns:
            _Optional[float]: The last fixing value, or None if not set.
        """
        return self._last_fixing

    @last_fixing.setter
    def last_fixing(self, last_fixing: _Optional[float]):
        """
        Setter for the last fixing value.

        Args:
            last_fixing (_Optional[float]): The last fixing value, or None if not set.
        """
        if last_fixing is not None and not isinstance(last_fixing, (float, int)):
            raise ValueError("Last fixing must be a float or None.")
        self._last_fixing = float(last_fixing) if last_fixing is not None else None

    @property
    def nr_annual_payments(self) -> _Optional[float]:
        """
        Getter for the number of annual payments.

        Returns:
            _Optional[float]: The number of annual payments, or None if frequency is not set.
        """
        if self._nr_annual_payments is None:
            self._nr_annual_payments = self.get_nr_annual_payments()
        return self._nr_annual_payments

    @nr_annual_payments.setter
    def nr_annual_payments(self, value: _Optional[float]):
        """
        Setter for the number of annual payments.

        Args:
            value (_Optional[float]): The number of annual payments, or None if frequency is not set.
        """
        if value is not None and (not isinstance(value, (float, int)) or value <= 0):
            raise ValueError("Number of annual payments must be a positive float or None.")
        self._nr_annual_payments = float(value) if value is not None else None

    @property
    def schedule(self) -> Schedule:
        """
        Getter for the dates of the instrument.

        Returns:
            Schedule: The schedule of the instrument.
        """
        return self.get_schedule()

    @schedule.setter
    def schedule(self, schedule: Schedule):
        """
        Setter for the dates of the instrument.

        Args:
            schedule (Schedule): The schedule of the instrument.
        """
        if not isinstance(schedule, Schedule):
            raise ValueError("Schedule must be a Schedule object.")
        self._schedule = schedule

    @property
    def dates(self) -> _List[datetime]:
        """
        Getter for the dates of the instrument that mark start and end dates of the accrual periods.

        Returns:
            _List[datetime]: The dates of the instrument.
        """
        if self._dates is None:
            # Try to get schedule, fallback to empty list if not possible
            try:
                schedule = self._schedule if self._schedule is not None else self.get_schedule()
                if schedule is not None:
                    self._dates = schedule._roll_out(
                        from_=self._start_date if not self._backwards else self._end_date,
                        to_=self._end_date if not self._backwards else self._start_date,
                        term=_term_to_period(self._frequency),
                        long_stub=self._stub_type_is_Long,
                        backwards=self._backwards,
                    )
                    if isinstance(self._notional, LinearNotionalStructure):
                        self._notional.n_steps = len(self._dates)
                        self._notional._notional = list(
                            np.linspace(self._notional.start_notional, self._notional.end_notional, self._notional.n_steps)
                        )
                        self._notional.start_date = self._dates[:-1]
                        self._notional.end_date = self._dates[1:]
                else:
                    self._dates = []
            except Exception as e:
                # Optionally log the error here
                self._dates = []
        return self._dates if self._dates is not None else []

    @dates.setter
    def dates(self, dates: _List[datetime]):
        """
        Setter for the dates of the instrument that mark start and end dates of the accrual periods.

        Args:
            dates (_List[datetime]): The dates of the instrument.
        """
        if not _is_ascending_date_list(dates):
            raise ValueError("Dates must be a list of ascending datetime objects.")
        self._dates = dates

    @property
    def index(self) -> float:
        """
        Getter for instrument's index.

        Returns:
            float: Instrument's index.
        """
        return self._index

    @index.setter
    def index(self, index: _Union[InterestRateIndex, str]):
        """
        Setter for instrument's index.

        Args:
            index (_Union[InterestRateIndex, str]): instrument's index.
        """
        self._index = index
        self._ir_index = index if isinstance(index, InterestRateIndex) else get_index_by_alias(index)
        self._frequency = self._ir_index.value.tenor

    @property
    def ir_index(self) -> InterestRateIndex:
        """
        Getter for instrument's interest rate index.

        Returns:
            InterestRateIndex: Instrument's interest rate index.
        """
        return self._ir_index

    @ir_index.setter
    def ir_index(self, ir_index: InterestRateIndex):
        """
        Setter for instrument's interest rate index.

        Args:
            ir_index (InterestRateIndex): Instrument's interest rate index.
        """
        self._ir_index = ir_index

    # endregion

    def _validate(self):
        """Validates the parameters of the instrument."""
        _check_start_at_or_before_end(self._first_fixing_date, self._start_date)
        _check_start_before_end(self._start_date, self._end_date)
        # _check_start_at_or_before_end(self._end_date, self._maturity_date) # TODO special case modified following BCC
        _check_non_negativity(self._payment_days)
        _check_non_negativity(self._spot_days)
        if not isinstance(self._calendar, (_HolidayBase, str)):
            raise ValueError("Calendar must be a HolidayBase or string.")

    def get_schedule(self) -> Schedule:
        """Returns the schedule of the accrual periods of the instrument."""
        return Schedule(
            start_day=self._start_date,
            end_day=self._end_date,
            time_period=_string_to_period(self._frequency),
            backwards=self._backwards,
            stub_type_is_Long=self._stub_type_is_Long,
            business_day_convention=self._business_day_convention,
            roll_convention=self._roll_convention,
            calendar=self._calendar,
        )

    def get_nr_annual_payments(self) -> float:
        """Returns the number of annual payments of the instrument."""
        if self._frequency is None:
            logger.warning("Frequency is not set. Returning 0.")
            return 0.0
        freq = _string_to_period(self._frequency)
        if freq.years > 0 or freq.months > 0 or freq.days > 0:
            nr = 12.0 / (freq.years * 12 + freq.months + freq.days * 12 / 365.0)
        else:
            raise ValueError("Frequency must be positive.")
        if nr.is_integer() is False:
            logger.warning("Number of annual payments is not a whole number but a decimal.")
        return nr

    @abc.abstractmethod
    def _to_dict(self) -> dict:
        pass


class FixedRateBondSpecification(DeterministicCashflowBondSpecification):

    def __init__(
        self,
        obj_id: str,
        notional: _Union[NotionalStructure, float],
        currency: _Union[Currency, str],
        issue_date: _Union[date, datetime],
        maturity_date: _Union[date, datetime],
        coupon: float,
        frequency: _Union[Period, str],
        amortization_scheme: _Optional[_Union[str, AmortizationScheme]] = None,
        business_day_convention: RollConvention = "ModifiedFollowing",
        issuer: Optional[_Union[Issuer, str]] = None,
        securitization_level: Optional[_Union[SecuritizationLevel, str]] = "NONE",
        rating: Optional[_Union[Rating, str]] = "NONE",
        day_count_convention: _Union[DayCounterType, str] = "ActActICMA",
        spot_days: int = 2,
        calendar: Optional[_Union[_HolidayBase, str]] = _ECB(),
        stub_type_is_Long: bool = True,
        adjust_start_date: bool = True,
        adjust_end_date: bool = False,
    ):
        if not is_business_day(issue_date, calendar) and adjust_start_date:
            start_date = roll_day(issue_date, calendar=calendar, business_day_convention=business_day_convention)
        else:
            start_date = issue_date
        if not is_business_day(maturity_date, calendar) and adjust_end_date:
            end_date = roll_day(maturity_date, calendar=calendar, business_day_convention=business_day_convention)
        else:
            end_date = maturity_date
        if not is_business_day(maturity_date, calendar):
            maturity_date = roll_day(maturity_date, calendar=calendar, business_day_convention=business_day_convention)
        super().__init__(
            obj_id=obj_id,
            spot_days=spot_days,
            issue_date=issue_date,
            start_date=start_date,
            end_date=end_date,
            maturity_date=maturity_date,
            notional=notional,
            amortization_scheme=amortization_scheme,
            currency=currency,
            coupon=coupon,
            coupon_type="fix",
            frequency=frequency,
            day_count_convention=day_count_convention,
            business_day_convention=business_day_convention,
            payment_days=0,
            stub_type_is_Long=stub_type_is_Long,
            issuer=issuer,
            rating=rating,
            securitization_level=securitization_level,
            calendar=calendar,
        )
        self._adjust_start_date = adjust_start_date
        self._adjust_end_date = adjust_end_date

    @property
    def adjust_start_date(self) -> bool:
        return self._adjust_start_date

    @adjust_start_date.setter
    def adjust_start_date(self, value: bool):
        self._adjust_start_date = value
        if not is_business_day(self._issue_date, self._calendar) and self._adjust_start_date:
            self._start_date = roll_day(self._issuedate, calendar=self._calendar, business_day_convention=self._business_day_convention)

    @property
    def adjust_end_date(self) -> bool:
        return self._adjust_end_date

    @adjust_end_date.setter
    def adjust_end_date(self, value: bool):
        self._adjust_end_date = value
        if not is_business_day(self._maturity_date, self._calendar) and self._adjust_end_date:
            self._end_date = roll_day(self._maturity_date, calendar=self._calendar, business_day_convention=self._business_day_convention)

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
            coupon = np.random.choice([0.0, 0.01, 0.03, 0.05])
            period = np.random.choice(["1Y", "6M", "3M"])
            result.append(
                FixedRateBondSpecification(
                    obj_id=f"ID_{i}",
                    notional=notional,
                    frequency=period,
                    currency=currency,
                    issue_date=issue_date,
                    maturity_date=maturity_date,
                    coupon=coupon,
                    securitization_level=securitization_level,
                    day_count_convention=daycounter,
                )
            )
        return result

    def _to_dict(self) -> Dict:
        dict = {
            "obj_id": self.obj_id,
            "issuer": self._issuer,
            "securitization_level": self._securitization_level,
            "issue_date": serialize_date(self._issue_date),
            "maturity_date": serialize_date(self._maturity_date),
            "currency": self._currency,
            "notional": self._notional,
            "rating": self._rating,
            "frequency": self._frequency,
            "day_count_convention": self._day_count_convention,
            "business_day_convention": self._business_day_convention,
            "coupon": self._coupon,
            "spot_days": self._spot_days,
            "calendar": getattr(self._calendar, "name", self._calendar.__class__.__name__),
            "adjust_start_date": self._adjust_start_date,
            "adjust_end_date": self._adjust_end_date,
        }
        return dict


class ZeroBondSpecification(DeterministicCashflowBondSpecification):

    def __init__(
        self,
        obj_id: str,
        notional: float,
        currency: _Union[Currency, str],
        issue_date: _Union[date, datetime],
        maturity_date: _Union[date, datetime],
        amortization_scheme: _Optional[_Union[str, AmortizationScheme]] = None,
        issue_price: float = 100.0,
        calendar: Optional[_Union[_HolidayBase, str]] = _ECB(),
        business_day_convention: RollConvention = "ModifiedFollowing",
        issuer: Optional[_Union[Issuer, str]] = None,
        securitization_level: Optional[_Union[SecuritizationLevel, str]] = "NONE",
        rating: Optional[_Union[Rating, str]] = "NONE",
    ):
        if not is_business_day(maturity_date, calendar):
            maturity_date = roll_day(maturity_date, calendar=calendar, business_day_convention=business_day_convention)
        super().__init__(
            obj_id=obj_id,
            issue_date=issue_date,
            start_date=issue_date,
            end_date=maturity_date,
            maturity_date=maturity_date,
            notional=notional,
            amortization_scheme=amortization_scheme,
            issue_price=issue_price,
            currency=currency,
            business_day_convention=business_day_convention,
            coupon_type="zero",
            issuer=issuer,
            rating=rating,
            securitization_level=securitization_level,
            calendar=calendar,
        )

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
        for i in range(n_samples):
            issue_price = np.random.choice([90.0, 95.0, 99.0])
            m = np.random.choice([0, 12, 6, 3])
            result.append(
                ZeroBondSpecification(
                    obj_id=f"ID_{i}",
                    notional=notional,
                    issue_price=issue_price,
                    currency=currency,
                    issue_date=issue_date,
                    maturity_date=maturity_date + relativedelta(months=m),
                    securitization_level=securitization_level,
                )
            )
        return result

    def _to_dict(self) -> Dict:
        dict = {
            "obj_id": self.obj_id,
            "issuer": self._issuer,
            "securitization_level": self._securitization_level,
            "issue_date": serialize_date(self._issue_date),
            "maturity_date": serialize_date(self._maturity_date),
            "currency": self._currency,
            "notional": self._notional,
            "issue_price": self._issue_price,
            "rating": self._rating,
            "business_day_convention": self._business_day_convention,
            "calendar": getattr(self._calendar, "name", self._calendar.__class__.__name__),
        }
        return dict


class FloatingRateBondSpecification(DeterministicCashflowBondSpecification):

    def __init__(
        self,
        obj_id: str,
        notional: _Union[NotionalStructure, float],
        currency: _Union[Currency, str],
        issue_date: _Union[date, datetime],
        maturity_date: _Union[date, datetime],
        margin: float,
        frequency: Optional[_Union[Period, str]] = None,
        amortization_scheme: _Optional[_Union[str, AmortizationScheme]] = None,
        index: Optional[_Union[InterestRateIndex, str]] = None,
        business_day_convention: RollConvention = "ModifiedFollowing",
        issuer: Optional[_Union[Issuer, str]] = None,
        securitization_level: Optional[_Union[SecuritizationLevel, str]] = "NONE",
        rating: Optional[_Union[Rating, str]] = "NONE",
        day_count_convention: DayCounterType = "ActActICMA",
        fixings: Optional[FixingTable] = None,
        spot_days: int = 2,
        calendar: Optional[_Union[_HolidayBase, str]] = _ECB(),
        stub_type_is_Long: bool = True,
        adjust_start_date: bool = True,
        adjust_end_date: bool = False,
    ):
        if not is_business_day(issue_date, calendar) and adjust_start_date:
            start_date = roll_day(issue_date, calendar=calendar, business_day_convention=business_day_convention)
        else:
            start_date = issue_date
        first_fixing_date = calc_start_day(start_date, f"{spot_days}D", business_day_convention=business_day_convention, calendar=calendar)
        if not is_business_day(maturity_date, calendar) and adjust_end_date:
            end_date = roll_day(maturity_date, calendar=calendar, business_day_convention=business_day_convention)
        else:
            end_date = maturity_date
        if not is_business_day(maturity_date, calendar):
            maturity_date = roll_day(maturity_date, calendar=calendar, business_day_convention=business_day_convention)
        if index is None and frequency is None:
            raise ValueError("Either index or frequency must be provided for a floating rate bond.")
        elif index is not None:
            if isinstance(index, str):
                # get_index_by_alias will raise if alias unknown
                self._ir_index = get_index_by_alias(index)
            else:
                self._ir_index = index
            # derive frequency from the index tenor if available
            try:
                frequency = self._ir_index.value.tenor
            except Exception:
                # leave frequency as provided if index doesn't expose tenor
                frequency = frequency
        else:
            # no index info given, rely on provided frequency
            frequency = frequency
            self._ir_index = None
        super().__init__(
            obj_id=obj_id,
            fixings=fixings,
            spot_days=spot_days,
            issue_date=issue_date,
            start_date=start_date,
            end_date=end_date,
            maturity_date=maturity_date,
            notional=notional,
            amortization_scheme=amortization_scheme,
            currency=currency,
            margin=margin,
            coupon_type="float",
            frequency=frequency,
            index=index,
            ir_index=self._ir_index,
            day_count_convention=day_count_convention,
            business_day_convention=business_day_convention,
            notional_exchange=True,
            payment_days=0,
            stub_type_is_Long=stub_type_is_Long,
            issuer=issuer,
            rating=rating,
            securitization_level=securitization_level,
        )
        self._adjust_start_date = adjust_start_date
        self._adjust_end_date = adjust_end_date

    @property
    def adjust_start_date(self) -> bool:
        return self._adjust_start_date

    @adjust_start_date.setter
    def adjust_start_date(self, value: bool):
        self._adjust_start_date = value
        if not is_business_day(self._issue_date, self._calendar) and self._adjust_start_date:
            self._start_date = roll_day(self._issuedate, calendar=self._calendar, business_day_convention=self._business_day_convention)

    @property
    def adjust_end_date(self) -> bool:
        return self._adjust_end_date

    @adjust_end_date.setter
    def adjust_end_date(self, value: bool):
        self._adjust_end_date = value
        if not is_business_day(self._maturity_date, self._calendar) and self._adjust_end_date:
            self._end_date = roll_day(self._maturity_date, calendar=self._calendar, business_day_convention=self._business_day_convention)

    @staticmethod
    def _create_sample(n_samples: int, seed: int = None):
        result = []
        if seed is not None:
            np.random.seed(seed)

        issue_date = datetime(2025, 1, 1)
        maturity_date = datetime(2027, 1, 1)
        notional = 100.0
        currency = Currency.EUR
        fixings = FixingTable()
        securitization_level = SecuritizationLevel.SUBORDINATED
        daycounter = "ACT_ACT"
        for i in range(n_samples):
            margin = np.random.choice([0.0, 1, 3, 5])
            period = np.random.choice(["1Y", "6M", "3M"])
            result.append(
                FloatingRateBondSpecification(
                    obj_id=f"ID_{i}",
                    notional=notional,
                    frequency=period,
                    currency=currency,
                    issue_date=issue_date,
                    maturity_date=maturity_date,
                    margin=margin,
                    securitization_level=securitization_level,
                    day_count_convention=daycounter,
                    fixings=fixings,
                )
            )
        return result

    def _to_dict(self) -> Dict:
        dict = {
            "obj_id": self.obj_id,
            "issuer": self._issuer,
            "securitization_level": self._securitization_level,
            "issue_date": serialize_date(self._issue_date),
            "maturity_date": serialize_date(self._maturity_date),
            "currency": self._currency,
            "notional": self._notional,
            "rating": self._rating,
            "frequency": self._frequency,
            "day_count_convention": self._day_count_convention,
            "business_day_convention": self._business_day_convention,
            "fixings": self._fixings._to_dict() if isinstance(self._fixings, FixingTable) else self._fixings,
            "ir_index": self._ir_index,
            "index": self._index,
            "margin": self._margin,
            "spot_days": self._spot_days,
            "calendar": getattr(self._calendar, "name", self._calendar.__class__.__name__),
            "adjust_start_date": self._adjust_start_date,
            "adjust_end_date": self._adjust_end_date,
        }
        return dict


def bonds_main():
    # zero coupon bond
    zero_coupon_bond = ZeroBondSpecification(
        obj_id="US500769CH58",
        issue_price=85.0,
        issue_date=datetime(2007, 6, 29),
        maturity_date=datetime(2037, 6, 29),
        currency="USD",
        notional=1000,
        issuer="KfW",
        securitization_level=SecuritizationLevel.SENIOR_UNSECURED,
    )
    # print("Zero Coupon Bond Specification:")
    # print(zero_coupon_bond._to_dict())
    # print(zero_coupon_bond.notional_amount())


if __name__ == "__main__":
    bonds_main()
