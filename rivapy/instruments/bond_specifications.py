import abc
import calendar
from turtle import fd
from flask import logging
import numpy as np
from rivapy.instruments._logger import logger
from rivapy.instruments.components import Issuer
from rivapy.marketdata.fixing_table import FixingTable
import rivapy.tools.interfaces as interfaces
from scipy.optimize import brentq

from collections import defaultdict
from typing import Optional, Dict, Tuple, List as _List, Union as _Union, Optional as _Optional
from dateutil.relativedelta import relativedelta
from rivapy.tools.enums import Currency, Rating, SecuritizationLevel, RollConvention, InterestRateIndex, get_index_by_alias
from rivapy.tools.datetools import _date_to_datetime, Schedule, Period, DayCounterType, DayCounter, _string_to_period
from rivapy.instruments.components import NotionalStructure, ConstNotionalStructure, VariableNotionalStructure, ResettingNotionalStructure
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

    def __init__(
        self,
        obj_id: str,
        issue_date: _Union[date, datetime],
        maturity_date: _Union[date, datetime],
        currency: _Union[Currency, str] = "EUR",
        notional: _Union[NotionalStructure, float] = 100.0,
        issuer: str = None,
        securitization_level: _Union[SecuritizationLevel, str] = SecuritizationLevel.NONE,
        rating: _Union[Rating, str] = Rating.NONE,
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
        self._notional = notional
        self._rating = Rating.to_string(rating)
        # validate dates
        self._validate_derived_issued_instrument()

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
    def notional(self) -> float:
        """
        Getter for bond's face value.

        Returns:
            float: Bond's face value.
        """
        return self._notional

    @notional.setter
    def notional(self, notional):
        self._notional = _check_positivity(notional)

    # endregion


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
        index_alias: _Union[InterestRateIndex, str] = None,
        index: _Optional[InterestRateIndex] = None,
        currency: _Union[Currency, str] = Currency.EUR,
        notional_exchange: bool = True,
        coupon: float = 0.0,
        margin: float = 0.0,
        day_count_convention: _Union[DayCounterType, str] = DayCounterType.ACT360,
        business_day_convention: _Union[RollConvention, str] = RollConvention.MODIFIED_FOLLOWING,
        roll_convention: _Union[RollRule, str] = RollRule.NONE,
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
            issuer,
            securitization_level,
            Rating.to_string(rating),
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
        self._index_alias = index_alias
        if index is not None:
            self._index = index
        else:
            self._index = None
        self._day_count_convention = day_count_convention
        self._business_day_convention = business_day_convention
        self._roll_convention = roll_convention
        self._calendar = calendar
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

    def _validate(self):
        """Validates the parameters of the instrument."""
        _check_positivity(self._notional)
        _check_start_at_or_before_end(self._first_fixing_date, self._start_date)
        _check_start_before_end(self._start_date, self._end_date)
        # _check_start_at_or_before_end(self._end_date, self._maturity_date) # TODO special case modified following BCC
        _check_non_negativity(self._payment_days)
        _check_non_negativity(self._spot_days)
        # if not isinstance(self._frequency, (Period, str)):
        #     raise ValueError("Frequency must be a Period object or string.")
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
        notional: float,
        currency: _Union[Currency, str],
        issue_date: _Union[date, datetime],
        maturity_date: _Union[date, datetime],
        coupon: float,
        frequency: _Union[Period, str],
        business_day_convention: RollConvention = RollConvention.MODIFIED_FOLLOWING,
        issuer: Optional[_Union[Issuer, str]] = None,
        securitization_level: Optional[_Union[SecuritizationLevel, str]] = SecuritizationLevel.NONE,
        rating: Optional[_Union[Rating, str]] = Rating.NONE,
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
        issue_price: float = 100.0,
        calendar: Optional[_Union[_HolidayBase, str]] = _ECB(),
        business_day_convention: RollConvention = RollConvention.MODIFIED_FOLLOWING,
        issuer: Optional[_Union[Issuer, str]] = None,
        securitization_level: Optional[_Union[SecuritizationLevel, str]] = SecuritizationLevel.NONE,
        rating: Optional[_Union[Rating, str]] = Rating.NONE,
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
        index_alias: Optional[str] = None,
        index: Optional[InterestRateIndex] = None,
        business_day_convention: RollConvention = RollConvention.MODIFIED_FOLLOWING,
        issuer: Optional[_Union[Issuer, str]] = None,
        securitization_level: Optional[_Union[SecuritizationLevel, str]] = SecuritizationLevel.NONE,
        rating: Optional[_Union[Rating, str]] = Rating.NONE,
        day_count_convention: DayCounterType = DayCounterType.ActActICMA,
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
        if index_alias is None and frequency is None:
            raise ValueError("Either index or frequency must be provided for a floating rate bond.")
        elif index_alias is not None:
            self._index_alias = index_alias
            index = get_index_by_alias(index_alias)
            frequency = index.value.tenor
        else:
            frequency = frequency
        super().__init__(
            obj_id=obj_id,
            fixings=fixings,
            spot_days=spot_days,
            issue_date=issue_date,
            start_date=start_date,
            end_date=end_date,
            maturity_date=maturity_date,
            notional=notional,
            currency=currency,
            margin=margin,
            coupon_type="float",
            frequency=frequency,
            index_alias=index_alias,
            index=index,
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
        daycounter = DayCounterType.ACT_ACT
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
            "index_alias": self._index_alias,
            "index": self._index,
            "margin": self._margin,
            "spot_days": self._spot_days,
            "calendar": getattr(self._calendar, "name", self._calendar.__class__.__name__),
            "adjust_start_date": self._adjust_start_date,
            "adjust_end_date": self._adjust_end_date,
        }
        return dict


# class ZeroCouponBondSpecification(BondBaseSpecification):
#     def __init__(
#         self,
#         obj_id: str,
#         issue_date: _Union[date, datetime],
#         maturity_date: _Union[date, datetime],
#         currency: str = "EUR",
#         notional: float = 100.0,
#         issuer: str = None,
#         securitization_level: _Union[SecuritizationLevel, str] = None,
#         rating: _Union[Rating, str] = Rating.NONE,
#     ):
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
#         super().__init__(obj_id, issue_date, maturity_date, currency, notional, issuer, securitization_level)

#     @staticmethod
#     def _create_sample(
#         n_samples: int, seed: int = None, ref_date=None, issuers: _List[str] = None, sec_levels: _List[str] = None, currencies: _List[str] = None
#     ):
#         specs = BondBaseSpecification._create_sample(**locals())
#         result = []
#         for i, b in enumerate(specs):
#             result.append(ZeroCouponBondSpecification("ZC_BND_" + str(i), **b))
#         return result

#     def _validate_derived_bond(self):
#         pass

#     def _validate_derived_issued_instrument(self):
#         pass

#     def expected_cashflows(self) -> _List[Tuple[datetime, float]]:
#         """Return a list of all expected cashflows (here only the final notional) together with their payment date.

#         Returns:
#             _List[Tuple[datetime, float]]: The resulting list of all cashflows.
#         """
#         return [(self.maturity_date, self.notional)]


# class PlainVanillaCouponBondSpecification(BondBaseSpecification):
#     def __init__(
#         self,
#         obj_id: str,
#         issue_date: _Union[date, datetime],
#         maturity_date: _Union[date, datetime],
#         accrual_start: _Union[date, datetime],
#         coupon_freq: str,
#         coupon: float,
#         currency: str = "EUR",
#         notional: float = 100.0,
#         issuer: str = None,
#         securitization_level: _Union[SecuritizationLevel, str] = None,
#         stub: bool = True,
#         rating: _Union[Rating, str] = Rating.NONE,
#     ):
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
#         super().__init__(obj_id, issue_date, maturity_date, currency, notional, issuer, securitization_level, rating)

#         self.accrual_start = accrual_start
#         self.coupon_freq = coupon_freq
#         self.coupon = coupon
#         self.stub = stub

#     def expected_cashflows(self) -> _List[Tuple[datetime, float]]:
#         """Return a list of all expected cashflows (final notional and coupons) together with their payment date.

#         Returns:
#             _List[Tuple[datetime, float]]: The resulting list of all cashflows.
#         """
#         # if self.coupon_freq != 'Y':
#         #    raise Exception('Cannot calc cashflows for other than yearly coupons. Missing transformation from yearly coupon to .... ')
#         period = Period.from_string(self.coupon_freq)
#         coupon_multiplier = 1.0
#         if period.years > 0:
#             coupon_multiplier = period.years
#         elif period.months > 0:
#             coupon_multiplier = period.months / 12.0
#         elif period.days > 0:
#             coupon_multiplier = period.days / 365.0
#         schedule = Schedule(self.accrual_start, self.maturity_date, period, stub=self.stub).generate_dates(ends_only=True)
#         result = [(d, self.coupon * coupon_multiplier * self.notional) for d in schedule]
#         result.insert(
#             0, (self.accrual_start, 0.0)
#         )  # the first entry of this schedule is the accrual start which has a cashflow of zero and is just used for accrual calculation
#         result.append((self.maturity_date, self.notional))
#         return result

#     def _to_dict(self) -> dict:
#         result = {
#             "accrual_start": self.accrual_start,
#             "coupon_freq": self.coupon_freq,
#             "coupon": self.coupon,
#         }
#         result.update(super(PlainVanillaCouponBondSpecification, self)._to_dict())
#         return result

#     @staticmethod
#     def _create_sample(
#         n_samples: int, seed: int = None, ref_date=None, issuers: _List[str] = None, sec_levels: _List[str] = None, currencies: _List[str] = None
#     ):
#         specs = BondBaseSpecification._create_sample(**locals())
#         result = []
#         coupons = np.arange(0.0, 0.09, 0.0025)
#         for i, b in enumerate(specs):
#             b["coupon_freq"] = np.random.choice(["3M", "6M", "9M", "1Y"], p=[0.1, 0.4, 0.1, 0.4])
#             issue_date = b["issue_date"]
#             b["accrual_start"] = issue_date + timedelta(days=np.random.randint(low=0, high=10))
#             b["coupon"] = np.random.choice(coupons)
#             result.append(PlainVanillaCouponBondSpecification("BND_PV_" + str(i), **b))
#         return result


# class FixedRateBondSpecification(BondBaseSpecification):
#     def __init__(
#         self,
#         obj_id: str,
#         issue_date: _Union[date, datetime],
#         maturity_date: _Union[date, datetime],
#         coupon_payment_dates: _List[_Union[date, datetime]],
#         coupons: _List[float],
#         currency: str = "EUR",
#         notional: float = 100.0,
#         issuer: str = None,
#         securitization_level: _Union[SecuritizationLevel, str] = None,
#         rating: _Union[Rating, str] = Rating.NONE,
#     ):
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
#             raise Exception(
#                 "Inconsistent combination of issue date '"
#                 + str(issue_date)
#                 + "', payment dates '"
#                 + str(coupon_payment_dates)
#                 + "', and maturity date '"
#                 + str(maturity_date)
#                 + "'."
#             )
#             # TODO: Clarify if inconsistency should be shown explicitly.
#         if len(coupon_payment_dates) == len(coupons):
#             self.__coupons = coupons
#         else:
#             raise Exception("Number of coupons " + str(coupons) + " is not equal to number of coupon payment dates " + str(coupon_payment_dates))

#     @staticmethod
#     def _create_sample(n_samples: int, seed: int = None, ref_date=None, issuers: _List[str] = None):
#         specs = BondBaseSpecification._create_sample(**locals())
#         result = []
#         coupons = np.arange(0.01, 0.09, 0.005)
#         for i, b in enumerate(specs):
#             issue_date = b["issue_date"]
#             n_coupons = np.random.randint(low=1, high=20)
#             days_coupon_period = np.random.choice([90.0, 180.0, 365.0], p=[0.2, 0.2, 0.6])
#             b["coupon_payment_dates"] = [issue_date + timedelta(days=(i + 1) * days_coupon_period) for i in range(n_coupons)]
#             coupon = np.random.choice(coupons)
#             b["coupons"] = [coupon] * n_coupons
#             b["maturity_date"] = b["coupon_payment_dates"][-1]
#             result.append(FixedRateBondSpecification("BND_FR_" + str(i), **b))
#         return result

#     def _validate_derived_bond(self):
#         self.__coupon_payment_dates = _datetime_to_date_list(self.__coupon_payment_dates)
#         # validation of dates' consistency
#         if not _is_ascending_date_list(self.__issue_date, self.__coupon_payment_dates, self.__maturity_date):
#             raise Exception(
#                 "Inconsistent combination of issue date '"
#                 + str(self.__issue_date)
#                 + "', payment dates '"
#                 + str(self.__coupon_payment_dates)
#                 + "', and maturity date '"
#                 + str(self.__maturity_date)
#                 + "'."
#             )
#             # TODO: Clarify if inconsistency should be shown explicitly.
#         if len(self.__coupon_payment_dates) != len(self.__coupons):
#             raise Exception(
#                 "Number of coupons " + str(self.__coupons) + " is not equal to number of coupon payment dates " + str(self.__coupon_payment_dates)
#             )

#     def _validate_derived_issued_instrument(self):
#         pass

#     def _to_dict(self) -> dict:
#         result = {"coupon_payment_dates": self.__coupon_payment_dates, "coupons": self.__coupons}
#         result.update(super(FixedRateBondSpecification, self)._to_dict())
#         return result

#     @classmethod
#     def from_master_data(
#         cls,
#         obj_id: str,
#         issue_date: _Union[date, datetime],
#         maturity_date: _Union[date, datetime],
#         coupon: float,
#         tenor: _Union[Period, str],
#         backwards: bool = True,
#         stub: bool = False,
#         business_day_convention: _Union[RollConvention, str] = RollConvention.FOLLOWING,
#         calendar: _Union[_HolidayBase, str] = None,
#         currency: str = "EUR",
#         notional: float = 100.0,
#         issuer: str = None,
#         securitisation_level: _Union[SecuritizationLevel, str] = None,
#     ):
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
#         return FixedRateBondSpecification(
#             obj_id, issue_date, maturity_date, coupon_payment_dates, coupons, currency, notional, issuer, securitisation_level
#         )

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
#     def __init__(
#         self,
#         obj_id: str,
#         issue_date: _Union[date, datetime],
#         maturity_date: _Union[date, datetime],
#         coupon_period_dates: _List[_Union[date, datetime]],
#         day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
#         spreads: _List[float] = None,
#         reference_index: str = "dummy_curve",
#         currency: str = "EUR",
#         notional: float = 100.0,
#         issuer: str = None,
#         securitisation_level: _Union[SecuritizationLevel, str] = None,
#     ):
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
#         # super().__init__(obj_id, issue_date, maturity_date, currency, notional, issuer, securitisation_level)
#         BondBaseSpecification.__init__(self, obj_id, issue_date, maturity_date, currency, notional, issuer, securitisation_level)
#         self.__coupon_period_dates = _datetime_to_date_list(coupon_period_dates)
#         # validation of dates' consistency
#         if not _is_ascending_date_list(issue_date, coupon_period_dates, maturity_date, False):
#             raise Exception(
#                 "Inconsistent combination of issue date '"
#                 + str(issue_date)
#                 + "', payment dates '"
#                 + str(coupon_period_dates)
#                 + "', and maturity date '"
#                 + str(maturity_date)
#                 + "'."
#             )
#             # TODO: Clarify if inconsistency should be shown explicitly.
#         self.__day_count_convention = DayCounterType.to_string(day_count_convention)
#         if spreads is None:
#             self.__spreads = [0.0] * (len(coupon_period_dates) - 1)
#         elif len(spreads) == len(coupon_period_dates) - 1:
#             self.__spreads = spreads
#         else:
#             raise Exception("Number of spreads " + str(spreads) + " does not fit to number of coupon periods " + str(coupon_period_dates))
#         if reference_index == "":
#             # do not leave reference curve empty as this causes pricer to ignore floating rate coupons!
#             self.__reference_index = "dummy_curve"
#         else:
#             self.__reference_index = reference_index

#     @classmethod
#     def from_master_data(
#         cls,
#         obj_id: str,
#         issue_date: _Union[date, datetime],
#         maturity_date: _Union[date, datetime],
#         tenor: _Union[Period, str],
#         backwards: bool = True,
#         stub: bool = False,
#         business_day_convention: _Union[RollConvention, str] = RollConvention.FOLLOWING,
#         calendar: _Union[_HolidayBase, str] = None,
#         day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
#         spread: float = 0.0,
#         reference_index: str = "dummy_curve",
#         currency: str = "EUR",
#         notional: float = 100.0,
#         issuer: str = None,
#         securitisation_level: _Union[SecuritizationLevel, str] = None,
#     ):
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
#         return FloatingRateNoteSpecification(
#             obj_id,
#             issue_date,
#             maturity_date,
#             coupon_period_dates,
#             day_count_convention,
#             spreads,
#             reference_index,
#             currency,
#             notional,
#             issuer,
#             securitisation_level,
#         )

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
#     def daycount_convention(self, day_count_convention: _Union[DayCounterType, str]) -> str:
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


# class FixedToFloatingRateNoteSpecification(FixedRateBondSpecification, FloatingRateBondSpecification):
#     def __init__(
#         self,
#         obj_id: str,
#         issue_date: _Union[date, datetime],
#         maturity_date: _Union[date, datetime],
#         coupon_payment_dates: _List[_Union[date, datetime]],
#         coupons: _List[float],
#         coupon_period_dates: _List[_Union[date, datetime]],
#         day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
#         spreads: _List[float] = None,
#         reference_index: str = "dummy_curve",
#         currency: str = "EUR",
#         notional: float = 100.0,
#         issuer: str = None,
#         securitisation_level: _Union[SecuritizationLevel, str] = None,
#     ):
#         """
#         Fixed-to-floating rate note specification by providing fixed rate coupons and fixed rate coupon payment dates
#         as well as floating rate coupon periods directly.
#         """
#         # TODO FIX THIS CLASS!!!!!!!!!!!!!!!!
#         raise Exception("Not working properly, @Stefan: Please fix me!!!!")
#         FixedRateBondSpecification.__init__(
#             self, obj_id, issue_date, maturity_date, coupon_payment_dates, coupons, currency, notional, issuer, securitisation_level
#         )

#         FloatingRateNoteSpecification.__init__(
#             self,
#             obj_id,
#             issue_date,
#             maturity_date,
#             coupon_period_dates,
#             day_count_convention,
#             spreads,
#             reference_index,
#             currency,
#             notional,
#             issuer,
#             securitisation_level,
#         )

#     @classmethod
#     def from_master_data(
#         cls,
#         obj_id: str,
#         issue_date: _Union[date, datetime],
#         fixed_to_float_date: _Union[date, datetime],
#         maturity_date: _Union[date, datetime],
#         coupon: float,
#         tenor_fixed: _Union[Period, str],
#         tenor_float: _Union[Period, str],
#         backwards_fixed: bool = True,
#         backwards_float: bool = True,
#         stub_fixed: bool = False,
#         stub_float: bool = False,
#         business_day_convention_fixed: _Union[RollConvention, str] = RollConvention.FOLLOWING,
#         business_day_convention_float: _Union[RollConvention, str] = RollConvention.FOLLOWING,
#         calendar_fixed: _Union[_HolidayBase, str] = None,
#         calendar_float: _Union[_HolidayBase, str] = None,
#         day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
#         spread: float = 0.0,
#         reference_index: str = "dummy_curve",
#         currency: _Union[str, int] = "EUR",
#         notional: float = 100.0,
#         issuer: str = None,
#         securitisation_level: _Union[SecuritizationLevel, str] = None,
#     ):
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
#         fixed_rate_part = FixedRateBondSpecification.from_master_data(
#             obj_id,
#             issue_date,
#             fixed_to_float_date,
#             coupon,
#             tenor_fixed,
#             backwards_fixed,
#             stub_fixed,
#             business_day_convention_fixed,
#             calendar_fixed,
#             currency,
#             notional,
#             issuer,
#             securitisation_level,
#         )
#         floating_rate_part = FloatingRateNoteSpecification.from_master_data(
#             obj_id,
#             fixed_to_float_date,
#             maturity_date,
#             tenor_float,
#             backwards_float,
#             stub_float,
#             business_day_convention_float,
#             calendar_float,
#             day_count_convention,
#             spread,
#             reference_index,
#             currency,
#             notional,
#             issuer,
#             securitisation_level,
#         )
#         return FixedToFloatingRateNoteSpecification(
#             obj_id,
#             issue_date,
#             maturity_date,
#             fixed_rate_part.coupon_payment_dates,
#             fixed_rate_part.coupons,
#             floating_rate_part.coupon_period_dates,
#             day_count_convention,
#             floating_rate_part.spreads,
#             reference_index,
#             currency,
#             notional,
#             issuer,
#             securitisation_level,
#         )


# ToDo
# Unit tests
# re-arrange test files
# allow for features of bonds:
# -- amortizing CF --> implement amortizing cash flows / amortization cash flows
# -- fix coupons provided as cf list


#
# class BondBaseSpecification(interfaces.FactoryObject):
#     """Abstract base class for bond specifications."""

#     def __init__(
#         self,
#         obj_id: str,
#         schedule: Schedule,
#         notional: float,
#         currency: Union[Currency, str],
#         issue_date: Union[date, datetime],
#         maturity_date: Union[date, datetime],
#         spread: float = 0.0,
#         issuer: Optional[str] = None,
#         securitization_level: Optional[Union[SecuritizationLevel, str]] = SecuritizationLevel.NONE,
#         rating: Optional[Union[Rating, str]] = Rating.NONE,
#         day_count_convention: DayCounterType = DayCounterType.ActActICMA,
#     ):
#         """
#         Initializes the base bond specification.

#         Args:
#             obj_id (str): A unique identifier for the bond, e.g., ISIN.
#             schedule (Schedule): The payment schedule of the bond.
#             notional (float): The face value of the bond.
#             currency (Union[Currency, str]): The currency of the bond.
#             issue_date (Union[date, datetime]): The date the bond was issued.
#             maturity_date (Union[date, datetime]): Maturity date of the bond.
#             coupon (float): The annual coupon rate (e.g., 0.05 for 5%).
#             spread (float): Credit spread.
#             issuer (Optional[str], optional): The issuer of the bond. Defaults to None.
#             securitization_level (Optional[Union[SecuritizationLevel, str]], optional): The securitization level. Defaults to SecuritizationLevel.NONE.
#             rating (Optional[Union[Rating, str]], optional): The credit rating of the bond. Defaults to Rating.NONE.
#             day_count_convention (DayCounterType, optional): The day count convention for accrual calculations. Defaults to DayCounterType.ActActICMA.
#         """

#         self.obj_id = obj_id

#         if not isinstance(schedule, Schedule):
#             raise TypeError("schedule must be an instance of rivapy.tools.datetools.Schedule.")

#         self._schedule = schedule
#         self._notional = notional
#         self._currency = currency
#         self._issue_date = _date_to_datetime(issue_date)
#         self._maturity_date = _date_to_datetime(maturity_date)
#         self._spread = spread
#         self._issuer = issuer
#         self._securitization_level = securitization_level
#         self._rating = rating

#         _check_start_before_end(self._issue_date, self._maturity_date)

#         if spread < 0:
#             raise ValueError("Spread must be non-negative.")

#         self._accrual_day_counter = DayCounter(day_count_convention)
#         self._day_count_convention = day_count_convention

#         self._coupon_freq = self._schedule.time_period

#         if self._day_count_convention == DayCounterType.ActActICMA:
#             _check = False
#             for cp_freq_str in ["1Y", "6M", "3M"]:
#                 if self._coupon_freq == Period.from_string(cp_freq_str):
#                     _check = True
#                     break
#             if _check == False:
#                 raise ValueError("For the Act/Act ICMA day count convention only a coupon frequency of 1Y, 6M or 3M is supported!")

#     def _to_dict(self) -> Dict:
#         # TODO: further addtion to the dictionary like Schedule
#         return_dict = {
#             "obj_id": self.obj_id,
#             "issuer": self._issuer,
#             "securitization_level": self._securitization_level,
#             "issue_date": self._issue_date,
#             "maturity_date": self._maturity_date,
#             "spread": self._spread,
#             "currency": self._currency,
#             "notional": self._notional,
#             "rating": self._rating,
#             "day_count_convention": self._day_count_convention,
#         }
#         return return_dict

#     @property
#     def schedule(self) -> Schedule:
#         """The bond's schedule of payments."""
#         return self._schedule

#     @property
#     def issue_date(self) -> datetime:
#         """The bond's issue date as a datetime object."""
#         return self._issue_date

#     @issue_date.setter
#     def issue_date(self, value: Union[date, datetime]):
#         self._issue_date = _date_to_datetime(value)

#     @property
#     def maturity_date(self) -> datetime:
#         """The bond's issue date as a datetime object."""
#         return self._maturity_date

#     @maturity_date.setter
#     def maturity_date(self, value: Union[date, datetime]):
#         self._maturity_date = _date_to_datetime(value)

#     @property
#     def spread(self) -> float:
#         """The bond's credit spread"""
#         return self._spread

#     @spread.setter
#     def spread(self, value: float):
#         self._spread = _check_positivity(value)

#     @property
#     def notional(self) -> float:
#         """The bond's notional amount (face value)."""
#         return self._notional

#     @notional.setter
#     def notional(self, value: float):
#         self._notional = _check_positivity(value)

#     @property
#     def currency(self) -> str:
#         """The bond's currency as a string."""
#         return self._currency

#     @currency.setter
#     def currency(self, value: Union[Currency, str]):
#         self._currency = Currency.to_string(value)

#     @property
#     def issuer(self) -> Optional[str]:
#         """The bond's issuer."""
#         return self._issuer

#     @issuer.setter
#     def issuer(self, value: Optional[str]):
#         self._issuer = value

#     @property
#     def securitization_level(self) -> str:
#         """The bond's securitization level as a string."""
#         return self._securitization_level

#     @securitization_level.setter
#     def securitization_level(self, value: Union[SecuritizationLevel, str]):
#         self._securitization_level = SecuritizationLevel.to_string(value)

#     @property
#     def rating(self) -> str:
#         """The bond's credit rating as a string."""
#         return self._rating

#     @rating.setter
#     def rating(self, value: Union[Rating, str]):
#         self._rating = Rating.to_string(value)

#     @property
#     def day_count_convention(self) -> str:
#         return self._day_count_convention

#     @day_count_convention.setter
#     def day_count_convention(self, value: Union[DayCounterType, str]):
#         self._day_count_convention = DayCounterType.to_string(value)

#     def _get_coupon_frequency(self):
#         if self._coupon_freq.years > 0:
#             coupon_frequency = 1.0 / self._coupon_freq.years
#         else:
#             coupon_frequency = 12.0 / self._coupon_freq.months

#         return coupon_frequency

#     @abc.abstractmethod
#     def expected_cashflows(self) -> List[Tuple[datetime, float]]:
#         """
#         Computes all expected cashflows of the bond.

#         Returns:
#             List[Tuple[datetime, float]]: A list of tuples, where each tuple contains
#                                           the payment date and the cashflow amount.
#         """
#         pass

#     @abc.abstractmethod
#     def compute_dirty_price(self, discount_curve: DiscountCurve) -> float:
#         """
#         Computes the dirty price of the bond.
#         The dirty price is the price of a bond including any accrued interest.

#         Args:
#             discount_curve (DiscountCurve): The curve used to discount future cashflows.

#         Returns:
#             float: The calculated dirty price.
#         """
#         pass

#     @abc.abstractmethod
#     def compute_clean_price(self, discount_curve: DiscountCurve) -> float:
#         """
#         Computes the clean price of the bond by discounting all future cashflows.
#         The clean price is the price of a bond including any accrued interest.

#         Args:
#             discount_curve (DiscountCurve): The curve used to discount future cashflows.

#         Returns:
#             float: The calculated clean price.
#         """
#         pass

#     @abc.abstractmethod
#     def compute_yield(self, price: float, val_date: datetime) -> float:
#         """
#         Computes the yield-to-maturity (YTM) of the bond.

#         Args:
#             price (float): The dirty price of the bond.
#             val_date (datetime): The valuation date.

#         Returns:
#             float: The computed yield-to-maturity.
#         """
#         pass


# class FixedRateBond(BondBaseSpecification):
#     """
#     Represents a fixed-rate bond with regular coupon payments.
#     """

#     def __init__(
#         self,
#         obj_id: str,
#         schedule: Schedule,
#         notional: float,
#         currency: Union[Currency, str],
#         issue_date: Union[date, datetime],
#         maturity_date: Union[date, datetime],
#         coupon: float,
#         spread: float = 0.0,
#         issuer: Optional[str] = None,
#         securitization_level: Optional[Union[SecuritizationLevel, str]] = SecuritizationLevel.NONE,
#         rating: Optional[Union[Rating, str]] = Rating.NONE,
#         day_count_convention: DayCounterType = DayCounterType.ActActICMA,
#     ):
#         """
#         Initializes a fixed-rate bond.

#         Args:
#             obj_id (str): A unique identifier for the bond.
#             schedule (Schedule): The payment schedule of the bond.
#             notional (float): The face value of the bond.
#             currency (Union[Currency, str]): The currency of the bond.
#             issue_date (Union[date, datetime]): The date the bond was issued.
#             maturity_date (Union[date, datetime]): Maturity date of the bond.
#             coupon (float): The annual coupon rate (e.g., 0.05 for 5%).
#             spread (float): Credit spread.
#             issuer (Optional[str], optional): The issuer of the bond. Defaults to None.
#             securitization_level (Optional[Union[SecuritizationLevel, str]], optional): The securitization level. Defaults to SecuritizationLevel.NONE.
#             rating (Optional[Union[Rating, str]], optional): The credit rating of the bond. Defaults to Rating.NONE.
#             day_count_convention (DayCounterType, optional): The day count convention for accrual calculations. Defaults to DayCounterType.ActActICMA.
#         """
#         super().__init__(
#             obj_id,
#             schedule,
#             notional,
#             currency,
#             issue_date,
#             maturity_date,
#             spread,
#             issuer,
#             securitization_level,
#             rating,
#             day_count_convention,
#         )
#         if coupon < 0:
#             raise ValueError("Coupon rate must be non-negative.")

#         self._coupon = coupon
#         self.__schedule_dates = self._schedule.generate_dates(ends_only=False)
#         self._cashflows = self.expected_cashflows()

#     @property
#     def coupon(self) -> float:
#         """The bond's annual coupon rate"""
#         return self._coupon

#     @coupon.setter
#     def coupon(self, value: float):
#         self._coupon = _check_positivity(value)

#     def _to_dict(self) -> Dict:
#         _dict = super()._to_dict()
#         _dict["coupon"] = self._coupon
#         return _dict

#     @staticmethod
#     def _create_sample(n_samples: int, seed: int = None):
#         result = []
#         if seed is not None:
#             np.random.seed(seed)

#         issue_date = datetime(2025, 1, 1)
#         maturity_date = datetime(2027, 1, 1)
#         notional = 100.0
#         currency = Currency.EUR
#         securitization_level = SecuritizationLevel.SUBORDINATED
#         daycounter = DayCounterType.ACT_ACT
#         for i in range(n_samples):
#             coupon = np.random.choice([0.0, 0.01, 0.03, 0.05])
#             period = np.random.choice(["1Y", "6M", "3M"])
#             schedule = Schedule(
#                 start_day=issue_date, end_day=maturity_date, time_period=Period.from_string(period), business_day_convention=RollConvention.UNADJUSTED
#             )
#             result.append(
#                 {
#                     "obj_id": f"ID_{i}",
#                     "schedule": schedule,
#                     "notional": notional,
#                     "currency": currency,
#                     "issue_date": issue_date,
#                     "coupon": coupon,
#                     "securitization_level": securitization_level,
#                     "day_count_convention": daycounter,
#                 }
#             )

#     def expected_cashflows(self) -> List[Tuple[datetime, float]]:
#         """
#         Computes all expected cashflows (coupons and notional) of the bond.

#         Returns:
#             List[Tuple[datetime, float]]: A sorted list of (date, amount) tuples for each cashflow.
#         """
#         # Generate all schedule dates, which are already business-day adjusted

#         cashflows = []
#         # Get frequency information needed for Act/Act ICMA calculation
#         coupon_freq = self._get_coupon_frequency()

#         # --- Iterate over all coupon periods ---
#         for i in range(len(self.__schedule_dates) - 1):
#             period_start_dt = _date_to_datetime(self.__schedule_dates[i])
#             payment_date_dt = _date_to_datetime(self.__schedule_dates[i + 1])

#             # Skip cashflows that are paid out before or on the issue date
#             if payment_date_dt <= self.issue_date:
#                 continue

#             # The accrual for the coupon calculation always uses the full period from the schedule
#             # to correctly handle stub periods (short/long first or last coupons).
#             year_fraction_for_coupon = self._accrual_day_counter.yf(
#                 period_start_dt, payment_date_dt, coupon_schedule=self.__schedule_dates, coupon_frequency=coupon_freq  # coupon_frequency_int
#             )

#             coupon_amount = self._notional * self._coupon * year_fraction_for_coupon
#             if coupon_amount > 0.0:
#                 cashflows.append((payment_date_dt, coupon_amount))

#         # Add notional at maturity date (which is the last date in the schedule)
#         maturity_payment_date = _date_to_datetime(self._maturity_date)
#         if maturity_payment_date >= self._issue_date:
#             cashflows.append((maturity_payment_date, self._notional))

#         # Use a dictionary to sum amounts for cashflows on the same date (e.g., last coupon + notional)
#         combined_cashflows = defaultdict(float)
#         for cf_date, amount in cashflows:
#             # Normalize datetime to date to ensure correct grouping if time components differ
#             normalized_date = cf_date.replace(hour=0, minute=0, second=0, microsecond=0)
#             combined_cashflows[normalized_date] += amount

#         # Convert back to list of tuples and sort by date
#         return sorted(combined_cashflows.items(), key=lambda x: x[0])

#     def compute_accrued_interest(self, valuation_date: Union[date, datetime]) -> float:
#         """
#         Computes the accrued interest of the bond on a given valuation date.

#         Args:
#             valuation_date (Union[date, datetime]): The date for which to calculate the accrued interest.

#         Returns:
#             float: The amount of accrued interest. Returns 0.0 if the valuation date is
#                    outside the bond's life (before issue or on/after maturity).
#         """
#         val_date_dt = _date_to_datetime(valuation_date)

#         # For zero-coupon bonds, accrued interest is always zero. This also prevents division by zero errors.
#         if self._coupon == 0.0:
#             return 0.0

#         # No accrued interest if valuation is outside the bond's life
#         if val_date_dt >= self.maturity_date or val_date_dt < self.issue_date:
#             return 0.0

#         current_accrual_start = None
#         current_accrual_end = None

#         if len(self.__schedule_dates) < 2:
#             return 0.0

#         # Find the coupon period that contains the valuation date
#         for i in range(len(self.__schedule_dates) - 1):
#             p_start_dt = _date_to_datetime(self.__schedule_dates[i])
#             p_end_dt = _date_to_datetime(self.__schedule_dates[i + 1])

#             if p_start_dt <= val_date_dt < p_end_dt:
#                 current_accrual_start = p_start_dt
#                 current_accrual_end = p_end_dt
#                 break

#         if current_accrual_start is None or current_accrual_end is None:
#             return 0.0

#         # Get frequency information needed for Act/Act ICMA
#         coupon_frequency = self._get_coupon_frequency()

#         # Calculate accrued interest year fraction for the period [current_accrual_start, val_date_dt]
#         accrued_year_fraction = self._accrual_day_counter.yf(current_accrual_start, val_date_dt, self.__schedule_dates, coupon_frequency)
#         return self._notional * self._coupon * accrued_year_fraction

#     def compute_clean_price(self, value_date: datetime, discount_curve: DiscountCurve) -> float:
#         """
#         Computes the clean price of the bond by discounting all future cashflows.
#         The clean price is the price of a bond including any accrued interest.

#         Args:
#             discount_curve (DiscountCurve): The curve used to discount future cashflows.

#         Returns:
#             float: The calculated clean price.
#         """
#         # val_date_dt = _date_to_datetime(discount_curve.valuation_date)
#         # cashflows = self.cashflows#self.expected_cashflows()
#         ref_date = discount_curve.refdate

#         pv_cashflows = 0.0
#         for c in self._cashflows:
#             if c[0] > value_date:
#                 rate = discount_curve.value(refdate=ref_date, d=value_date)
#                 yf = self._accrual_day_counter.yf(d1=value_date, d2=c[0])
#                 df = 1 / ((1 + rate + self._spread) ** yf)
#                 pv_cashflows += df * c[1]
#         return pv_cashflows

#     def compute_dirty_price(self, value_date: datetime, discount_curve: DiscountCurve) -> float:
#         """
#         Computes the dirty price of the bond.
#         Dirty Price = Clean Price + Accrued Interest.

#         Args:
#             discount_curve (DiscountCurve): The curve used to discount future cashflows.

#         Returns:
#             float: The dirty price of the bond.
#         """
#         clean_price = self.compute_clean_price(value_date, discount_curve)
#         accrued = self.compute_accrued_interest(value_date)
#         return clean_price + accrued

#     def compute_yield(
#         self, dirty_price: float, val_date: datetime, yield_search_lower_bound: float = -0.2, yield_search_upper_bound: float = 1.5
#     ) -> float:
#         """
#         Computes the yield-to-maturity (YTM) for a given dirty price.
#         This method uses the brentq root-finding algorithm to find the yield that
#         equates the present value of future cashflows to the given dirty price.

#         Args:
#             dirty_price (float): The dirty price of the bond.
#             val_date (datetime): The valuation date.
#             yield_search_lower_bound (float, optional): The lower bound for the yield search. Defaults to -0.2.
#             yield_search_upper_bound (float, optional): The upper bound for the yield search. Defaults to 1.5.

#         Returns:
#             float: The calculated yield-to-maturity (annually compounded).
#         """
#         valuation_datetime = _date_to_datetime(val_date)
#         # cashflows = self.expected_cashflows()

#         # For ActActICMA, we need the schedule and frequency for the day counter
#         all_schedule_dates = [_date for _date, cpn in self._cashflows]
#         coupon_freq = self._get_coupon_frequency()

#         def target_function(r: float) -> float:
#             """
#             Calculates the difference between the PV of cashflows (for a given yield r) and the dirty price.
#             The root of this function is the desired YTM.
#             """
#             # Calculate the dirty price for a given yield 'r' without creating a full DiscountCurve object. This ensures we use the bond's specific day counter for the yield calculation, matching QuantLib.
#             pv_cashflows = 0.0
#             for cf_date, amount in self._cashflows:
#                 if cf_date > valuation_datetime:
#                     # Calculate year fraction for discounting using the bond's accrual day counter
#                     yf = self._accrual_day_counter.yf(valuation_datetime, cf_date, coupon_schedule=all_schedule_dates, coupon_frequency=coupon_freq)

#                     # Discount the cashflow (annually compounded, matching QL's default)
#                     df = 1.0 / ((1.0 + r + self._spread) ** yf)
#                     pv_cashflows += df * amount

#             return pv_cashflows - dirty_price

#         # Use brentq to find the root of the target function (i.e., the yield)
#         result = brentq(target_function, yield_search_lower_bound, yield_search_upper_bound, full_output=False)
#         return result


# class FloatingRateBond(BondBaseSpecification):
#     def __init__(
#         self,
#         obj_id: str,
#         schedule: Schedule,
#         notional: float,
#         currency: Union[Currency, str],
#         issue_date: Union[date, datetime],
#         maturity_date: Union[date, datetime],
#         coupon_ref_curve: DiscountCurve,
#         coupon_spread: float,
#         fixing_date: Union[date, datetime],
#         fixing_coupon: float,
#         spread: float = 0.0,
#         issuer: Optional[str] = None,
#         securitization_level: Optional[Union[SecuritizationLevel, str]] = SecuritizationLevel.NONE,
#         rating: Optional[Union[Rating, str]] = Rating.NONE,
#         day_count_convention: DayCounterType = DayCounterType.ActActICMA,
#     ):
#         super().__init__(
#             obj_id,
#             schedule,
#             notional,
#             currency,
#             issue_date,
#             maturity_date,
#             spread,
#             issuer,
#             securitization_level,
#             rating,
#             day_count_convention,
#         )
#         self._coupon_ref_curve = coupon_ref_curve
#         self._coupon_spread = coupon_spread
#         self._fixing_date = _date_to_datetime(fixing_date)
#         self._fixing_coupon = fixing_coupon

#         _check_start_before_end(self._fixing_date, self._maturity_date)
#         self.__schedule_dates = self._schedule.generate_dates(ends_only=False)

#         self._coupons: List[Tuple[datetime, datetime, float]] = []

#         self._cashflows = self.expected_cashflows()

#     @property
#     def cashflows(self) -> List[Tuple[datetime, float]]:
#         return self._cashflows

#     @property
#     def coupons(self) -> List[Tuple[datetime, datetime, float]]:
#         return self._coupons

#     @property
#     def coupon_ref_curve(self) -> DiscountCurve:
#         return self._coupon_ref_curve

#     @property
#     def coupon_spread(self) -> float:
#         return self._coupon_spread

#     @coupon_spread.setter
#     def coupon_spread(self, value: float):
#         self._coupon_spread = _check_positivity(value=value)

#     @property
#     def fixing_date(self) -> datetime:
#         return self._fixing_date

#     @fixing_date.setter
#     def fixing_date(self, value: Union[datetime, date]):
#         self._fixing_date = _date_to_datetime(value)

#     @property
#     def fixing_coupon(self) -> float:
#         return self._fixing_coupon

#     @fixing_coupon.setter
#     def fixing_coupon(self, value: float):
#         self._fixing_coupon = _check_positivity(value=value)

#     def _to_dict(self):
#         _dict = {"coupon_spread": self._coupon_spread, "fixing_date": self._fixing_date, "fixing_coupon": self._fixing_coupon}
#         return {**super()._to_dict(), **_dict}

#     def expected_cashflows(self) -> List[Tuple[datetime, float]]:
#         """
#         Computes all expected cashflows (coupons and notional) of the bond.

#         Returns:
#             List[Tuple[datetime, float]]: A sorted list of (date, amount) tuples for each cashflow.
#         """

#         cashflows = []
#         # Get frequency information needed for Act/Act ICMA calculation
#         coupon_freq = self._get_coupon_frequency()

#         _fixed_coupon = False
#         _curve_ref_date = self._coupon_ref_curve.refdate

#         for i in range(len(self.__schedule_dates) - 1):
#             period_start_dt = _date_to_datetime(self.__schedule_dates[i])
#             payment_date_dt = _date_to_datetime(self.__schedule_dates[i + 1])

#             if payment_date_dt < self._fixing_date:
#                 continue

#             if (not _fixed_coupon) and (self._fixing_date <= payment_date_dt):
#                 # first coupon payment after the fixing date has a fixed coupon rate
#                 coupon = self._fixing_coupon
#                 _fixed_coupon = True

#             elif (
#                 (_fixed_coupon)
#                 and (payment_date_dt < _date_to_datetime(self._coupon_ref_curve.get_dates()[0]))
#                 and (self._fixing_date < payment_date_dt)
#             ):
#                 # coupon payment falls between fixing date (without being the first coupon payment after the fixing date) and the start of the coupon rate curve
#                 raise ValueError(
#                     f"Coupon payment {payment_date_dt} is not the first coupon payment after fixing and happens before the start date of the coupon reference curve {_date_to_datetime(self._coupon_ref_curve.get_dates()[0])}"
#                 )
#             else:
#                 coupon = self._coupon_ref_curve.value(refdate=_curve_ref_date, d=payment_date_dt)

#             year_fraction_for_coupon = self._accrual_day_counter.yf(
#                 period_start_dt, payment_date_dt, coupon_schedule=self.__schedule_dates, coupon_frequency=coupon_freq  # coupon_frequency_int
#             )

#             print(f"Coupon rate: {coupon}")
#             coupon = self._notional * coupon * year_fraction_for_coupon
#             if coupon > 0.0:
#                 cashflows.append((payment_date_dt, coupon))

#             self._coupons.append((period_start_dt, payment_date_dt, coupon))

#         # Add notional at maturity date (which is the last date in the schedule)
#         maturity_payment_date = _date_to_datetime(self._maturity_date)
#         if maturity_payment_date >= self._issue_date:
#             cashflows.append((maturity_payment_date, self._notional))

#         combined_cashflows = defaultdict(float)
#         for cf_date, amount in cashflows:
#             # Normalize datetime to date to ensure correct grouping if time components differ
#             normalized_date = cf_date.replace(hour=0, minute=0, second=0, microsecond=0)
#             combined_cashflows[normalized_date] += amount

#         # Convert back to list of tuples and sort by date
#         return sorted(combined_cashflows.items(), key=lambda x: x[0])

#     def compute_accrued_interest(self, valuation_date: Union[date, datetime]) -> float:
#         """
#         Computes the accrued interest of the bond on a given valuation date.

#         Args:
#             valuation_date (Union[date, datetime]): The date for which to calculate the accrued interest.

#         Returns:
#             float: The amount of accrued interest. Returns 0.0 if the valuation date is
#                    outside the bond's life (before issue or on/after maturity).
#         """
#         val_date_dt = _date_to_datetime(valuation_date)

#         if (val_date_dt < self._coupons[0][0]) or (val_date_dt > self._coupons[-1][1]):
#             raise ValueError(f" Valuation date {val_date_dt} is out of bonds {self._coupons[0][0]} / {self._coupons[-1][1]}")

#         coupon_frequency = self._get_coupon_frequency()

#         for dt_start, dt_end, coupon in self._coupons:
#             if dt_start <= val_date_dt < dt_end:
#                 accrued_year_fraction = self._accrual_day_counter.yf(dt_start, val_date_dt, self.__schedule_dates, coupon_frequency)
#                 return self._notional * coupon * accrued_year_fraction

#     def compute_clean_price(self, discount_curve):
#         return super().compute_clean_price(discount_curve)

#     def compute_dirty_price(self, discount_curve):
#         return super().compute_dirty_price(discount_curve)

#     def compute_yield(self, price, val_date):
#         return super().compute_yield(price, val_date)
