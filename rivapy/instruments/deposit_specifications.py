from abc import abstractmethod as _abstractmethod
from typing import List as _List, Union as _Union, Tuple, Optional as _Optional
import numpy as np
from datetime import datetime, date, timedelta
from holidays import HolidayBase as _HolidayBase
from holidays import EuropeanCentralBank as _ECB
from rivapy.tools.datetools import Period, _date_to_datetime, _datetime_to_date_list, _term_to_period
from rivapy.tools.enums import DayCounterType, RollConvention, SecuritizationLevel, Currency, Rating, RollRule
from rivapy.tools._validators import _check_positivity, _check_start_before_end, _string_to_calendar, _is_ascending_date_list
import rivapy.tools.interfaces as interfaces
from rivapy.tools.datetools import Period, Schedule


class DepositSpecification(interfaces.FactoryObject):

    def __init__(
        # ToDo:
        # - check application of bdc
        # - check application of roll convention
        # - add term (and calculate maturity date from start date)
        # - consider payment date vs end of accrual period date
        # . spot days
        self,
        obj_id: str,
        fixing_date: _Union[date, datetime],
        start_date: _Union[date, datetime],
        maturity_date: _Union[date, datetime],
        currency: _Union[Currency, str] = "EUR",
        notional: float = 100.0,
        rate: float = 0.00,
        term: _Optional[_Union[Period, str]] = None,
        day_count_convention: _Union[DayCounterType, str] = DayCounterType.ACT360,
        business_day_convention: _Union[RollConvention, str] = RollConvention.MODIFIED_FOLLOWING,
        roll_convention: _Union[RollRule, str] = RollRule.EOM,
        spot_days: int = 0,
        calendar: _Union[_HolidayBase, str] = _ECB(),
        issuer: _Optional[str] = None,
        securitization_level: _Union[SecuritizationLevel, str] = SecuritizationLevel.NONE,
        rating: _Union[Rating, str] = Rating.NONE,
    ):
        """Base Deposit specification.

        Args:
            obj_id (str): (Preferably) Unique label of the deposit.
            fixing_date (_Union[date, datetime]): Date on which the reference rate (e.g. EURIBOR) is set for floating rate deposits.
            start_date (_Union[date, datetime]): Date on when deposits begins for accrual or settlement.
            maturity_date (_Union[date, datetime]): Date when deposit is repaid, defining the end of the interest period. Must lie after the start_date.
            currency (str, optional): Currency as alphabetic, Defaults to 'EUR'.
            notional (float, optional): Deposit's notional/face value. Must be positive. Defaults to 100.0.
            rate (float): Deposit fixed interest rate.
            term (_Union[Period, str], optional): Deposit term. If provided, it is used to calculate the maturity date from the start date.
            day_count_convention (Union[DayCounter, str], optional): Day count convention for determining period
                                                                     length. Defaults to DayCounter.ThirtyU360.
            business_day_convention (Union[RollConvention, str], optional): Set of rules defining the adjustment of
                                                                            days to ensure each date being a business
                                                                            day with respect to a given holiday
                                                                            calendar. Defaults to
                                                                            RollConvention.FOLLOWING
            roll_convention (Union[RollRule],str], optional): Roll convention to be applied when building a schedule. Defaults to RollRule.NONE.
            spot_days (int, optional): Number of days after fixing date when the deposit is actually settled. Defaults to 2.
            issuer (str, optional): Name/id of issuer. Defaults to None.
            securitization_level (_Union[SecuritizationLevel, str], optional): Securitization level. Defaults to None.
            rating (_Union[Rating, str]): Paper rating.
        """
        self.obj_id = obj_id
        # TODO: check for combinations of dates / period
        # if start date and period
        # if start date and maturity date
        # if T/N or O/N then reset spot lag
        # if spot days is NONE and term not None

        if fixing_date is not None:
            self.fixing_date = fixing_date
        elif start_date is not None:
            self.fixing_date = start_date

        self._start_date = start_date
        self._maturity_date = maturity_date
        self._currency = currency
        self._notional = notional
        self._rate = rate
        self._term = _term_to_period(term) if term is not None else None
        self._roll_convention = RollRule.to_string(roll_convention)
        self._spot_days = spot_days
        self._day_count_convention = day_count_convention
        self._business_day_convention = business_day_convention
        self._calendar = _string_to_calendar(calendar)
        if issuer is not None:
            self._issuer = issuer
        if securitization_level is not None:
            self._securitization_level = SecuritizationLevel.to_string(securitization_level)
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
            start_date = ref_date + timedelta(days=np.random.randint(low=-365, high=0))
            result.append(
                {
                    "fixing_date": start_date + timedelta(days=np.random.randint(low=-2, high=0)),
                    "start_date": start_date,
                    "maturity_date": ref_date + timedelta(days=days),
                    "currency": np.random.choice(currencies),
                    "notional": np.random.choice([100.0, 1000.0, 10_000.0, 100_0000.0]),
                    "rate": np.random.choice([0.01, 0.02, 0.03, 0.04, 0.05]),
                    "issuer": np.random.choice(issuers),
                    "securitization_level": np.random.choice(sec_levels),
                }
            )
        return result

    def _validate_derived_issued_instrument(self):
        self._start_date, self._maturity_date = _check_start_before_end(self._start_date, self._maturity_date)

    def _to_dict(self) -> dict:
        result = {
            "obj_id": self.obj_id,
            "fixing_date:": self.fixing_date,
            "start_date": self.start_date,
            "maturity_date": self.maturity_date,
            "currency": self.currency,
            "notional": self.notional,
            "rate": self.rate,
            "day_count_convention": self.day_count_convention,
            "roll_convention": self._roll_convention,
            "spot_days": self._spot_days,
            "business_day_convention": self.business_day_convention,
            "issuer": self.issuer,
            "securitization_level": self.securitization_level,
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
    def rate(self) -> float:
        """
        Getter for instrument's rate.

        Returns:
            float: Instrument's rate.
        """
        return self._rate

    @rate.setter
    def rate(self, rate: float):
        """
        Setter for instrument's rate.

        Args:
            rate(float): interest rate of the instrument.
        """
        self._rate = rate

    @property
    def rating(self) -> str:
        return self._rating

    @rating.setter
    def rating(self, rating: _Union[Rating, str]):
        self._rating = Rating.to_string(rating)

    @property
    def securitization_level(self) -> str:
        """
        Getter for instrument's securitisation level.

        Returns:
            str: Instrument's securitisation level.
        """
        return self._securitization_level

    @securitization_level.setter
    def securitization_level(self, value: _Union[SecuritizationLevel, str]):
        self._securitization_level = SecuritizationLevel.to_string(value)

    @property
    def start_date(self) -> date:
        """
        Getter for deposit's start date.

        Returns:
            date: deposit's start date.
        """
        return self._start_date

    @start_date.setter
    def start_date(self, start_date: _Union[datetime, date]):
        """
        Setter for deposit's start date.

        Args:
            start_date (Union[datetime, date]): deposit's start date.
        """
        self._start_date = _date_to_datetime(start_date)

    @property
    def maturity_date(self) -> datetime:
        """
        Getter for deposit's maturity date.

        Returns:
            date: deposit's maturity date.
        """
        return self._maturity_date

    @maturity_date.setter
    def maturity_date(self, maturity_date: _Union[datetime, date]):
        """
        Setter for deposit's maturity date.

        Args:
            maturity_date (Union[datetime, date]): deposit's maturity date.
        """
        self._maturity_date = _date_to_datetime(maturity_date)

    @property
    def currency(self) -> str:
        """
        Getter for deposit's currency.

        Returns:
            str: deposit's  currency code
        """
        return self._currency

    @currency.setter
    def currency(self, currency: str):
        self._currency = Currency.to_string(currency)

    @property
    def notional(self) -> float:
        """
        Getter for deposit's face value.

        Returns:
            float: deposit's face value.
        """
        return self._notional

    @notional.setter
    def notional(self, notional):
        self._notional = _check_positivity(notional)

    @property
    def day_count_convention(self) -> str:
        """
        Getter for FRA's day count convention.

        Returns:
            str: FRA's day count convention.
        """
        return self._day_count_convention

    @day_count_convention.setter
    def day_count_convention(self, day_count_convention: _Union[DayCounterType, str]):
        self._day_count_convention = DayCounterType.to_string(day_count_convention)

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

    # endregion

    def expected_cashflows(self) -> _List[Tuple[datetime, float]]:
        """Return a list of all expected cashflows (final notional and coupons) together with their payment date.

        Returns:
            _List[Tuple[datetime, float]]: The resulting list of all cashflows.
        """
        # if self.coupon_freq != 'Y':
        #    raise Exception('Cannot calc cashflows for other than yearly coupons. Missing transformation from yearly coupon to .... ')

        # assume for this deposits it is like a short term zero-coupon bond
        # follwoing N*(1+r*t), N: notional, r:rate, t:period, ie.e yearfrac between start and end date

        # Adjust maturity date
        # assumption is ECB holiday schedule... from holiday module...
        adjusted_maturity = adjust_date(self.maturity_date, self.business_day_convention, _ECB)
        # TODO needs verification
        period_yf = day_count_fraction(self.start_date, adjusted_maturity, self.day_count_convention)
        # TODO needs verification

        interest = self.notional * self.rate * period_yf

        result = [
            (self.start_date, 0.0)
        ]  # the first entry of this schedule is the accrual start which has a cashflow of zero and is just used for accrual calculation
        result.append((self.maturity_date, interest + self.notional))

        # schedule = Schedule(self.accrual_start, self.maturity_date, period, stub=self.stub).generate_dates(ends_only=True)
        # result = [(d, self.coupon*coupon_multiplier*self.notional) for d in schedule]
        # result.insert(0, (self.accrual_start, 0.0))# the first entry of this schedule is the accrual start which has a cashflow of zero and is just used for accrual calculation
        # result.append((self.maturity_date, self.notional))
        return result

    @property
    def coupon_payment_dates(self) -> _List[date]:
        """
        Getter for payment dates for fixed coupons.

        Returns:
            List[date]: List of dates for fixed coupon payments.
        """
        return self.__coupon_payment_dates

    @property
    def coupons(self) -> _List[float]:
        """
        Getter for fixed coupon payments.

        Returns:
            List[float]: List of coupon amounts expressed as annualised fractions of deposit's face value.
        """
        return self.__coupons

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
