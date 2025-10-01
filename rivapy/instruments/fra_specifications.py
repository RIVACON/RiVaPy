from abc import abstractmethod as _abstractmethod
from typing import List as _List, Union as _Union, Tuple, Optional as _Optional
import numpy as np
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from holidays import HolidayBase as _HolidayBase, ECB as _ECB
from rivapy.instruments.components import Issuer
from rivapy.tools.datetools import (
    Period,
    Schedule,
    _date_to_datetime,
    _datetime_to_date_list,
    _term_to_period,
    roll_day,
    calc_start_day,
    serialize_date,
)
from rivapy.tools.enums import (
    DayCounterType,
    RollConvention,
    SecuritizationLevel,
    Currency,
    Rating,
    Instrument,
    InterestRateIndex,
    get_index_by_alias,
)
from rivapy.tools._validators import _check_positivity, _check_start_before_end, _string_to_calendar, _is_ascending_date_list
import rivapy.tools.interfaces as interfaces


class ForwardRateAgreementSpecification(interfaces.FactoryObject):

    def __init__(
        self,
        obj_id: str,
        trade_date: _Union[date, datetime],
        notional: float,
        rate: float,
        start_date: _Union[date, datetime],
        end_date: _Union[date, datetime],
        udlID: str,
        rate_start_date: _Union[date, datetime],
        rate_end_date: _Union[date, datetime],
        maturity_date: _Union[date, datetime] = None,
        day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
        business_day_convention: _Union[RollConvention, str] = RollConvention.FOLLOWING,
        rate_day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
        rate_business_day_convention: _Union[RollConvention, str] = RollConvention.FOLLOWING,
        calendar: _Union[_HolidayBase, str] = None,
        currency: _Union[Currency, str] = "EUR",
        # ex_settle: int =0,
        payment_days: int = 0,
        spot_days: int = 2,
        start_period: int = None,
        # _Optional[_Union[Period, str]] = None,
        end_period: int = None,
        index_alias: str = None,
        issuer: _Optional[_Union[Issuer, str]] = None,
        securitization_level: _Union[SecuritizationLevel, str] = SecuritizationLevel.NONE,
        rating: _Union[Rating, str] = Rating.NONE,
    ):
        """Constructor for Forward Rate Agreement specification.

        Args:
            obj_id (str): (Preferably) Unique label of the FRA
            trade_date (_Union[date, datetime]): FRA Trade date.
            maturity_date (_Union[date, datetime]): FRA's maturity/expiry date. Must lie after the trade_date.
            notional (float, optional): Fra's notional/face value. Must be positive.
            rate (float): Agreed upon forward rate, a.k.a. FRA rate.
            start_date (_Union[date, datetime]): start date of the interest rate (FRA_rate) reference period from which interest is accrued.
            end_date (_Union[date, datetime]): end date of the interest rate (FRA_rate) reference period from which interest is accrued.
            udlID (str): ID of the underlying Index rate used for the floating rate for fixing.
            rate_start_date (_Union[date, datetime]): start date of fixing period for the floating rate
            rate_end_date (_Union[date, datetime]): end date of fixing period for the floating rate
            day_count_convention (Union[DayCounter, str], optional): Day count convention for determining period
                                                                     length. Defaults to DayCounter.ThirtyU360.
            business_day_convention (Union[RollConvention, str], optional): Set of rules defining the adjustment of
                                                                            days to ensure each date being a business
                                                                            day with respect to a given holiday
                                                                            calendar. Defaults to
                                                                            RollConvention.FOLLOWING
            rate_day_count_convention (Union[DayCounter, str], optional): Day count convention for determining period
                                                                     length. Defaults to DayCounter.ThirtyU360.
            rate_business_day_convention (Union[RollConvention, str], optional): Set of rules defining the adjustment of
                                                                            days to ensure each date being a business
                                                                            day with respect to a given holiday
                                                                            calendar. Defaults to
                                                                            RollConvention.FOLLOWING
            calendar (Union[HolidayBase, str], optional): Holiday calendar defining the bank holidays of a country or
                                                          province (but not all non-business days as for example
                                                          Saturdays and Sundays).
                                                          Defaults (through constructor) to holidays.ECB
                                                          (= Target2 calendar) between start_day and end_day.
            currency (str, optional): Currency as alphabetic, Defaults to 'EUR'.
            payment_days (int): Number of days for payment after the start date. Defaults to 0.
            spot_days (int): time difference between fixing date and start dategiven in days.
            start_period (int): forward start period given in months e.g. 1 from 1Mx4M
            end_period (int): forward end period given in months e.g. 4 from 1Mx4M
            index_alias (str): ID of the underlying Index rate used for the floating rate for fixing.
            issuer (str, optional): Name/id of issuer. Defaults to None.
            securitization_level (_Union[SecuritizationLevel, str], optional): Securitization level. Defaults to None.
            rating (_Union[Rating, str]): Paper rating.
        """
        # positional arguments
        self.obj_id = obj_id
        self._trade_date = trade_date
        self._notional = notional
        self._rate = rate
        self._start_date = start_date
        self._end_date = end_date
        if maturity_date is None:
            self._maturity_date = self._end_date
        else:
            self._maturity_date = maturity_date
        self._udlID = udlID
        self._rate_start_date = rate_start_date
        self._rate_end_date = rate_end_date
        # optional arguments
        self._day_count_convention = day_count_convention  # TODO: correct syntax with setter?? HN
        self._business_day_convention = RollConvention.to_string(business_day_convention)
        self._rate_day_count_convention = rate_day_count_convention
        self._rate_business_day_convention = RollConvention.to_string(rate_business_day_convention)
        if calendar is None:
            self._calendar = _ECB(years=range(trade_date.year, end_date.year + 1))
        else:
            self._calendar = _string_to_calendar(calendar)
        self._currency = currency
        # self.ex_settle = ex_settle
        # self.trade_settle = trade_settle
        self._fixing_date = calc_start_day(self._start_date, f"{spot_days}D", self._business_day_convention, self._calendar)
        self._spot_days = spot_days

        # if start_period is not None:
        self.start_period = start_period

        # if end_period is not None:
        self.end_period = end_period

        if index_alias is not None:
            self._index_alias = index_alias
            self._index = get_index_by_alias(index_alias)
            self._indexdata = self._index.value
        if issuer is not None:
            self._issuer = issuer
        if securitization_level is not None:
            self.securitization_level = securitization_level
        self._rating = Rating.to_string(rating)
        self._payment_days = payment_days

        # give dates where applicable as optional, if not given, calculate based on spot lag, index spot lag, and forward period YMxZM (e.g. 1Mx4M)
        # e.g. for trade date D1 and spotLag, S1, and start_period = 1Mx4M
        # start_date = D1 + S1 + 1Month # this is the date it starts accruing interest
        # but how much interest? -> the pre agreed FRA rate, fixed
        # how is it settled? -> at settledate=start date, and using
        # The floating rate index (e.g., LIBOR, SOFR, EURIBOR) used to determine the settlement amoun
        # This is determined at the fixing_date ( usually spot lag before, e.g. 2 days)

        # if trade date, spotlag, startperiod,endperiod give, then recalcualte start_datet etc...
        # TODO: get clarification on roll_day function
        if trade_date and spot_days and start_period and end_period:
            spot_date = roll_day(
                day=trade_date + timedelta(days=spot_days),  # need holiday
                calendar=self.calendar,
                business_day_convention=self.rate_business_day_convention,
                start_day=None,
            )

            self._start_date = roll_day(
                day=spot_date + relativedelta(months=start_period),  # need holiday
                calendar=self.calendar,
                business_day_convention=self.rate_business_day_convention,
                start_day=None,
            )  # spot_date + start_period #need roll convention: ddc, bdc, holiday, date
            self._end_date = roll_day(
                day=self.start_date + relativedelta(months=end_period - start_period),  # need holiday
                calendar=self.calendar,
                business_day_convention=self.rate_business_day_convention,
                start_day=None,
            )  # start_date + end_period #need roll convention: ddc, bdc, holiday, date

        # VALIDATE DATES
        # TODO:         self._validate_derived_issued_instrument()

    @staticmethod
    def _create_sample(
        n_samples: int, seed: int = None, ref_date=None, issuers: _List[str] = None, sec_levels: _List[str] = None, currencies: _List[str] = None
    ) -> _List["ForwardRateAgreementSpecification"]:
        """Create a random sample of multiple instruments of this type with varied specification parameters.

        Args:
            n_samples (int): The number of desired sample objects
            seed (int, optional): Seed number to allow repeated result. Defaults to None.
            ref_date (_type_, optional): Reference date . Defaults to None.
            issuers (_List[str], optional): list of issuers. Defaults to None.
            sec_levels (_List[str], optional): list of possible securitization levels. Defaults to None.
            currencies (_List[str], optional): list of possible currencies used. Defaults to None.

        Returns:
            _List[ForwardRateAgreementSpecification]: where each entry is a dict representing with the information needed to specify an instrument.
        """
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
        for i in range(n_samples):
            days = int(15.0 * 365.0 * np.random.beta(2.0, 2.0)) + 1
            trade_date = ref_date + timedelta(days=np.random.randint(low=-365, high=0))
            maturity_date = ref_date + timedelta(days=days)
            start_date = ref_date + relativedelta(months=np.random.randint(low=1, high=3))
            end_date = start_date + relativedelta(months=np.random.choice([3, 6]))
            # spot_days=2, fixing pre_lag =2
            result.append(
                ForwardRateAgreementSpecification(
                    obj_id=f"Deposit_{i}",
                    trade_date=trade_date,
                    maturity_date=maturity_date,
                    notional=np.random.choice([100.0, 1000.0, 10_000.0, 100_0000.0]),
                    rate=np.random.choice([0.01, 0.02, 0.03, 0.04, 0.05]),
                    start_date=start_date,
                    end_date=end_date,
                    udlID="dummy_underlying_index",
                    rate_start_date=start_date - timedelta(days=2),
                    rate_end_date=end_date - timedelta(days=2),
                    # "day_count_convention": self.day_count_convention, #TODO
                    # "business_day_convention": self.business_day_convention,
                    # "rate_day_count_convention": self.rate_day_count_convention,
                    # "rate_business_day_convention": self.rate_business_day_convention,
                    calendar=_ECB(years=range(trade_date.year, maturity_date.year + 1)),
                    currency=np.random.choice(currencies),
                    # "spot_days": self.spot_days, # not needed if start dates given
                    # "start_period": self.start_period,
                    # "end_period": self.end_period,
                    issuer=np.random.choice(issuers),
                    securitization_level=np.random.choice(sec_levels),
                )
            )
        return result

    def _validate_derived_issued_instrument(self):
        self.__trade_date, self.__maturity_date = _check_start_before_end(self.__trade_date, self.__maturity_date)

    def _to_dict(self) -> dict:

        result = {
            "obj_id": self.obj_id,
            "trade_date": serialize_date(self.trade_date),
            "maturity_date": serialize_date(self.maturity_date),
            "notional": self.notional,
            "rate": self.rate,
            "start_date": serialize_date(self.start_date),
            "end_date": serialize_date(self.end_date),
            "udlID": self.udlID,
            "rate_start_date": serialize_date(self.rate_start_date),
            "rate_end_date": serialize_date(self.rate_end_date),
            "day_count_convention": self.day_count_convention,
            "business_day_convention": self.business_day_convention,
            "rate_day_count_convention": self.rate_day_count_convention,
            "rate_business_day_convention": self.rate_business_day_convention,
            "calendar": getattr(self.calendar, "name", self.calendar.__class__.__name__),
            "currency": self.currency,
            "payment_days": self.payment_days,
            "spot_days": self.spot_days,
            "start_period": self.start_period,
            "end_period": self.end_period,
            "issuer": self.issuer,
            "securitization_level": self.securitization_level,
            "rating": self.rating,
        }
        return result

    def get_schedule(self) -> Schedule:
        """Returns the schedule of the accrual periods of the instrument."""
        return Schedule(
            start_day=self._start_date,
            end_day=self._end_date,
            time_period=self._frequency,
            backwards=True,
            stub_type_is_Long=True,
            business_day_convention=self.business_day_convention,
            roll_convention=None,
            calendar=self._calendar,
        )
        # region properties

    @property
    def calendar(self):
        """Calender used for this instrument

        Returns:
            _type_: _description_
        """
        return self._calendar

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
    def udlID(self) -> str:
        """
        Getter for ID of the instruments underlying.

        Returns:
            str: Instrument's udlID.
        """
        return self._udlID

    @udlID.setter
    def udlID(self, udlID: str):
        """
        Setter for ID of the instruments underlying.

        Args:
            udlID(str): udlID of the instrument.
        """
        self._udlID = udlID

    @property
    def rating(self) -> str:
        """Getter for instrument's rating.

        Returns:
            str: instrument's rating
        """
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
        return self._securitization_level

    @securitization_level.setter
    def securitization_level(self, securitisation_level: _Union[SecuritizationLevel, str]):
        self._securitization_level = SecuritizationLevel.to_string(securitisation_level)

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
            (float): interest rate of the instrument.
        """
        self._rate = rate

    @property
    def trade_date(self) -> date:
        """
        Getter for FRA's issue date.

        Returns:
            date: FRA's issue date.
        """
        return self._trade_date

    @trade_date.setter
    def trade_date(self, trade_date: _Union[datetime, date]):
        """
        Setter for FRA's issue date.

        Args:
            issue (Union[datetime, date]): FRA's issue date.
        """
        self._trade_date = _date_to_datetime(trade_date)

    @property
    def start_date(self) -> date:
        """
        Getter for FRA's start date.

        Returns:
            date: FRA's start date.
        """
        return self._start_date

    @start_date.setter
    def start_date(self, start_date: _Union[datetime, date]):
        """
        Setter for FRA's start date.

        Args:
            start_date (Union[datetime, date]): FRA's start date.
        """
        self._start_date = _date_to_datetime(start_date)

    @property
    def end_date(self) -> date:
        """
        Getter for FRA's end date.

        Returns:
            date: FRA's end date.
        """
        return self._end_date

    @end_date.setter
    def end_date(self, end_date: _Union[datetime, date]):
        """
        Setter for FRA's end date.

        Args:
            end_date (Union[datetime, date]): FRA's end date.
        """
        self._end_date = _date_to_datetime(end_date)

    @property
    def rate_start_date(self) -> date:
        """
        Getter for FRA's underlying rate start date.

        Returns:
            date: FRA's underlaying rate start date.
        """
        return self._rate_start_date

    @start_date.setter
    def rate_start_date(self, start_date: _Union[datetime, date]):
        """
        Setter for FRA's underlying rate start date.

        Args:
            start_date (Union[datetime, date]): FRA's underlaying rate start date.
        """
        self._rate_start_date = _date_to_datetime(start_date)

    @property
    def rate_end_date(self) -> date:
        """
        Getter for FRA's underlying rate end date.

        Returns:
            date: FRA's underlying rate end date.
        """
        return self._rate_end_date

    @end_date.setter
    def rate_end_date(self, end_date: _Union[datetime, date]):
        """
        Setter for FRA's underlying rate end date.

        Args:
            end_date (Union[datetime, date]): FRA's underlying rate end date.
        """
        self._rate_end_date = _date_to_datetime(end_date)

    @property
    def maturity_date(self) -> date:
        """
        Getter for FRA's maturity date.

        Returns:
            date: FRA's maturity date.
        """
        return self._maturity_date

    @maturity_date.setter
    def maturity_date(self, maturity_date: _Union[datetime, date]):
        """
        Setter for FRA's maturity date.

        Args:
            maturity_date (Union[datetime, date]): FRA's maturity date.
        """
        self._maturity_date = _date_to_datetime(maturity_date)

    @property
    def currency(self) -> str:
        """
        Getter for FRA's currency.

        Returns:
            str: FRA's  currency code
        """
        return self._currency

    @currency.setter
    def currency(self, currency: str):
        self._currency = Currency.to_string(currency)

    @property
    def notional(self) -> float:
        """
        Getter for FRA's face value.

        Returns:
            float: FRA's face value.
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
    def day_count_convention(self, day_count_convention: _Union[DayCounterType, str]) -> str:
        self._day_count_convention = DayCounterType.to_string(day_count_convention)

    @property
    def rate_day_count_convention(self) -> str:
        """
        Getter for FRA's underlying rate's day count convention.

        Returns:
            str: FRA's underlying rate's day count convention.
        """
        return self._rate_day_count_convention

    @rate_day_count_convention.setter
    def rate_day_count_convention(self, rate_day_count_convention: _Union[DayCounterType, str]) -> str:
        self._rate_day_count_convention = DayCounterType.to_string(rate_day_count_convention)

    @property
    def business_day_convention(self) -> str:
        """
        Getter for FRA's underlying rate's business_day_convention.

        Returns:
            str: FRA's underlying rate's business_day_convention.
        """
        return self._business_day_convention

    @business_day_convention.setter
    def business_day_convention(self, business_day_convention: _Union[DayCounterType, str]) -> str:
        self._business_day_convention = DayCounterType.to_string(business_day_convention)

    @property
    def rate_business_day_convention(self) -> str:
        """
        Getter for FRA's underlying rate's business_day_convention.

        Returns:
            str: FRA's underlying rate's business_day_convention.
        """
        return self._rate_business_day_convention

    @rate_business_day_convention.setter
    def rate_business_day_convention(self, business_day_convention: _Union[DayCounterType, str]) -> str:
        """Setter for FRA's underlying rate's business_day_convention."""
        self._rate_business_day_convention = DayCounterType.to_string(business_day_convention)

    @property
    def spot_days(self) -> int:
        """Getter for the spot lag given in days

        Returns:
            float: _description_
        """
        return self._spot_days

    @property
    def start_period(self) -> int:
        """Getter for the start period, given in Months

        Returns:
            float: _description_
        """
        return self._start_period

    @start_period.setter
    def start_period(self, start_period) -> int:
        """setter for the start period, given in Months

        Returns:
            float: _description_
        """
        self._start_period = start_period

    @property
    def end_period(self) -> int:
        """Getter for the spot lag

        Returns:
            float: _description_
        """
        return self._end_period

    @end_period.setter
    def end_period(self, end_period) -> int:
        """setter for the end period, given in Months

        Returns:
            float: _description_
        """
        self._end_period = end_period

    @property
    def index(self) -> str:
        """Getter for the underlying Index rate used for the floating rate for fixing.

        Returns:
            str: _description_
        """
        return self._index

    @index.setter
    def index(self, index: str):
        self._index = index

    def ins_type(self):
        """Return instrument type

        Returns:
            Instrument: Forward rate agreement
        """
        return Instrument.FRA

    @property
    def payment_days(self) -> int:
        """Getter for the number of settlement days.

        Returns:
            int: Number of settlement days.
        """
        return self._payment_days

    @payment_days.setter
    def payment_days(self, payment_days: int):
        self._payment_days = payment_days

    # temp placeholder
    def get_end_date(self):
        return self.maturity_date

    # endregion
