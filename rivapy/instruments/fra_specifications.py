from abc import abstractmethod as _abstractmethod
from typing import List as _List, Union as _Union, Tuple
import numpy as np
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from holidays import HolidayBase as _HolidayBase, ECB as _ECB
from rivapy.tools.datetools import Period, Schedule, _date_to_datetime, _datetime_to_date_list, _term_to_period
from rivapy.tools.enums import DayCounterType, RollConvention, SecuritizationLevel, Currency, Rating
from rivapy.tools._validators import _check_positivity, _check_start_before_end, _string_to_calendar, _is_ascending_date_list
import rivapy.tools.interfaces as interfaces
from rivapy.tools.datetools import Period, Schedule, roll_day


class ForwardRateAgreementSpecification(interfaces.FactoryObject):

    def __init__(
        self,
        obj_id: str,
        issue_date: _Union[date, datetime],
        maturity_date: _Union[date, datetime],
        notional: float,
        rate: float,
        start_date: _Union[date, datetime],
        end_date: _Union[date, datetime],
        udlID: str,
        rate_start_date: _Union[date, datetime],
        rate_end_date: _Union[date, datetime],
        day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
        business_day_convention: _Union[RollConvention, str] = RollConvention.FOLLOWING,
        rate_day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
        rate_business_day_convention: _Union[RollConvention, str] = RollConvention.FOLLOWING,
        calendar: _Union[_HolidayBase, str] = None,
        currency: _Union[Currency, str] = "EUR",
        # ex_settle: int =0,
        # trade_settle: int= 0,
        spot_lag: int = None,
        start_period: int = None,
        end_period: int = None,
        issuer: str = None,
        securitization_level: _Union[SecuritizationLevel, str] = SecuritizationLevel.NONE,
        rating: _Union[Rating, str] = Rating.NONE,
    ):
        """Constructor for Forward Rate Agreement specification.

        Args:
            obj_id (str): (Preferably) Unique label of the FRA
            issue_date (_Union[date, datetime]): FRA Trade date.
            maturity_date (_Union[date, datetime]): FRA's maturity/expiry date. Must lie after the issue_date.
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
            spot_lag (int): time difference between issue/trade date and spot_date given in days.
            start_period (int): forward start period given in months e.g. 1 from 1Mx4M
            end_period (int): forward end period given in months e.g. 4 from 1Mx4M
            issuer (str, optional): Name/id of issuer. Defaults to None.
            securitization_level (_Union[SecuritizationLevel, str], optional): Securitization level. Defaults to None.
            rating (_Union[Rating, str]): Paper rating.
        """
        # positional arguments
        self.obj_id = obj_id
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.notional = notional
        self.rate = rate
        self.start_date = start_date
        self.end_date = end_date
        self.udlID = udlID
        self.rate_start_date = rate_start_date
        self.rate_end_date = rate_end_date

        # optional arguments
        self.day_count_convention = day_count_convention  # TODO: correct syntax with setter?? HN
        self.business_day_convention = RollConvention.to_string(business_day_convention)
        self.rate_day_count_convention = rate_day_count_convention
        self.rate_business_day_convention = RollConvention.to_string(rate_business_day_convention)
        if calendar is None:
            self.calendar = _ECB(years=range(issue_date.year, maturity_date.year + 1))
        else:
            self.calendar = _string_to_calendar(calendar)
        self.currency = currency
        # self.ex_settle = ex_settle
        # self.trade_settle = trade_settle
        if spot_lag is not None:
            self.spot_lag = spot_lag
        if start_period is not None:
            self.start_period = start_period
        if end_period is not None:
            self.end_period = end_period
        if issuer is not None:
            self.issuer = issuer
        if securitization_level is not None:
            self.securitization_level = securitization_level
        self.rating = Rating.to_string(rating)

        # give dates where applicable as optional, if not given, calculate based on spot lag, index spot lag, and forward period YMxZM (e.g. 1Mx4M)
        # e.g. for trade date D1 and spotLag, S1, and start_period = 1Mx4M
        # start_date = D1 + S1 + 1Month # this is the date it starts accruing interest
        # but how must interest? the pre agreed FRA rate, fixed
        # how is it settled? at settledate=start date, and using
        # The floating rate index (e.g., LIBOR, SOFR, EURIBOR) used to determine the settlement amoun
        # This is determined at the fixing_date ( usually spot lag before, e.g. 2 days)

        # if trade date, spotlag, startperiod,endperiod give, then recalcualte start_datet etc...
        # TODO: get clarification on roll_day function
        if issue_date and spot_lag and start_period and end_period:
            spot_date = roll_day(
                day=issue_date + timedelta(days=spot_lag),  # need holiday
                calendar=self.calendar,
                business_day_convention=self.rate_business_day_convention,
                start_day=None,
            )

            self.start_date = roll_day(
                day=spot_date + relativedelta(months=start_period),  # need holiday
                calendar=self.calendar,
                business_day_convention=self.rate_business_day_convention,
                start_day=None,
            )  # spot_date + start_period #need roll convention: ddc, bdc, holiday, date
            self.end_date = roll_day(
                day=self.start_date + relativedelta(months=start_period),  # need holiday
                calendar=self.calendar,
                business_day_convention=self.rate_business_day_convention,
                start_day=None,
            )  # start_date + end_period #need roll convention: ddc, bdc, holiday, date

        # VALIDATE DATES
        # TODO:         self._validate_derived_issued_instrument()

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
            maturity_date = ref_date + timedelta(days=days)
            start_date = ref_date + relativedelta(months=np.random.randint(low=1, high=3))
            end_date = start_date + relativedelta(months=np.random.choice([3, 6]))
            # spot_lag=2, fixing pre_lag =2
            result.append(
                {
                    "issue_date": issue_date,
                    "maturity_date": maturity_date,
                    "notional": np.random.choice([100.0, 1000.0, 10_000.0, 100_0000.0]),
                    "rate": np.random.choice([0.01, 0.02, 0.03, 0.04, 0.05]),
                    "start_date": start_date,
                    "end_date": end_date,
                    "udlID": "dummy_underlying_index",  #
                    "rate_start_date": start_date - timedelta(days=2),  # does not account for roll convention ...
                    "rate_end_date": end_date - timedelta(days=2),
                    # "day_count_convention": self.day_count_convention, #TODO
                    # "business_day_convention": self.business_day_convention,
                    # "rate_day_count_convention": self.rate_day_count_convention,
                    # "rate_business_day_convention": self.rate_business_day_convention,
                    "calendar": _ECB(years=range(issue_date.year, maturity_date.year + 1)),
                    "currency": np.random.choice(currencies),
                    # "spot_lag": self.spot_lag, # not needed if start dates given
                    # "start_period": self.start_period,
                    # "end_period": self.end_period,
                    "issuer": np.random.choice(issuers),
                    "securitization_level": np.random.choice(sec_levels),
                }
            )
        return result

    def _validate_derived_issued_instrument(self):
        self.__issue_date, self.__maturity_date = _check_start_before_end(self.__issue_date, self.__maturity_date)

    def _to_dict(self) -> dict:
        result = {
            "obj_id": self.obj_id,
            "issue_date": self.issue_date,
            "maturity_date": self.maturity_date,
            "notional": self.notional,
            "rate": self.rate,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "udlID": self.udlID,
            "rate_start_date": self.rate_start_date,
            "rate_end_date": self.rate_end_date,
            "day_count_convention": self.day_count_convention,
            "business_day_convention": self.business_day_convention,
            "rate_day_count_convention": self.rate_day_count_convention,
            "rate_business_day_convention": self.rate_business_day_convention,
            "calendar": self.calendar,
            "currency": self.currency,
            "spot_lag": self.spot_lag,
            "start_period": self.start_period,
            "end_period": self.end_period,
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
        return self.__issuer

    @issuer.setter
    def issuer(self, issuer: str):
        """
        Setter for instrument's issuer.

        Args:
            issuer(str): Issuer of the instrument.
        """
        self.__issuer = issuer

    @property
    def rating(self) -> str:
        return self.__rating

    @rating.setter
    def rating(self, rating: _Union[Rating, str]) -> str:
        self.__rating = Rating.to_string(rating)

    @property
    def securitization_level(self) -> str:
        """
        Getter for instrument's securitisation level.

        Returns:
            str: Instrument's securitisation level.
        """
        return self.__securitization_level

    @securitization_level.setter
    def securitization_level(self, securitisation_level: _Union[SecuritizationLevel, str]):
        self.__securitization_level = SecuritizationLevel.to_string(securitisation_level)

    @property
    def issue_date(self) -> date:
        """
        Getter for FRA's issue date.

        Returns:
            date: FRA's issue date.
        """
        return self.__issue_date

    @issue_date.setter
    def issue_date(self, issue_date: _Union[datetime, date]):
        """
        Setter for FRA's issue date.

        Args:
            issue_date (Union[datetime, date]): FRA's issue date.
        """
        self.__issue_date = _date_to_datetime(issue_date)

    @property
    def maturity_date(self) -> date:
        """
        Getter for FRA's maturity date.

        Returns:
            date: FRA's maturity date.
        """
        return self.__maturity_date

    @maturity_date.setter
    def maturity_date(self, maturity_date: _Union[datetime, date]):
        """
        Setter for FRA's maturity date.

        Args:
            maturity_date (Union[datetime, date]): FRA's maturity date.
        """
        self.__maturity_date = _date_to_datetime(maturity_date)

    @property
    def currency(self) -> str:
        """
        Getter for FRA's currency.

        Returns:
            str: FRA's  currency code
        """
        return self.__currency

    @currency.setter
    def currency(self, currency: str):
        self.__currency = Currency.to_string(currency)

    @property
    def notional(self) -> float:
        """
        Getter for FRA's face value.

        Returns:
            float: FRA's face value.
        """
        return self.__notional

    @notional.setter
    def notional(self, notional):
        self.__notional = _check_positivity(notional)

    @property
    def daycount_convention(self) -> str:
        """
        Getter for FRA's day count convention.

        Returns:
            str: FRA's day count convention.
        """
        return self.__day_count_convention

    @daycount_convention.setter
    def daycount_convention(self, day_count_convention: _Union[DayCounterType, str]) -> str:
        self.__day_count_convention = DayCounterType.to_string(day_count_convention)

    @property
    def rate_daycount_convention(self) -> str:
        """
        Getter for FRA's day count convention.

        Returns:
            str: FRA's day count convention.
        """
        return self.__rate_day_count_convention

    @rate_daycount_convention.setter
    def rate_daycount_convention(self, rate_day_count_convention: _Union[DayCounterType, str]) -> str:
        self.__rate_day_count_convention = DayCounterType.to_string(rate_day_count_convention)
