# -*- coding: utf-8 -*-
import re
from typing import Callable, Dict, Any, Optional as _Optional
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from dateutil.rrule import WE
from calendar import monthrange, isleap
from typing import List as _List, Union as _Union, Callable
from holidays import HolidayBase as _HolidayBase

# from holidays import DE
from holidays.financial.european_central_bank import ECB as _ECB
from rivapy.tools.enums import RollConvention, DayCounterType, RollRule
from rivapy.tools._validators import _string_to_calendar


# TODO: Switch to locally configured logger.
from rivapy.tools._logger import logger


class DayCounter:

    def __init__(self, daycounter: _Union[str, DayCounterType]):
        self._dc = DayCounterType.to_string(daycounter)
        self._yf = DayCounter.get(self._dc)

    def yf(
        self,
        d1: _Union[date, datetime],
        d2: _Union[_Union[date, datetime], _List[_Union[date, datetime]]],
        coupon_schedule: _List[_Union[date, datetime]] = None,  # Added optional argument
        coupon_frequency: float = None,  # Added optional argument
    ) -> _Union[float, _List[float]]:

        if self._dc == DayCounterType.ActActICMA.value:
            if coupon_schedule is None or coupon_frequency is None:
                raise ValueError("For ActActICMA, 'coupon_schedule' and 'coupon_frequency' must be provided.")
            if isinstance(d2, list):
                return [self._yf(d1, d2_, coupon_schedule, coupon_frequency) for d2_ in d2]
            else:
                return self._yf(d1, d2, coupon_schedule, coupon_frequency)
        else:
            if isinstance(d2, list):
                return [self._yf(d1, d2_) for d2_ in d2]
            else:
                return self._yf(d1, d2)

    @staticmethod
    def get(daycounter: _Union[str, DayCounterType]) -> Callable[[_Union[date, datetime], _Union[date, datetime]], float]:
        dc = DayCounterType.to_string(daycounter)

        mapping = {
            DayCounterType.Act365Fixed.value: DayCounter.yf_Act365Fixed,
            DayCounterType.ACT_ACT.value: DayCounter.yf_ActAct,
            DayCounterType.ACT360.value: DayCounter.yf_Act360,
            DayCounterType.ThirtyU360.value: DayCounter.yf_30U360,
            DayCounterType.ThirtyE360.value: DayCounter.yf_30E360,
            DayCounterType.Thirty360ISDA.value: DayCounter.yf_30360ISDA,
            DayCounterType.ActActICMA.value: DayCounter.yf_ActActICMA,
        }

        if dc in mapping:
            return mapping[dc]
        else:
            raise NotImplementedError(f"{dc} not yet implemented.")

    @staticmethod
    def yf_ActActICMA(
        d1: _Union[date, datetime], d2: _Union[date, datetime], coupon_schedule: _List[_Union[date, datetime]], coupon_frequency: _Union[int, float]
    ) -> float:
        """This method implements the Act/Act ICMA day count convention which is used for Bonds.

        Args:
            d1 (_Union[date, datetime]): start date of the period for which the year fraction is calculated.
            d2 (_Union[date, datetime]): end date of the period for which the year fraction is calculated.
            coupon_schedule (_List[_Union[date, datetime]]): Sorted list of all coupon payment days.
            coupon_frequency (int): Number of coupon payments per year (e.g., 1 for annual, 2 for semi-annual)

        Returns:
            float: year fraction
        """
        if coupon_frequency == 0:
            raise ValueError("Coupon frequency must be greater than 0.")
        d1_dt = _date_to_datetime(d1)
        d2_dt = _date_to_datetime(d2)
        coupon_schedule_dt = [_date_to_datetime(cs_date) for cs_date in coupon_schedule]

        yf = 0.0
        for i in range(len(coupon_schedule_dt) - 1):
            cp_start_dt = coupon_schedule_dt[i]
            cp_end_dt = coupon_schedule_dt[i + 1]

            # consider overlapping periods only
            if d1_dt <= cp_end_dt and d2_dt >= cp_start_dt:
                fraction_period_start_dt = max(d1_dt, cp_start_dt)
                fraction_period_end_dt = min(d2_dt, cp_end_dt)

                days_cp = (cp_end_dt - cp_start_dt).days
                days_fraction = (fraction_period_end_dt - fraction_period_start_dt).days

                yf += days_fraction / (days_cp * coupon_frequency)

        return yf

    @staticmethod
    def yf_Act365Fixed(d1: _Union[date, datetime], d2: _Union[date, datetime]) -> float:
        """This method implements the Act/365f day count convention.
        The actual number of days between d2 and d1 is divided by 365.

        Args:
            d1 (_Union[date, datetime]): start date
            d2 (_Union[date, datetime]): end date

        Returns:
            float: year fraction
        """
        d1_dt = _date_to_datetime(d1)
        d2_dt = _date_to_datetime(d2)
        return (d2_dt - d1_dt).total_seconds() / (365.0 * 24 * 60 * 60)

    @staticmethod
    def yf_ActAct(d1: _Union[date, datetime], d2: _Union[date, datetime]) -> float:
        """This method implements the Act/Act ISDA day count convention.
        The acutal number of days between d2 and d1 is divded by the acutal number of days in the respective year.
        In cases where d2 and d1 are located in different years, the period is split into sub periods and the year fraction is calculated on each sub period with its respective
        number of days in that year. This is especially important if d1 is located in a regular year and d2 is located in a leap year.

        Args:
            d1 (_Union[date, datetime]): start date
            d2 (_Union[date, datetime]): end date

        Returns:
            float: year fraction
        """
        d1_dt = _date_to_datetime(d1)
        d2_dt = _date_to_datetime(d2)

        if d1_dt > d2_dt:
            raise ValueError("d1 must be before d2")

        # Calculate the fraction for each year the period spans
        current_date_dt = d1_dt
        year_fraction = 0.0

        while current_date_dt < d2_dt:
            # Ensure year_end_dt and start_of_year_dt are datetime, preserving tzinfo if present
            year_end_dt = datetime(current_date_dt.year, 12, 31, tzinfo=current_date_dt.tzinfo)
            start_of_year_dt = datetime(current_date_dt.year, 1, 1, tzinfo=current_date_dt.tzinfo)
            days_in_year = (year_end_dt - start_of_year_dt).days + 1  # Actual days in the year

            # If the period ends within the same year
            if d2_dt.year == current_date_dt.year:
                year_fraction += (d2_dt - current_date_dt).days / days_in_year
                break

            # Add the fraction for the remaining days in the current year
            year_fraction += ((year_end_dt - current_date_dt).days + 1) / days_in_year
            # Move to the start of the next year
            current_date_dt = datetime(current_date_dt.year + 1, 1, 1, tzinfo=current_date_dt.tzinfo)

        return year_fraction

    @staticmethod
    def yf_Act360(d1: _Union[date, datetime], d2: _Union[date, datetime]) -> float:
        """This method implements the Act/360 day count convention.
        Here the actual number of days between d2 and d1 is computed and divided by 360, since this day count convention assumes that each year contains 360 days.

        Args:
            d1 (_Union[date, datetime]): start date
            d2 (_Union[date, datetime]): end date

        Returns:
            float: _description_
        """
        d1_dt = _date_to_datetime(d1)
        d2_dt = _date_to_datetime(d2)
        return ((d2_dt - d1_dt).days) / 360.0  # Ensure float division

    # @staticmethod
    # def yf_Bus252(d1: _Union[date, datetime], d2: _Union[date, datetime])->float:
    #     """This method implements the Bus/252 day count convention.

    #     Args:
    #         d1 (_Union[date, datetime]): start date
    #         d2 (_Union[date, datetime]): end date

    #     Returns:
    #         float: _description_
    #     """
    #     return ((d2 - d1).days)/252

    @staticmethod
    def yf_30U360(d1: _Union[date, datetime], d2: _Union[date, datetime]) -> float:
        """This method implements the 30U360 convention.
        The following logic is applied:

        1. If d2.day == 31 and d1.day >= 30 -> d2.day = 30
        2. If d1.day == 31 -> d1.day = 30
        3. If (d1.day == EndOfMonth(Feb) and d1.month==2) and (d2.day == EndOfMonth(Feb) and d2.month==2) -> d2.day = 30
        4. If (d1.day == EndOfMonth(Feb) and d1.month==2) -> d1.day = 30

        Args:
            d1 (_Union[date, datetime]): start date
            d2 (_Union[date, datetime]): end date

        Returns:
            float: year fraction
        """
        d1_dt = _date_to_datetime(d1)
        d2_dt = _date_to_datetime(d2)

        m_range1 = monthrange(d1_dt.year, d1_dt.month)
        m_range2 = monthrange(d2_dt.year, d2_dt.month)

        day1 = d1_dt.day
        day2 = d2_dt.day

        if (d2_dt.day == 31) and (d1_dt.day >= 30):
            day2 = 30

        if d1_dt.day == 31:
            day1 = 30

        if (d1_dt.day == m_range1[-1] and d1_dt.month == 2) and (d2_dt.day == m_range2[-1] and d2_dt.month == 2):  # Corrected d1.month to d2_dt.month
            day2 = 30

        if d1_dt.day == m_range1[-1] and d1_dt.month == 2:
            day1 = 30
        return (d2_dt.year - d1_dt.year) + (d2_dt.month - d1_dt.month) / 12.0 + (day2 - day1) / 360.0

    @staticmethod
    def yf_30360ISDA(d1: _Union[date, datetime], d2: _Union[date, datetime]) -> float:
        """This method implements the 30/360 ISDA (Bond Basis) day count convention.
        The following logic is applied:

        1. If d2.day == 31 and d1.day >= 30 -> d2.day = 30
        2. If d1.day == 31 -> d1.day = 30

        Args:
            d1 (_Union[date, datetime]): start date
            d2 (_Union[date, datetime]): end date

        Returns:
            float: year fraction
        """
        d1_dt = _date_to_datetime(d1)
        d2_dt = _date_to_datetime(d2)

        day1 = d1_dt.day
        day2 = d2_dt.day

        if (d2_dt.day == 31) and (d1_dt.day >= 30):  # Original logic used d1.day here
            day2 = 30

        if d1_dt.day == 31:
            day1 = 30
        return (d2_dt.year - d1_dt.year) + (d2_dt.month - d1_dt.month) / 12.0 + (day2 - day1) / 360.0

    @staticmethod
    def yf_30E360(d1: _Union[date, datetime], d2: _Union[date, datetime]) -> float:
        """This day count convention implements the Eurobond Basis day count convention.
        The following logic is applied:

        1. If d1.day >= 30 -> d1.day = 30
        2. If d2.day >= 30 -> d2.day = 30

        Args:
            d1 (_Union[date, datetime]): start date
            d2 (_Union[date, datetime]): end date

        Returns:
            float: year fraction
        """
        d1_dt = _date_to_datetime(d1)
        d2_dt = _date_to_datetime(d2)

        def _adjust_day(day: int):
            if day >= 30:
                return 30
            return day

        day1 = _adjust_day(d1_dt.day)
        day2 = _adjust_day(d2_dt.day)

        return (d2_dt.year - d1_dt.year) + (d2_dt.month - d1_dt.month) / 12.0 + (day2 - day1) / 360.0


class Period:
    def __init__(self, years: int = 0, months: int = 0, days: int = 0):
        """
        Time Period expressed in years, months and days.

        Args:
            years (int, optional): Number of years in time period. Defaults to 0.
            months (int, optional): Number of months in time period. Defaults to 0.
            days (int, optional): Number of days in time period. Defaults to 0.
        """
        self.years = years
        self.months = months
        self.days = days

    @staticmethod
    def from_string(period: str):
        """Creates a Period from a string

        Args:
            period (str): The string defining the period. The string must be defined by the number of days/months/years followed by one of the letters 'Y'/'M'/'D', i.e. '6M' means 6 months, or 'O/N', or 'T/N'.

        Returns:
            Period: The resulting period

        Examples:
            .. code-block:: python

                >>> p = Period('6M')  # period of 6 months
                >>> p = Period('1Y') #period of 1 year
        """
        if period == "T/N" or period == "O/N":
            return Period(days=1)
        else:
            period_length = int(period[:-1])
            period_type = period[1]
            if period_type == "Y":
                return Period(years=period_length)
            elif period_type == "M":
                return Period(months=period_length)
            elif period_type == "D":
                return Period(days=period_length)
            raise Exception(
                period + " is not a valid period string. See documentation of tools.datetools.Period for deocumentation of valid strings."
            )

    @property
    def years(self) -> int:
        """
        Getter for years of period.

        Returns:
            int: Number of years for specified time period.
        """
        return self.__years

    @years.setter
    def years(self, years: int):
        """
        Setter for years of period.

        Args:
            years(int): Number of years.
        """
        self.__years = years

    @property
    def months(self) -> int:
        """
        Getter for months of period.

         Returns:
             int: Number of months for specified time period.
        """
        return self.__months

    @months.setter
    def months(self, months: int):
        """
        Setter for months of period.

        Args:
            months(int): Number of months.
        """
        self.__months = months

    @property
    def days(self) -> int:
        """
        Getter for number of days in time period.

        Returns:
            int: Number of days for specified time period.
        """
        return self.__days

    @days.setter
    def days(self, days: int):
        """
        Setter for days of period.

        Args:
            days(int): Number of days.
        """
        self.__days = days

    def __eq__(self, other: "Period"):
        return self.years == other.years and self.months == other.months and self.days == other.days


class Schedule:
    def __init__(
        self,
        start_day: _Union[date, datetime],
        end_day: _Union[date, datetime],
        time_period: _Union[Period, str],
        backwards: bool = True,
        # stub_mode: str = "automatic", # could alternatively be "force" or "none" (i.e. force a stub period even if not necessary, or do not allow stub periods at all)
        stub_type_is_Long: bool = True,
        # stub_placement: str = "ending", # could alternatively be "beginning" (i.e. place stub period at the end, at the beginning)
        business_day_convention: _Union[RollConvention, str] = RollConvention.MODIFIED_FOLLOWING,
        calendar: _Optional[_Union[_HolidayBase, str]] = None,
        roll_convention: _Union[RollRule, str] = RollRule.NONE,
        settle_days: int = 0,
        ref_date: _Optional[_Union[date, datetime]] = None,
    ):
        """
        A schedule is a list of dates, e.g. of coupon payments, fixings, etc., which is defined by its first (= start
        day) and last (= end day) day, by its distance between two consecutive dates (= time period) and by the
        procedure for rolling out the schedule, more precisely by the direction (backwards/forwards) and the dealing
        with incomplete periods (stubs). Moreover, the schedule ensures to comply to business day conventions with
        respect to a specified holiday calendar.

        Args:
            start_day (_Union[date, datetime]): Schedule's first day - beginning of the schedule.
            end_day (_Union[date, datetime]): Schedule's last day - end of the schedule.
            time_period (_Union[Period, str]): Time distance between two consecutive dates.
            backwards (bool, optional): Defines direction for rolling out the schedule. True means the schedule will be
                                        rolled out (backwards) from end day to start day. Defaults to True.
            stub_type_is_Long (bool, optional): Defines if a stub period is accepted (False) to be shorter than
                                   the others, or if its remaining days are added to the neighbouring period (True).
                                   Defaults to True.
            business_day_convention (_Union[RollConvention, str], optional): Set of rules defining the adjustment of
                                                                             days to ensure each date being a business
                                                                             day with respect to a given holiday
                                                                             calendar. Defaults to
                                                                             RollConvention.MODIFIED_FOLLOWING
            calendar (_Union[_HolidayBase, str], optional): Holiday calendar defining the bank holidays of a country or
                                                          province (but not all non-business days as for example
                                                          Saturdays and Sundays).
                                                          Defaults (through constructor) to holidays.ECB
                                                          (= Target2 calendar) between start_day and end_day.
            roll_convention (_Union[RollRule, str], optional): Defines the roll convention for the schedule.
            settle_days (int, optional): Number of days for settlement. Defaults to 0.
            ref_date (_Optional[_Union[date, datetime]]): Reference date for the schedule. If provided, the schedule will be shortened and include the dates that are after the reference date plus the immediate date before it.

        Examples:

            .. code-block:: python

                >>> from datetime import date
                >>> from rivapy.tools import Schedule
                >>> schedule = Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 3, 0), True, False, RollConvention.UNADJUSTED, holidays_de).generate_dates(False),
                       [date(2020, 8, 21), date(2020, 11, 21), date(2021, 2, 21), date(2021, 5, 21), date(2021, 8, 21)])
        """
        self.start_day = start_day
        self.end_day = end_day
        self.time_period = time_period
        self.backwards = backwards
        self.stub_type_is_Long = stub_type_is_Long
        self.business_day_convention = business_day_convention
        self.calendar = calendar
        self.roll_convention = roll_convention
        self.settle_days = settle_days
        self.ref_date = ref_date

    @property
    def start_day(self):
        """
        Getter for schedule's start date.

        Returns:
            Start date of specified schedule.
        """
        return self.__start_day

    @start_day.setter
    def start_day(self, start_day: _Union[date, datetime]):
        self.__start_day = _date_to_datetime(start_day)

    @property
    def end_day(self):
        """
        Getter for schedule's end date.

        Returns:
            End date of specified schedule.
        """
        return self.__end_day

    @end_day.setter
    def end_day(self, end_day: _Union[date, datetime]):
        self.__end_day = _date_to_datetime(end_day)

    @property
    def time_period(self):
        """
        Getter for schedule's time period.

        Returns:
            Time period of specified schedule.
        """
        return self.__time_period

    @time_period.setter
    def time_period(self, time_period: _Union[Period, str]):
        self.__time_period = _term_to_period(time_period)

    @property
    def backwards(self):
        """
        Getter for schedule's roll out direction.

        Returns:
            True, if rolled out from end day to start day.
            False, if rolled out from start day to end day.
        """
        return self.__backwards

    @backwards.setter
    def backwards(self, backwards: bool):
        self.__backwards = backwards

    @property
    def stub_type_is_Long(self):
        """
        Getter for potential existence of shlong periods (stub_type_is_long).

        Returns:
            True, if a shorter period is allowed.
            False, if only a longer period is allowed.
        """
        return self.stub_type_is_Long

    @stub_type_is_Long.setter
    def stub_type_is_Long(self, stub_type_is_Long: bool):
        self.__stub_type_is_Long = stub_type_is_Long

    @property
    def business_day_convention(self):
        """
        Getter for schedule's business day convention.

        Returns:
            Business day convention of specified schedule.
        """
        return self.__business_day_convention

    @business_day_convention.setter
    def business_day_convention(self, business_day_convention: _Union[RollConvention, str]):
        self.__business_day_convention = RollConvention.to_string(business_day_convention)

    @property
    def calendar(self):
        """
        Getter for schedule's holiday calendar.

        Returns:
            Holiday calendar of specified schedule.
        """
        return self.__calendar

    @calendar.setter
    def calendar(self, calendar: _Union[_HolidayBase, str]):
        if calendar is None:
            self.__calendar: _HolidayBase = _ECB(years=range(self.__start_day.year, self.__end_day.year + 1))
        else:
            self.__calendar: _HolidayBase = _string_to_calendar(calendar)

    @property
    def roll_convention(self):
        """
        Getter for schedule's roll convention.

        Returns:
            Roll convention of specified schedule.
        """
        return self.__roll_convention

    @roll_convention.setter
    def roll_convention(self, roll_convention: _Union[RollRule, str]):
        """
        Setter for schedule's roll convention.

        Args:
            roll_convention (Union[RollRule, str]): Roll convention of specified schedule.
        """
        if isinstance(roll_convention, str):
            roll_convention = RollRule[roll_convention.upper()]
        self.__roll_convention = roll_convention.value

    @staticmethod
    def _generate_eom_dates(from_, to_, term, direction, backwards) -> _List[date]:
        dates = []
        if _is_ambiguous_date(from_):
            from_ = datetime(from_.year, from_.month, monthrange(from_.year, from_.month)[-1])
        if _is_ambiguous_date(to_):
            to_ = datetime(to_.year, to_.month, monthrange(to_.year, to_.month)[-1])
        while ((not backwards) & (from_ <= to_)) | (backwards & (to_ <= from_)):
            dates.append(from_)
            from_ += direction * relativedelta(years=term.years, months=term.months, day=31)
        return dates

    @staticmethod
    def _generate_none_dates(from_, to_, term, direction, backwards) -> _List[date]:
        dates = []
        while ((not backwards) & (from_ <= to_)) | (backwards & (to_ <= from_)):
            dates.append(from_)
            from_ += direction * relativedelta(years=term.years, months=term.months, days=term.days)
        return dates

    @staticmethod
    def _generate_dom_dates(from_, to_, term, direction, backwards) -> _List[date]:
        dates = []
        date = from_
        i = 0
        days = from_.day
        while ((not backwards) & (date <= to_)) | (backwards & (to_ <= date)):
            dates.append(date)
            i += 1
            date = from_ + direction * relativedelta(years=term.years, months=term.months * i, day=days)
        return dates

    @staticmethod
    def _generate_imm_dates(from_, to_, term, direction, backwards) -> _List[date]:
        dates = []
        from_new = from_
        # shift to next IMM if necessary, i.e. from_ may not be part of the generated dates
        if not _is_IMM_date(from_):
            from_new = _date_to_datetime(next_IMM_date(from_))
        if backwards:
            from_new = _date_to_datetime(next_IMM_date(from_ - relativedelta(months=3)))
        # adjust term if not a multiple of 3 months
        term_new = term
        if term.months % 3 != 0:
            term_new = next_IMM_period(term)
        while ((not backwards) & (from_new <= to_)) | (backwards & (to_ <= from_new)):
            dates.append(from_new)
            from_new += direction * relativedelta(years=term_new.years, months=term_new.months, day=1, weekday=WE(3))
        return dates

    @staticmethod
    def _generate_dates_by_roll_convention(roll_convention_, from_, to_, term, direction, backwards) -> _List[date]:
        # Ensure roll_convention_ is an enum instance
        if isinstance(roll_convention_, str):
            roll_convention_ = RollRule[roll_convention_.upper()]
        elif not isinstance(roll_convention_, RollRule):
            raise Exception(f"Invalid roll convention type: {type(roll_convention_)}")

        RollConventionMap = {
            RollRule.EOM: Schedule._generate_eom_dates,
            RollRule.NONE: Schedule._generate_none_dates,
            RollRule.DOM: Schedule._generate_dom_dates,
            RollRule.IMM: Schedule._generate_imm_dates,
        }
        if roll_convention_ not in RollConventionMap:
            raise Exception(f"Unknown roll convention '{roll_convention_}'! Must be one of {list(RollConventionMap.keys())}")
        return RollConventionMap[roll_convention_](from_, to_, term, direction, backwards)

    # ToDo: clarify what is done here --> automatic stub, allow_stub control if long or short, always at the end when rolling forward, at the beginning when rolling backwards
    # ToDo: add tests, check out deposits and FRAs
    @staticmethod
    def _roll_out(
        from_: _Union[date, datetime],
        to_: _Union[date, datetime],
        term: _Union[Period, str],
        backwards: bool = False,
        long_stub: bool = True,
        roll_convention_: _Union[RollRule, str] = "NONE",
        ref_date: _Optional[_Union[date, datetime]] = None,
    ) -> _List[date]:
        """
        Rolls out dates from from_ to to_ in the specified direction applying the given term under consideration of the
        specification for allowing shorter periods.

        Args:
            from_ (_Union[date, datetime]): Beginning of the roll out mechanism.
            to_ (_Union[date, datetime]): End of the roll out mechanism.
            term (Period): Difference between rolled out dates.
            backwards (bool): Direction of roll out mechanism: backwards if True, forwards if False.
            long_stub (bool): Defines if periods longer than term are allowed.

        Returns:
            Date schedule not adjusted to business days.
        """
        if isinstance(term, str):
            term = _term_to_period(term)
        if isinstance(roll_convention_, str):
            roll_convention_ = RollRule[roll_convention_.upper()]
        # convert datetime to date (if necessary):
        from_ = _date_to_datetime(from_)
        to_ = _date_to_datetime(to_)
        # check input consistency:
        if (not backwards) & (from_ < to_):
            direction = +1
        elif backwards & (from_ > to_):
            direction = -1
        else:
            raise Exception(
                "From-date '"
                + str(from_)
                + "' and to-date '"
                + str(to_)
                + "' are not consistent with roll direction (backwards = '"
                + str(backwards)
                + "')!"
            )
        # generates a list of dates ...
        dates = Schedule._generate_dates_by_roll_convention(roll_convention_, from_, to_, term, direction, backwards)
        # return empty list if no dates were generated
        if dates == []:
            logger.info("No dates were generated!")
            return dates
        if roll_convention_ == RollRule.EOM and _is_ambiguous_date(from_):
            from_ = datetime(from_.year, from_.month, monthrange(from_.year, from_.month)[-1])
        if roll_convention_ == RollRule.EOM and _is_ambiguous_date(to_):
            to_ = datetime(to_.year, to_.month, monthrange(to_.year, to_.month)[-1])
        if _date_to_datetime(dates[-1]) != to_:
            # ... by adding a short stub or ...
            if not long_stub or len(dates) == 1:  # 2025 HN
                dates.append(to_)
            # ... by extending last period.
            else:
                dates[-1] = to_

        if ref_date is not None:
            dates = [
                d for d in dates if d >= calc_start_day(ref_date, term, roll_convention=roll_convention_)
            ]  # Keep only dates after the reference date plus the last date before the reference date.
        if backwards:
            dates.reverse()
        return dates

    def generate_dates(self, ends_only: bool) -> _List[date]:
        """
        Generate list of schedule days according to the schedule specification, in particular with regards to business
        day convention, roll_convention and calendar given.

        Args:
            ends_only (bool): Flag to indicate if period beginnings shall be included, e.g. for defining accrual
                                periods: True, if only period ends shall be included, e.g. for defining payment dates.

        Returns:
            List[date]: List of schedule dates (including start and end date) adjusted to rolling convention.
        """
        # roll out dates ignoring any business day issues
        if self.__backwards:
            schedule_dates = Schedule._roll_out(
                self.__end_day, self.__start_day, self.__time_period, True, self.__stub_type_is_Long, self.__roll_convention
            )
            # schedule_dates.reverse()
        else:
            schedule_dates = Schedule._roll_out(
                self.__start_day, self.__end_day, self.__time_period, False, self.__stub_type_is_Long, self.__roll_convention
            )

        # adjust according to business day convention
        rolled_schedule_dates = [roll_day(schedule_dates[0], self.__calendar, self.__business_day_convention, schedule_dates[0], self.settle_days)]
        for i in range(1, len(schedule_dates)):
            rolled_schedule_dates.append(
                roll_day(schedule_dates[i], self.__calendar, self.__business_day_convention, rolled_schedule_dates[i - 1], self.settle_days)
            )
        if ends_only:
            rolled_schedule_dates.pop(0)

        logger.debug(
            "Schedule dates successfully calculated from '"
            + str(self.__start_day)
            + "' to '"
            + str(self.__end_day)
            + "' adjusted by business day convention and settlement days."
        )
        return rolled_schedule_dates


def _date_to_datetime(date_time: _Union[datetime, date]) -> datetime:
    """
    Converts a date to a datetime or leaves it unchanged if it is already of type datetime.

    Args:
        date_time (_Union[datetime, date]): Date(time) to be converted.

    Returns:
        datetime: (Potentially) Converted datetime.
    """
    if isinstance(date_time, datetime):
        return date_time
    elif isinstance(date_time, date):
        return datetime.combine(date_time, datetime.min.time())
    else:
        raise TypeError("'" + str(date_time) + "' must be of type datetime or date!")


def _datetime_to_date_list(date_times: _Union[_List[datetime], _List[date]]) -> _List[date]:
    """
    Converts types of date  list from datetime to date or leaves it unchanged if they are already of type date.

    Args:
        date_times (_Union[List[datetime], List[date]]): List of date(time)s to be converted.

    Returns:
        List[date]: List of (potentially) converted date(time)s.
    """
    if isinstance(date_times, list):
        return [_date_to_datetime(date_time) for date_time in date_times]
    else:
        raise TypeError("'" + str(date_times) + "' must be a list of type datetime or date!")


def _string_to_period(term: str) -> Period:
    """
    Converts terms, e.g. 1D, 3M, and 5Y, into periods, i.e. Period(0, 0, 1), Period(0, 3, 0), and Period(5, 0, 0),
    respectively.

    Args:
        term (str): Term to be converted into a period.

    Returns:
        Period: Period corresponding to the term specified.
    """
    if term == "T/N" or term == "O/N":
        return Period(days=1)
    else:
        unit = term[-1]
        try:
            measure = int(term[:-1])
        except ValueError:
            measure = 0
        if unit.upper() == "D":
            period = Period(0, 0, measure)
        elif unit.upper() == "M":
            period = Period(0, measure, 0)
        elif unit.upper() == "Y":
            period = Period(measure, 0, 0)
        else:
            raise Exception("Unknown term! Please use: 'D', 'M', or 'Y'.")
        return period


def _term_to_period(term: _Union[Period, str]) -> Period:
    """
    Converts a term provided as period or string into period format if necessary.

    Args:
        term (_Union[Period, str]): Tenor to be converted if provided as string.

    Returns:
        Period: Tenor (potentially converted) in(to) period format.
    """
    if isinstance(term, Period):
        return term
    elif isinstance(term, str):
        return _string_to_period(term)
    else:
        raise TypeError("The term '" + str(term) + "' must be provided as Period or string!")


def _period_to_string(period: _Union[Period, str]) -> str:
    """
    Converts a period into string format.

    Args:
        period (Period): Period to be converted.

    Returns:
        str: Period in string format.
    """
    if isinstance(period, Period):
        if period.years > 0 and period.months == 0 and period.days == 0:
            return str(period.years) + "Y"
        elif period.months > 0 and period.years == 0 and period.days == 0:
            return str(period.months) + "M"
        elif period.days > 0 and period.years == 0 and period.months == 0:
            return str(period.days) + "D"
        else:
            raise Exception("The period '" + str(period) + "' cannot be converted to string format!")
    elif isinstance(period, str):
        return period


def _is_ambiguous_date(day: _Union[date, datetime]) -> bool:
    """
    Checks if a given day is an ambiguous date, i.e. 30th of January, March, May, July, August, October or December.

    Args:
        day (_Union[date, datetime]): Day to be checked.

    Returns:
        bool: True if day is ambiguous date, False otherwise.
    """
    return ((day.day == 30) and (day.month in [1, 3, 5, 7, 8, 10, 12])) or ((day.day == 28 or day.day == 29) and day.month == 2)


def _is_IMM_date(day: _Union[date, datetime]) -> bool:
    """
    Checks if a given day is an IMM date, i.e. the third Wednesday of March, June, September or December.

    Args:
        day (_Union[date, datetime]): Day to be checked.

    Returns:
        bool: True if day is IMM date, False otherwise.
    """
    return (day.month in [3, 6, 9, 12]) and (day.weekday() == 2) and (day.day >= 15) and (day.day <= 21)


def next_IMM_date(from_date: _Union[date, datetime]) -> date:
    """
    Calculates the next IMM date (3rd Wednesday of March, June, September, December) on or after the given date.

    Args:
        from_date (_Union[date, datetime]): The date from which to find the next IMM date.

    Returns:
        date: The next IMM date on or after the given date.
    """
    from_date_dt = _date_to_datetime(from_date + relativedelta(days=1))
    year = from_date_dt.year
    month = from_date_dt.month

    # Determine the next IMM month
    if month <= 3:
        imm_month = 3
    elif month <= 6:
        imm_month = 6
    elif month <= 9:
        imm_month = 9
    else:
        imm_month = 12

    # Calculate the third Wednesday of the IMM month
    first_day_of_imm_month = datetime(year, imm_month, 1)
    first_wednesday = first_day_of_imm_month + relativedelta(weekday=WE(1))
    third_wednesday = first_wednesday + relativedelta(weeks=2)

    # If the calculated IMM date is before the from_date, move to the next IMM date
    if third_wednesday < from_date_dt:
        if imm_month == 12:
            imm_month = 3
            year += 1
        else:
            imm_month += 3

    first_day_of_imm_month = datetime(year, imm_month, 1)
    first_wednesday = first_day_of_imm_month + relativedelta(weekday=WE(1))
    third_wednesday = first_wednesday + relativedelta(weeks=2)
    logger.warning("Next IMM date from " + str(from_date) + " to next IMM date " + str(third_wednesday))
    return third_wednesday.date()


def next_IMM_period(period: Period) -> Period:
    """
    Adjusts the given period to the next multiple of 3 months, as IMM dates occur every 3 months.

    Args:
        period (Period): The original period.

    Returns:
        Period: The adjusted period.
    """

    months = period.months + (period.years * 12)
    if period.days > 0:
        months += 1  # If there are extra days, round up to the next month
    months = ((months + 2) // 3) * 3  # Round up to next multiple of 3
    return Period(0, months, 0)


def calc_end_day(
    start_day: _Union[date, datetime],
    term: str,
    business_day_convention: _Union[RollConvention, str] = RollConvention.UNADJUSTED,
    calendar: _Union[_HolidayBase, str] = _ECB(),
    roll_convention: _Union[RollRule, str] = RollRule.NONE,
) -> date:
    """
    Derives the end date of a time period based on the start day the the term given as string, e.g. 1D, 3M, or 5Y.
    If business day convention, corresponding calendar, and roll convention are provided the end date is additionally rolled accordingly.

    Args:
        start_day (_Union[date, datetime): Beginning of the time period with length term.
        term (str): Term defining the period from start to end date.
        business_day_convention (_Union[RollConvention, str], optional): Set of rules defining how to adjust
                                                                         non-business days. Defaults to None.
        calendar (_Union[_HolidayBase, str], optional): Holiday calender defining non-business days
                                                      (but not Saturdays and Sundays).
                                                      Defaults to None.
        roll_convention (_Union[RollRule, str], optional): Convention for rolling dates, e.g. "EOM" for end of month.

    Returns:
        date: End date potentially adjusted according to the specified business day convention with respect to the given
              calendar.
    """
    start_date = _date_to_datetime(start_day)
    period = _term_to_period(term)
    # Convert string roll_convention to enum if needed
    roll_conv = RollRule.to_string(roll_convention) if roll_convention is not None else "NONE"

    if roll_conv == RollRule.EOM.value and _is_ambiguous_date(start_date):  # add ambiguous dates, i.e. 30 of Jan, Mar, May, Jul, Aug, Oct, Dec
        end_date = start_date + relativedelta(years=period.years, months=period.months, day=31)
    elif roll_conv == RollRule.IMM.value:  # add IMM dates, i.e. 3rd Wednesday of Mar, Jun, Sep, Dec
        period_new = next_IMM_period(period)
        if _is_IMM_date(start_date):
            end_date = start_date + relativedelta(years=period_new.years, months=period_new.months, day=1, weekday=WE(3))
        else:
            end_date = next_IMM_date(start_date) + relativedelta(years=period_new.years, months=period_new.months, day=1, weekday=WE(3))
    elif roll_conv == RollRule.EOM.value or roll_conv == RollRule.IMM.value or roll_conv == RollRule.NONE.value or roll_conv == RollRule.DOM.value:
        end_date = start_date + relativedelta(years=period.years, months=period.months, days=period.days)
    else:
        raise Exception(
            "Unknown roll convention '" + str(roll_convention) + "'! Please use RollRule.NONE, RollRule.EOM, RollRule.DOM, or RollRule.IMM."
        )
    if (business_day_convention is not None) & (calendar is not None):
        end_date = roll_day(end_date, calendar, business_day_convention, start_date)

    return end_date


def calc_start_day(
    end_day: _Union[date, datetime],
    term: _Union[Period, str],
    business_day_convention: _Union[RollConvention, str] = "Unadjusted",
    calendar: _Union[_HolidayBase, str] = _ECB(),
    roll_convention: _Union[RollRule, str] = "NONE",
    max_iter: int = 10,
) -> date:
    """
    Derives the start date of a time period based on the end day, term, business day convention, calendar, and roll_convention.
    The start date may be a business day or not, depending on the business day convention provided.
    The function ensures that applying calc_end_day to the resulting start date with the same parameters returns the original end_day.
    Depending on the combination of the input parameters, a start date may not exist. In such cases, the function returns None and logs a warning.
    For other combinations, the start may not be unique. In such cases, the function returns the latest business day for which the end date is matched.

    Args:
        end_day (_Union[date, datetime]): End of the time period with length term.
        term (str): Term defining the period from start to end date.
        business_day_convention (_Union[RollConvention, str], optional): Set of rules defining how to adjust
                                                                         non-business days. Defaults to "Unadjusted".
        calendar (_Union[_HolidayBase, str], optional): Holiday calendar defining non-business days.
        roll_convention (_Union[RollRule, str], optional): Convention for rolling dates, e.g. "EOM" for end of month.
        max_iter (int, optional): Maximum number of iterations for the search. Defaults to 10.

    Returns:
        date: Start date such that calc_end_day(start_date, ...) == end_day.
    """
    end_date = _date_to_datetime(end_day)
    period = _term_to_period(term)
    if not (business_day_convention == "Unadjusted" or business_day_convention == RollConvention.UNADJUSTED) and not is_business_day(
        end_date, calendar
    ):
        logger.warning(
            f"Cannot not find a start date such that calc_end_day(start_date, ...) == end_day given combination of business day convention, calendar, and end_day."
        )
        return None
    start_date = next_or_previous_business_day(end_date - relativedelta(years=period.years, months=period.months, days=period.days), calendar, False)
    # Try to find the correct start_date such that calc_end_day(start_date, ...) == end_day
    for i in range(max_iter):
        candidate_end = calc_end_day(
            start_date,
            term,
            business_day_convention=business_day_convention,
            calendar=calendar,
            roll_convention=roll_convention,
        )
        if candidate_end == end_date:
            if not is_business_day(start_date, calendar):
                logger.warning(
                    f"Found start date {start_date} such that calc_end_day(start_date, ...) == end_day, but start date is not a business day in the given calendar."
                )
            return start_date
        # Adjust start_date by one day if not matching
        # If candidate_end < end_date, move start_date back; else, move forward
        delta = (_date_to_datetime(candidate_end) - end_date).days
        start_date -= relativedelta(days=delta if delta != 0 else 1)
    # If not found, handle error internally
    logger.error(f"Could not find a start date such that calc_end_day(start_date, ...) == end_day after {max_iter} iterations.")
    return None


def last_day_of_month(day: _Union[date, datetime]) -> date:
    """
    Derives last day of the month corresponding to the given day.

    Args:
        day (_Union[date, datetime]): Day defining month and year for derivation of month's last day.

    Returns:
        date: Date of last day of the corresponding month.
    """
    return date(day.year, day.month, monthrange(day.year, day.month)[1])


def is_last_day_of_month(day: _Union[date, datetime]) -> bool:
    """
    Checks if a given day is the last day of the corresponding month.

    Args:
        day (_Union[date, datetime]): Day to be checked.

    Returns:
        bool: True, if day is last day of the month, False otherwise.
    """
    return _date_to_datetime(day) == _date_to_datetime(last_day_of_month(day))


def is_business_day(day: _Union[date, datetime], calendar: _Union[_HolidayBase, str]) -> bool:
    """
    Checks if a given day is a business day in a given calendar.

    Args:
        day (_Union[date, datetime]): Day to be checked.
        calendar (_Union[_HolidayBase, str]): List of holidays defined by the given calendar.

    Returns:
        bool: True, if day is a business day, False otherwise.
    """
    # TODO: adjust for countries with weekend not on Saturday/Sunday (http://worldmap.workingdays.org/)
    return (day.isoweekday() < 6) & (day not in _string_to_calendar(calendar))


def last_business_day_of_month(day: _Union[date, datetime], calendar: _Union[_HolidayBase, str]) -> date:
    """
    Derives the last business day of a month corresponding to a given day based on the holidays set in the calendar.

    Args:
        day (_Union[date, datetime]): Day defining month and year for deriving the month's last business day.
        calendar (_Union[_HolidayBase, str]): List of holidays defined by the given calendar.

    Returns:
        date: Date of last business day of the corresponding month.
    """
    check_day = date(day.year, day.month, monthrange(day.year, day.month)[1])
    while not (is_business_day(check_day, calendar)):
        check_day -= relativedelta(days=1)
    return check_day


def is_last_business_day_of_month(day: _Union[date, datetime], calendar: _Union[_HolidayBase, str]) -> bool:
    """
    Checks it the given day is the last business day of the corresponding month.

    Args:
        day (_Union[date, datetime]): day to be checked
        calendar (_Union[_HolidayBase, str]): list of holidays defined by the given calendar

    Returns:
        bool: True if day is last business day of the corresponding month, False otherwise.
    """
    return _date_to_datetime(day) == _date_to_datetime(last_business_day_of_month(day, calendar))


def nearest_business_day(day: _Union[date, datetime], calendar: _Union[_HolidayBase, str], following_first: bool = True) -> date:
    """
    Derives nearest business day from given day for a given calendar. If there are equally near days preceding and
    following the flag following_first determines if the following day is preferred to the preceding one.

    Args:
        day (_Union[date, datetime]): Day for which the nearest business day is to be found.
        calendar (_Union[_HolidayBase, str]): List of holidays given by calendar.
        following_first (bool): Flag for deciding if following days are preferred to an equally near preceding day.
                                Default value is True.

    Returns:
        date: Nearest business day to given day according to given calendar.
    """
    distance = 0
    if following_first:
        direction = -1
    else:
        direction = +1

    day = _date_to_datetime(day)
    while not is_business_day(day, calendar):
        distance += 1
        direction *= -1
        day += direction * relativedelta(days=distance)
    return day


def nearest_last_business_day_of_month(day: _Union[date, datetime], calendar: _Union[_HolidayBase, str], following_first: bool = True) -> date:
    """
    Derives nearest last business day of a month from given day for a given calendar. If there are equally near days
    preceding and following the flag following_first determines if the following day is preferred to the preceding one.

    Args:
        day (_Union[date, datetime]): Day for which the nearest last business day of the month is to be found.
        calendar (_Union[_HolidayBase, str]): List of holidays given by calendar.
        following_first (bool, optional): Flag for deciding if following days are preferred to an equally near preceding
                                          day. Defaults to True.

    Returns:
        date: Nearest last business day of a month to given day according to given calendar.
    """
    distance = 0
    if following_first:
        direction = -1
    else:
        direction = +1

    day = _date_to_datetime(day)
    while not is_last_business_day_of_month(day, calendar):
        distance += 1
        direction *= -1
        day += direction * relativedelta(days=distance)
    return day


def next_or_previous_business_day(day: _Union[date, datetime], calendar: _Union[_HolidayBase, str], following_first: bool) -> date:
    """
    Derives the preceding or following business day to a given day according to a given calendar depending on the flag
    following_first. If the day is already a business day the function directly returns the day.

    Args:
        day (_Union[date, datetime]): Day for which the preceding or following business day is to be found.
        calendar (_HolidayBase): List of holidays defined by the calendar.
        following_first (bool): Flag to determine in the following (True) or preceding (False) business day is to be
        found.

    Returns:
        date: Preceding or following business day, respectively, or day itself if it is a business day.
    """
    if following_first:
        direction = +1
    else:
        direction = -1

    day = _date_to_datetime(day)
    while not is_business_day(day, calendar):
        day += direction * relativedelta(days=1)

    return day


def following(day: _Union[date, datetime], calendar: _Union[_HolidayBase, str]) -> date:
    """
    Derives the (potentially) adjusted business day according to the business day convention 'Following' for a specified
    day with respect to a specific calendar: The adjusted date is the following good business day.

    Args:
        day (_Union[date, datetime]): Day to be adjusted according to the roll convention if not already a business day.
        calendar (_Union[_HolidayBase, str]): Calendar defining holidays additional to weekends.

    Returns:
        date: Adjusted business day according to the roll convention 'Following' with respect to calendar if the day is
              not already a business day. Otherwise the (unadjusted) day is returned.
    """
    return next_or_previous_business_day(day, calendar, True)


def preceding(day: _Union[date, datetime], calendar: _Union[_HolidayBase, str]) -> date:
    """
    Derives the (potentially) adjusted business day according to the business day convention 'Preceding' for a specified
    day with respect to a specific calendar: The adjusted date is the preceding good business day.

    Args:
        day (_Union[date, datetime]): Day to be adjusted according to the roll convention if not already a business day.
        calendar (_Union[_HolidayBase, str]): Calendar defining holidays additional to weekends.

    Returns:
        date: Adjusted business day according to the roll convention 'Preceding' with respect to calendar if the day is
              not already a business day. Otherwise the (unadjusted) day is returned.
    """
    return next_or_previous_business_day(day, calendar, False)


def modified_following(day: _Union[date, datetime], calendar: _Union[_HolidayBase, str]) -> date:
    """
    Derives the (potentially) adjusted business day according to the business day convention 'Modified Following' for a
    specified day with respect to a specific calendar: The adjusted date is the following good business day unless the
    day is in the next calendar month, in which case the adjusted date is the preceding good business day.

    Args:
        day (_Union[date, datetime]): Day to be adjusted according to the roll convention if not already a business day.
        calendar (_Union[_HolidayBase, str]): Calendar defining holidays additional to weekends.

    Returns:
        date: Adjusted business day according to the roll convention 'Modified Following' with respect to calendar if
              the day is not already a business day. Otherwise the (unadjusted) day is returned.
    """
    next_day = next_or_previous_business_day(day, calendar, True)
    if next_day.month != day.month:
        return preceding(day, calendar)
    else:
        return next_day


def modified_following_eom(day: _Union[date, datetime], calendar: _Union[_HolidayBase, str], start_day: _Union[date, datetime]) -> date:
    """
    Derives the (potentially) adjusted business day according to the business day convention 'End of Month' for a
    specified day with respect to a specific calendar: Where the start date of a period is on the final business day of
    a particular calendar month, the end date is on the final business day of the end month (not necessarily the
    corresponding date in the end month).

    Args:
        day (_Union[date, datetime]): Day to be adjusted according to the roll convention if not already a business day.
        calendar (_Union[_HolidayBase, str]): Calendar defining holidays additional to weekends.
        start_day (_Union[date, datetime]): Day at which the period under consideration begins.

    Returns:
        date: Adjusted business day according to the roll convention 'End of Month' with respect to calendar.
    """
    if isinstance(start_day, date) | isinstance(start_day, datetime):
        if is_last_business_day_of_month(start_day, calendar):
            return nearest_last_business_day_of_month(day, calendar)
        else:
            return modified_following(day, calendar)
    else:
        raise Exception("The roll convention " + str(RollConvention.MODIFIED_FOLLOWING_EOM) + " cannot be evaluated without a start_day")


def modified_following_bimonthly(day: _Union[date, datetime], calendar: _Union[_HolidayBase, str]) -> date:
    """
    Derives the (potentially) adjusted business day according to the business day convention 'Modified Following
    Bimonthly' for a specified day with respect to a specific calendar: The adjusted date is the following good business
    day unless that day crosses the mid-month (15th) or end of a month, in which case the adjusted date is the preceding
    good business day.

    Args:
        day (_Union[date, datetime]): Day to be adjusted according to the roll convention if not already a business day.
        calendar (_Union[_HolidayBase, str]): Calendar defining holidays additional to weekends.

    Returns:
        date: Adjusted business day according to the roll convention 'Modified Following Bimonthly' with respect to
              calendar if the day is not already a business day. Otherwise the (unadjusted) day is returned.
    """
    next_day = next_or_previous_business_day(day, calendar, True)
    if (next_day.month != day.month) | ((next_day.day > 15) & (day.day <= 15)):
        return preceding(day, calendar)
    else:
        return next_day


def modified_preceding(day: _Union[date, datetime], calendar: _Union[_HolidayBase, str]) -> date:
    """
    Derives the (potentially) adjusted business day according to the business day convention 'Modified Preceding' for a
    specified day with respect to a specific calendar: The adjusted date is the preceding good business day unless the
    day is in the previous calendar month, in which case the adjusted date is the following good business day.

    Args:
        day (_Union[date, datetime]): Day to be adjusted according to the roll convention if not already a business day.
        calendar (_Union[_HolidayBase, str]): Calendar defining holidays additional to weekends.

    Returns:
        date: Adjusted business day according to the roll convention 'Modified Preceding' with respect to calendar if
              the day is not already a business day. Otherwise the (unadjusted) day is returned.
    """
    prev_day = next_or_previous_business_day(day, calendar, False)
    if prev_day.month != day.month:
        return following(day, calendar)
    else:
        return prev_day


# to be used in the switcher (identical argument list)
def unadjusted(day: _Union[date, datetime], _) -> date:
    """
    Leaves the day unchanged independent from the fact if it is already a business day.

    Args:
        day (_Union[date, datetime]): Day to be adjusted according to the roll convention.
        _: Placeholder for calendar argument.

    Returns:
        date: Unadjusted day.
    """
    return _date_to_datetime(day)


def roll_day(
    day: _Union[date, datetime],
    calendar: _Union[_HolidayBase, str],
    business_day_convention: _Union[RollConvention, str],
    start_day: _Optional[_Union[date, datetime]] = None,
    settle_days: int = 0,
) -> date:
    """
    Adjusts a given day according to the specified business day convention with respect to a given calendar or if the
    given day falls on a Saturday or Sunday. For some roll conventions not only the (end) day to be adjusted but also
    the start day of a period is relevant for the adjustment of the given (end) day.

    Args:
        day (_Union[date, datetime]): Day to be adjusted if it is a non-business day.
        calendar (_Union[_HolidayBase, str]): Holiday calendar defining non-business days (but not weekends).
        business_day_convention (_Union[RollConvention, str]): Set of rules defining how to adjust non-business days.
        start_day (_Union[date, datetime], optional): Period's start day that may influence the rolling of the end day.
                                                      Defaults to None.

    Returns:
        date: Adjusted day.
    """
    roll_convention = RollConvention.to_string(business_day_convention)
    # if start_day is not None:
    #    start_day = _date_to_datetime(start_day)

    switcher: Dict[str, Callable[..., date]] = {
        "Unadjusted": unadjusted,
        "Following": following,
        "ModifiedFollowing": modified_following,
        "ModifiedFollowingEOM": modified_following_eom,
        "ModifiedFollowingBimonthly": modified_following_bimonthly,
        "Nearest": nearest_business_day,
        "Preceding": preceding,
        "ModifiedPreceding": modified_preceding,
    }
    # Get the appropriate roll function from switcher dictionary

    roll_func = switcher.get(roll_convention)

    if roll_func is None:
        raise ValueError(f"Business day convention '{business_day_convention}' is not known!")

    # Check if roll_func expects three arguments (including self for methods)
    import inspect

    params = inspect.signature(roll_func).parameters
    if "start_day" in params and settle_days == 0:
        return roll_func(day, calendar, start_day)
    elif "start_day" in params and settle_days > 0:
        with_settlement = roll_func(day, calendar, start_day) + relativedelta(days=settle_days)
        return roll_func(with_settlement, calendar, start_day)
    elif "start_day" not in params and settle_days > 0:
        with_settlement = roll_func(day, calendar) + relativedelta(days=settle_days)
        return roll_func(with_settlement, calendar)
    else:
        return roll_func(day, calendar)


def serialize_date(val):
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    return val


# class PowerSchedule:
#     def __init__(self,
#                  start_day: _Union[date, datetime],
#                  end_day: _Union[date, datetime],
#                  time_period: _Union[Period, str],
#                  backwards: bool = True,
#                  business_day_convention: _Union[RollConvention, str] = RollConvention.MODIFIED_FOLLOWING,
#                  calendar: _Union[_HolidayBase, str] = None):
#         """

#         Args:
#             start_day (_Union[date, datetime]): Schedule's first day - beginning of the schedule.
#             end_day (_Union[date, datetime]): Schedule's last day - end of the schedule.
#             time_period (_Union[Period, str]): Time distance between two consecutive dates.
#             backwards (bool, optional): Defines direction for rolling out the schedule. True means the schedule will be
#                                         rolled out (backwards) from end day to start day. Defaults to True.
#             stub (bool, optional): Defines if the first/last period is accepted (True), even though it is shorter than
#                                    the others, or if it remaining days are added to the neighbouring period (False).
#                                    Defaults to True.
#             business_day_convention (_Union[RollConvention, str], optional): Set of rules defining the adjustment of
#                                                                              days to ensure each date being a business
#                                                                              day with respect to a given holiday
#                                                                              calendar. Defaults to
#                                                                              RollConvention.MODIFIED_FOLLOWING
#             calendar (_Union[_HolidayBase, str], optional): Holiday calendar defining the bank holidays of a country or
#                                                           province (but not all non-business days as for example
#                                                           Saturdays and Sundays).
#                                                           Defaults (through constructor) to holidays.ECB
#                                                           (= Target2 calendar) between start_day and end_day.

#         Examples:

#             .. code-block:: python

#                 >>> from datetime import date
#                 >>> from rivapy.tools import schedule
#                 >>> schedule = Schedule(date(2020, 8, 21), date(2021, 8, 21), Period(0, 3, 0), True, False, RollConvention.UNADJUSTED, holidays_de).generate_dates(False),
#                        [date(2020, 8, 21), date(2020, 11, 21), date(2021, 2, 21), date(2021, 5, 21), date(2021, 8, 21)])
#         """
#         self.start_day = start_day
#         self.end_day = end_day
#         self.time_period = time_period
#         self.backwards = backwards
#         self.business_day_convention = business_day_convention
#         self.calendar = calendar


#     @property
#     def start_day(self):
#         """
#         Getter for schedule's start date.

#         Returns:
#             Start date of specified schedule.
#         """
#         return self.__start_day

#     @start_day.setter
#     def start_day(self, start_day: _Union[date, datetime]):
#         self.__start_day = _date_to_datetime(start_day)

#     @property
#     def end_day(self):
#         """
#         Getter for schedule's end date.

#         Returns:
#             End date of specified schedule.
#         """
#         return self.__end_day

#     @end_day.setter
#     def end_day(self, end_day: _Union[date, datetime]):
#         self.__end_day = _date_to_datetime(end_day)

#     @property
#     def time_period(self):
#         """
#         Getter for schedule's time period.

#         Returns:
#             Time period of specified schedule.
#         """
#         return self.__time_period

#     @time_period.setter
#     def time_period(self, time_period: _Union[Period, str]):
#         self.__time_period = _term_to_period(time_period)

#     @property
#     def backwards(self):
#         """
#         Getter for schedule's roll out direction.

#         Returns:
#             True, if rolled out from end day to start day.
#             False, if rolled out from start day to end day.
#         """
#         return self.__backwards

#     @backwards.setter
#     def backwards(self, backwards: bool):
#         self.__backwards = backwards

#     @property
#     def stub(self):
#         """
#         Getter for potential existence of short periods (stubs).

#         Returns:
#             True, if a shorter period is allowed.
#             False, if only a longer period is allowed.
#         """
#         return self.__stub

#     @stub.setter
#     def stub(self, stub: bool):
#         self.__stub = stub

#     @property
#     def business_day_convention(self):
#         """
#         Getter for schedule's business day convention.

#         Returns:
#             Business day convention of specified schedule.
#         """
#         return self.__business_day_convention

#     @business_day_convention.setter
#     def business_day_convention(self, business_day_convention: _Union[RollConvention, str]):
#         self.__business_day_convention = RollConvention.to_string(business_day_convention)

#     @property
#     def calendar(self):
#         """
#         Getter for schedule's holiday calendar.

#         Returns:
#             Holiday calendar of specified schedule.
#         """
#         return self.__calendar

#     @calendar.setter
#     def calendar(self, calendar: _Union[_HolidayBase, str]):
#         if calendar is None:
#             self.__calendar = _ECB(years=range(self.__start_day.year, self.__end_day.year + 1))
#         else:
#             self.__calendar = _string_to_calendar(calendar)

#     @staticmethod
#     def _roll_out(from_: _Union[date, datetime], to_: _Union[date, datetime], term: Period, backwards: bool,
#                   allow_stub: bool) -> _List[date]:
#         """
#         Rolls out dates from from_ to to_ in the specified direction applying the given term under consideration of the
#         specification for allowing shorter periods.

#         Args:
#             from_ (_Union[date, datetime]): Beginning of the roll out mechanism.
#             to_ (_Union[date, datetime]): End of the roll out mechanism.
#             term (Period): Difference between rolled out dates.
#             backwards (bool): Direction of roll out mechanism: backwards if True, forwards if False.
#             allow_stub (bool): Defines if periods shorter than term are allowed.

#         Returns:
#             Date schedule not yet adjusted to any business day convention.
#         """
#         # convert datetime to date (if necessary):
#         from_ = _date_to_datetime(from_)
#         to_ = _date_to_datetime(to_)

#         # check input consistency:
#         if (~backwards) & (from_ < to_):
#             direction = +1
#         elif backwards & (from_ > to_):
#             direction = -1
#         else:
#             raise Exception("From-date '" + str(from_) + "' and to-date '" + str(to_) +
#                             "' are not consistent with roll direction (backwards = '" + str(backwards) + "')!")

#         # generates a list of dates ...
#         dates = []
#         # ... for forward rolling case  or  backward rolling case ...
#         while ((~backwards) & (from_ <= to_)) | (backwards & (to_ <= from_)):
#             dates.append(from_)
#             from_ += direction * relativedelta(years=term.years, months=term.months, days=term.days)
#             # ... and compete list for fractional periods ...
#         if dates[-1] != to_:
#             # ... by adding stub or ...
#             if allow_stub:
#                 dates.append(to_)
#             # ... by extending last period.
#             else:
#                 dates[-1] = to_
#         return dates

#     def generate_dates(self, ends_only: bool) -> _List[date]:
#         """
#         Generate list of schedule days according to the schedule specification, in particular with regards to business
#         day convention and calendar given.

#         Args:
#             ends_only (bool): Flag to indicate if period beginnings shall be included, e.g. for defining accrual
#                               periods: True, if only period ends shall be included, e.g. for defining payment dates.

#         Returns:
#             List[date]: List of schedule dates (including start and end date) adjusted to rolling convention.
#         """
#         # roll out dates ignoring any business day issues
#         if self.__backwards:
#             schedule_dates = Schedule._roll_out(self.__end_day, self.__start_day, self.__time_period,
#                                                 True, self.__stub)
#             schedule_dates.reverse()
#         else:
#             schedule_dates = Schedule._roll_out(self.__start_day, self.__end_day, self.__time_period,
#                                                 False, self.__stub)

#         # adjust according to business day convention
#         rolled_schedule_dates = [roll_day(schedule_dates[0], self.__calendar, self.__business_day_convention,
#                                           schedule_dates[0])]
#         [rolled_schedule_dates.append(roll_day(schedule_dates[i], self.__calendar, self.__business_day_convention,
#                                                rolled_schedule_dates[i - 1])) for i in range(1, len(schedule_dates))]

#         if ends_only:
#             rolled_schedule_dates.pop(0)

#         logger.debug("Schedule dates successfully calculated from '"
#                      + str(self.__start_day) + "' to '" + str(self.__end_day) + "'.")
#         return rolled_schedule_dates
