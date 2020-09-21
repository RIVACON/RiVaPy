# -*- coding: utf-8 -*-


from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from calendar import monthrange
from typing import List, Union, Tuple
from holidays import HolidayBase, ECB
from RiVaPy.tools.enums import Roll_Convention
from RiVaPy.tools.validators import roll_convention_to_string, string_to_calendar
import logging


# TODO: Switch to locally configured logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Period:
    def __init__(self, years: int = 0, months: int = 0, days: int = 0):
        """
        Time Period expressed in years, months and days.

        Args:
            years (int, optional): Number of years in time period. Defaults to 0.
            months (int, optional): Number of months in time period. Defaults to 0.
            days (int, optional): Number of days in time period. Defaults to 0.
        """
        self.__years = years
        self.__months = months
        self.__days = days

    @property
    def years(self):
        """
        Getter for years of period.

        Returns:
            Number of years for specified time period.
        """
        return self.__years

    @property
    def months(self):
        """
       Getter for months of period.

        Returns:
            Number of months for specified time period.
        """
        return self.__months

    @property
    def days(self):
        """
        Number of days in time period.

        Returns:
            Number of days for specified time period.
        """
        return self.__days


class Schedule:
    def __init__(self, start_day: Union[date, datetime], end_day: Union[date, datetime],
                 time_period: Union[Period, str], backwards: bool = True, stub: bool = False,
                 business_day_convention: Union[Roll_Convention, str] = Roll_Convention.MODIFIED_FOLLOWING,
                 calendar: Union[HolidayBase, str] = None):
        """
        A schedule is a list of dates, e.g. of coupon payments, fixings, etc., which is defined by its fist (= start
        day) and last (= end day) day, by its distance between two consecutive dates (= time period) and by the
        procedure for rolling out the schedule, more precisely by the direction (backwards/forwards) and the dealing
        with incomplete periods (stubs). Moreover, the schedule ensures to comply to business day conventions with
        respect to a specified holiday calendar.

        Args:
            start_day (Union[date, datetime]): Schedule's first day - beginning of the schedule.
            end_day (Union[date, datetime]): Schedule's last day - end of the schedule.
            time_period (Period): Time distance between two consecutive dates.
            backwards (bool, optional): Defines direction for rolling out the schedule. Defaults to Ture, i.e. the
                                        schedule will be rolled out (backwards) from end day to start day.
                                        Defaults to True.
            stub (bool, optional): Defines if the last period is accepted (True), even though it is shorter than the
                                   others, or if it remaining days are added to the neighbouring period (False).
                                   Defaults to True.
            business_day_convention (Union[Roll_Convention, str], optional): Set of rules defining the adjustment of
                                                                             days to ensure each date being a business
                                                                             day with respect to a given holiday
                                                                             calendar. Defaults to
                                                                             Roll_Convention.MODIFIED_FOLLOWING
            calendar (Union[HolidayBase, str], optional): Holiday calendar defining the bank holidays of a country or
                                                          province (but not all non-business days as for example
                                                          Saturdays and Sundays).
                                                          Defaults (through constructor) to holidays.ECB
                                                          (= Target2 calendar) between start_day and end_day.
        """
        # start and end day
        self.__start_day, self.__end_day = check_start_before_end(start_day, end_day)
        # time_period
        self.__time_period = tenor_to_period(time_period)
        # define roll procedure (direction and shorter periods)
        self.__backwards = backwards
        self.__stub = stub
        # define business day convention w.r.t. calendar
        self.__business_day_convention = roll_convention_to_string(business_day_convention)
        if calendar is None:
            self.__calendar = ECB(years=range(start_day.year, end_day.year + 1))
        else:
            self.__calendar = string_to_calendar(calendar)

    @property
    def start_day(self):
        """
        Getter for schedule's start date.

        Returns:
            Start date of specified schedule.
        """
        return self.__start_day

    @property
    def end_day(self):
        """
        Getter for schedule's end date.

        Returns:
            End date of specified schedule.
        """
        return self.__end_day

    @property
    def time_period(self):
        """
        Getter for schedule's time period.

        Returns:
            Time period of specified schedule.
        """
        return self.__time_period

    @property
    def backwards(self):
        """
        Getter for schedule's roll out direction.

        Returns:
            True, if rolled out from end day to start day.
            False, if rolled out from start day to end day.
        """
        return self.__backwards

    @property
    def stub(self):
        """
        Getter for potential existence of short periods (stubs).

        Returns:
            True, if a shorter period is allowed.
            False, if only a longer period is allowed.
        """
        return self.__stub

    @property
    def business_day_convention(self):
        """
        Getter for schedule's business day convention.

        Returns:
            Business day convention of specified schedule.
        """
        return self.__business_day_convention

    @property
    def calendar(self):
        """
        Getter for schedule's holiday calendar.

        Returns:
            Holiday calendar of specified schedule.
        """
        return self.__calendar

    @staticmethod
    def _roll_out(from_: Union[date, datetime], to_: Union[date, datetime], term: Period, forwards: bool,
                  allow_stub: bool) -> List[date]:
        """
        Rolls out dates from from_ to to_ in the specified direction applying the given term under consideration of the
        specification for allowing shorter periods.

        Args:
            from_ (Union[date, datetime]): Beginning of the roll out mechanism.
            to_ (Union[date, datetime]): End of the roll out mechanism.
            term (Period): Difference between rolled out dates.
            forwards (bool): Direction of roll out mechanism: forwards if True, backwards if False.
            allow_stub (bool): Defines if periods shorter than term are allowed.

        Returns:
            Date schedule not yet adjusted to any business day convention.
        """
        # convert datetime to date (if necessary):
        from_ = datetime_to_date(from_)
        to_ = datetime_to_date(to_)

        # check input consistency:
        if forwards & (from_ < to_):
            direction = +1
        elif (~forwards) & (from_ > to_):
            direction = -1
        else:
            raise Exception("From-date '" + str(from_) + "' and to-date '" + str(to_) +
                            "' are not consistent with roll direction (forwards = '" + str(forwards) + "')!")

        # generates a list of dates ...
        dates = []
        # ... for forward rolling case  or  backward rolling case ...
        while (forwards & (from_ <= to_)) | ((~forwards) & (to_ <= from_)):
            dates.append(from_)
            from_ += direction * relativedelta(years=term.years, months=term.months, days=term.days)
            # ... and compete list for fractional periods ...
        if dates[-1] != to_:
            # ... by adding stub or ...
            if allow_stub:
                dates.append(to_)
            # ... by extending last period.
            else:
                dates[-1] = to_
        return dates

    def generate_dates(self) -> List[date]:
        """
        Generate list of schedule days according to the schedule specification, in particular with regards to business
        day convention and calendar given.

        Returns:
            List of schedule dates (including start and end date) adjusted to rolling convention.
        """
        # roll out dates ignoring any business day issues
        if self.__backwards:
            schedule_dates = Schedule._roll_out(self.__end_day, self.__start_day, self.__time_period,
                                                False, self.__stub)
            schedule_dates.reverse()
        else:
            schedule_dates = Schedule._roll_out(self.__start_day, self.__end_day, self.__time_period,
                                                True, self.__stub)

        # adjust according to business day convention
        schedule_dates = [roll_day(schedule_dates[0], self.__calendar, self.__business_day_convention,
                                   schedule_dates[0])] + \
                         [roll_day(schedule_dates[i], self.__calendar, self.__business_day_convention,
                                   schedule_dates[i - 1]) for i in range(1, len(schedule_dates))]

        logger.debug("Schedule dates successfully calculated from '"
                     + str(self.__start_day) + "' to '" + str(self.__end_day) + "'.")
        return schedule_dates


def datetime_to_date(date_time: Union[datetime, date]) -> date:
    """
    Converts type of date from datetime to date or leaves it unchanged if it is already of type date.

    Args:
        date_time (Union[datetime, date]): Date(time) to be converted.

    Returns:
        date: (Potentially) Converted date(time).
    """
    if isinstance(date_time, datetime):
        return date_time.date()
    elif isinstance(date_time, date):
        return date_time
    else:
        raise Exception("'" + str(date_time) + "' must be of type datetime or date!")


def datetime_to_date_list(date_times: Union[List[datetime], List[date]]) -> List[date]:
    """
    Converts types of date  list from datetime to date or leaves it unchanged if they are already of type date.

    Args:
        date_times (Union[List[datetime], List[date]]): List of date(time)s to be converted.

    Returns:
        List[date]: List of (potentially) converted date(time)s.
    """
    if isinstance(date_times, list):
        return [datetime_to_date(date_time) for date_time in date_times]
    else:
        raise Exception("'" + str(date_times) + "' must be a list of type datetime or date!")


def check_start_before_end(start: Union[date, datetime], end: Union[date, datetime]) -> Tuple[date, date]:
    """
    Converts the two input dates from datetime to date format it necessary and checks if the first date is earlier
    than the second one.

    Args:
        start (Union[date, datetime]): Start date
        end (Union[date, datetime]): End date

    Returns:
        Tuple[date, date]: start date, end date
    """
    start_date = datetime_to_date(start)
    end_date = datetime_to_date(end)
    if start_date < end_date:
        return start_date, end_date
    else:
        raise Exception("'" + str(start) + "' must be earlier than '" + str(end) + "'!")


def is_ascending_date_list(start_date: date, dates: List[date], end_date: date,
                           exclude_start: bool = True, exclude_end: bool = False) -> bool:
    """
    Checks if all specified dates, e.g. coupon payment dates, fall between start date and end date. Start and end date
    are excluded dependent on the corresponding boolean flags. Moreover, the dates are verified to be ascending.

    Args:
        start_date (date): First day of the interval the dates shall foll in.
        dates (List[date]): List of dates to be tested if they are ascending and between start and end date.
        end_date (date): Last day of the interval the dates shall foll in.
        exclude_start (bool, optional): True, if start date does not belong to the interval. False, otherwise.
                                        Defaults to True.
        exclude_end (bool, optional): True, if end date does not belong to the interval. False, otherwise.
                                      Defaults to False.

    Returns:
        bool: True, if dates are ascending and fall between the interval given by start and end date. False, otherwise.
    """
    if dates[0] < start_date:
        return False
    elif exclude_start & (dates[0] == start_date):
        return False

    for i in range(1, len(dates)):
        if dates[i] <= dates[i-1]:
            return False

    if dates[-1] > end_date:
        return False
    elif exclude_end & (dates[-1] == end_date):
        return False

    return True


def term_to_period(term: str) -> Period:
    """
    Converts terms, e.g. 1D, 3M, and 5Y, into periods, i.e. Period(0, 0, 1), Period(0, 3, 0), and Period(5, 0, 0),
    respectively.

    Args:
        term (str): Term to be converted into a period.

    Returns:
        Period: Period corresponding to the term specified.
    """
    unit = term[-1]
    measure = int(term[:-1])

    if unit.upper() == 'D':
        period = Period(0, 0, measure)
    elif unit.upper() == 'M':
        period = Period(0, measure, 0)
    elif unit.upper() == 'Y':
        period = Period(measure, 0, 0)
    else:
        raise Exception("Unknown term! Please use: 'D', 'M', or 'Y'.")

    return period


def tenor_to_period(tenor: Union[Period, str]) -> Period:
    """
    Converts a tenor provided as period or string into period format if necessary.

    Args:
        tenor (Union[Period, str]): Tenor to be converted if provided as string.

    Returns:
        Period: Tenor (potentially converted) in(to) period format.
    """
    if isinstance(tenor, Period):
        return tenor
    elif isinstance(tenor, str):
        return term_to_period(tenor)
    else:
        raise Exception("Unknown tenor '" + str(tenor) + "'!")


def calc_end_day(start_day: Union[date, datetime], term: str,
                 business_day_convention: Union[Roll_Convention, str] = None,
                 calendar: Union[HolidayBase, str] = None) -> date:
    """
    Derives the end date of a time period based on the start day the the term given as string, e.g. 1D, 3M, or 5Y.
    If business day convention and corresponding calendar are provided the end date is additionally rolled accordingly.

    Args:
        start_day (Union[date, datetime): Beginning of the time period with length term.
        term (str): Term defining the period from start to end date.
        business_day_convention (Union[Roll_Convention, str], optional): Set of rules defining how to adjust 
                                                                         non-business days. Defaults to None.
        calendar (Union[HolidayBase, str], optional): Holiday calender defining non-business days
                                                      (but not Saturdays and Sundays).
                                                      Defaults to None.

    Returns:
        date: End date potentially adjusted according to the specified business day convention with respect to the given
              calendar.
    """
    start_date = datetime_to_date(start_day)
    period = term_to_period(term)
    end_date = start_date + relativedelta(years=period.years, months=period.months, days=period.days)
    if (business_day_convention is not None) & (calendar is not None):
        end_date = roll_day(end_date, calendar, business_day_convention, start_date)

    return end_date


def calc_start_day(end_day: Union[date, datetime], term: str, 
                   business_day_convention: Union[Roll_Convention, str] = None,
                   calendar: Union[HolidayBase, str] = None) -> date:
    """
    Derives the start date of a time period based on the end day the the term given as string, e.g. 1D, 3M, or 5Y.
    If business day convention and corresponding calendar are provided the start date is additionally rolled
    accordingly.

    Args:
        end_day (Union[date, datetime): End of the time period with length term.
        term (str): Term defining the period from start to end date.
        business_day_convention (Union[Roll_Convention, str], optional): Set of rules defining how to adjust 
                                                                         non-business days. Defaults to None.
        calendar (Union[HolidayBase, str], optional): Holiday calender defining non-business days
                                                      (but not Saturdays and Sundays).
                                                      Defaults to None.
    Returns:
        date: Start date potentially adjusted according to the specified business day convention with respect to the
              given calendar.
    """
    end_date = datetime_to_date(end_day)
    period = term_to_period(term)
    start_date = end_date - relativedelta(years=period.years, months=period.months, days=period.days)
    if (business_day_convention is not None) & (calendar is not None):
        start_date = roll_day(start_date, calendar, business_day_convention)

    return start_date


def last_day_of_month(day: Union[date, datetime]) -> date:
    """
    Derives last day of the month corresponding to the given day.

    Args:
        day (Union[date, datetime]): Day defining month and year for derivation of month's last day.

    Returns:
        date: Date of last day of the corresponding month.
    """
    return date(day.year, day.month, monthrange(day.year, day.month)[1])


def is_last_day_of_month(day: Union[date, datetime]) -> bool:
    """
    Checks if a given day is the last day of the corresponding month.

    Args:
        day (Union[date, datetime]): Day to be checked.

    Returns:
        bool: True, if day is last day of the month, False otherwise.
    """
    return datetime_to_date(day) == last_day_of_month(day)


def is_business_day(day: Union[date, datetime], calendar: Union[HolidayBase, str]) -> bool:
    """
    Checks if a given day is a business day in a given calendar.

    Args:
        day (Union[date, datetime]): Day to be checked.
        calendar (Union[HolidayBase, str]): List of holidays defined by the given calendar.

    Returns:
        bool: True, if day is a business day, False otherwise.
    """
    # TODO: adjust for countries with weekend not on Saturday/Sunday (http://worldmap.workingdays.org/)
    return (day.isoweekday() < 6) & (day not in string_to_calendar(calendar))


def last_business_day_of_month(day: Union[date, datetime], calendar: Union[HolidayBase, str]) -> date:
    """
    Derives the last business day of a month corresponding to a given day based on the holidays set in the calendar.

    Args:
        day (Union[date, datetime]): Day defining month and year for deriving the month's last business day.
        calendar (Union[HolidayBase, str]): List of holidays defined by the given calendar.

    Returns:
        date: Date of last business day of the corresponding month.
    """
    check_day = date(day.year, day.month, monthrange(day.year, day.month)[1])
    while not (is_business_day(check_day, calendar)):
        check_day -= timedelta(days=1)
    return check_day


def is_last_business_day_of_month(day: Union[date, datetime], calendar: Union[HolidayBase, str]) -> bool:
    """
    Checks it the given day is the last business day of the corresponding month.

    Args:
        day (Union[date, datetime]): day to be checked
        calendar (Union[HolidayBase, str]): list of holidays defined by the given calendar

    Returns:
        bool: True if day is last business day of the corresponding month, False otherwise.
    """
    return datetime_to_date(day) == last_business_day_of_month(day, calendar)


def nearest_business_day(day: Union[date, datetime], calendar: Union[HolidayBase, str], following_first: bool = True) \
        -> date:
    """
    Derives nearest business day from given day for a given calendar. If there are equally near days preceding and
    following the flag following_first determines if the following day is preferred to the preceding one.

    Args:
        day (Union[date, datetime]): Day for which the nearest business day is to be found. 
        calendar (Union[HolidayBase, str]): List of holidays given by calendar.
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

    day = datetime_to_date(day)
    while not is_business_day(day, calendar):
        distance += 1
        direction *= -1
        day += direction * timedelta(days=distance)
    return day


def nearest_last_business_day_of_month(day: Union[date, datetime], calendar: Union[HolidayBase, str],
                                       following_first: bool = True) -> date:
    """
    Derives nearest last business day of a month from given day for a given calendar. If there are equally near days
    preceding and following the flag following_first determines if the following day is preferred to the preceding one.

    Args:
        day (Union[date, datetime]): Day for which the nearest last business day of the month is to be found.
        calendar (Union[HolidayBase, str]): List of holidays given by calendar.
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

    day = datetime_to_date(day)
    while not is_last_business_day_of_month(day, calendar):
        distance += 1
        direction *= -1
        day += direction * timedelta(days=distance)
    return day


def next_or_previous_business_day(day: Union[date, datetime], calendar: Union[HolidayBase, str],
                                  following_first: bool) -> date:
    """
    Derives the preceding or following business day to a given day according to a given calendar depending on the flag
    following_first. If the day is already a business day the function directly returns the day.

    Args:
        day (Union[date, datetime]): Day for which the preceding or following business day is to be found.
        calendar (HolidayBase): List of holidays defined by the calendar.
        following_first (bool): Flag to determine in the following (True) or preceding (False) business day is to be
        found.

    Returns:
        date: Preceding or following business day, respectively, or day itself if it is a business day.
    """
    if following_first:
        direction = +1
    else:
        direction = -1

    datetime_to_date(day)
    while not is_business_day(day, calendar):
        day += direction * timedelta(days=1)

    return day


def following(day: Union[date, datetime], calendar: Union[HolidayBase, str]) -> date:
    """
    Derives the (potentially) adjusted business day according to the business day convention 'Following' for a specified
    day with respect to a specific calendar: The adjusted date is the following good business day.

    Args:
        day (Union[date, datetime]): Day to be adjusted according to the roll convention if not already a business day.
        calendar (Union[HolidayBase, str]): Calendar defining holidays additional to weekends.

    Returns:
        date: Adjusted business day according to the roll convention 'Following' with respect to calendar if the day is
              not already a business day. Otherwise the (unadjusted) day is returned.
    """
    return next_or_previous_business_day(day, calendar, True)


def preceding(day: Union[date, datetime], calendar: Union[HolidayBase, str]) -> date:
    """
    Derives the (potentially) adjusted business day according to the business day convention 'Preceding' for a specified
    day with respect to a specific calendar: The adjusted date is the preceding good business day.

    Args:
        day (Union[date, datetime]): Day to be adjusted according to the roll convention if not already a business day.
        calendar (Union[HolidayBase, str]): Calendar defining holidays additional to weekends.

    Returns:
        date: Adjusted business day according to the roll convention 'Preceding' with respect to calendar if the day is
              not already a business day. Otherwise the (unadjusted) day is returned.
    """
    return next_or_previous_business_day(day, calendar, False)


def modified_following(day: Union[date, datetime], calendar: Union[HolidayBase, str]) -> date:
    """
    Derives the (potentially) adjusted business day according to the business day convention 'Modified Following' for a
    specified day with respect to a specific calendar: The adjusted date is the following good business day unless the
    day is in the next calendar month, in which case the adjusted date is the preceding good business day.

    Args:
        day (Union[date, datetime]): Day to be adjusted according to the roll convention if not already a business day.
        calendar (Union[HolidayBase, str]): Calendar defining holidays additional to weekends.

    Returns:
        date: Adjusted business day according to the roll convention 'Modified Following' with respect to calendar if
              the day is not already a business day. Otherwise the (unadjusted) day is returned.
    """
    next_day = next_or_previous_business_day(day, calendar, True)
    if next_day.month > day.month:
        return preceding(day, calendar)
    else:
        return next_day


def modified_following_eom(day: Union[date, datetime], calendar: Union[HolidayBase, str],
                           start_day: Union[date, datetime]) -> date:
    """
    Derives the (potentially) adjusted business day according to the business day convention 'End of Month' for a
    specified day with respect to a specific calendar: Where the start date of a period is on the final business day of
    a particular calendar month, the end date is on the final business day of the end month (not necessarily the
    corresponding date in the end month).

    Args:
        day (Union[date, datetime]): Day to be adjusted according to the roll convention if not already a business day.
        calendar (Union[HolidayBase, str]): Calendar defining holidays additional to weekends.
        start_day (Union[date, datetime]): Day at which the period under consideration begins.

    Returns:
        date: Adjusted business day according to the roll convention 'End of Month' with respect to calendar.
    """
    if isinstance(start_day, date) | isinstance(start_day, datetime):
        if is_last_business_day_of_month(start_day, calendar):
            return nearest_last_business_day_of_month(day, calendar)
        else:
            return modified_following(day, calendar)
    else:
        raise Exception('The roll convention ' + str(Roll_Convention.MODIFIED_FOLLOWING_EOM)
                        + ' cannot be evaluated without a start_day')


def modified_following_bimonthly(day: Union[date, datetime], calendar: Union[HolidayBase, str]) -> date:
    """
    Derives the (potentially) adjusted business day according to the business day convention 'Modified Following
    Bimonthly' for a specified day with respect to a specific calendar: The adjusted date is the following good business
    day unless that day crosses the mid-month (15th) or end of a month, in which case the adjusted date is the preceding
    good business day.

    Args:
        day (Union[date, datetime]): Day to be adjusted according to the roll convention if not already a business day.
        calendar (Union[HolidayBase, str]): Calendar defining holidays additional to weekends.

    Returns:
        date: Adjusted business day according to the roll convention 'Modified Following Bimonthly' with respect to
              calendar if the day is not already a business day. Otherwise the (unadjusted) day is returned.
    """
    next_day = next_or_previous_business_day(day, calendar, True)
    if (next_day.month > day.month) | ((next_day.day > 15) & (day.day <= 15)):
        return preceding(day, calendar)
    else:
        return next_day


def modified_preceding(day: Union[date, datetime], calendar: Union[HolidayBase, str]) -> date:
    """
    Derives the (potentially) adjusted business day according to the business day convention 'Modified Preceding' for a
    specified day with respect to a specific calendar: The adjusted date is the preceding good business day unless the
    day is in the previous calendar month, in which case the adjusted date is the following good business day.

    Args:
        day (Union[date, datetime]): Day to be adjusted according to the roll convention if not already a business day.
        calendar (Union[HolidayBase, str]): Calendar defining holidays additional to weekends.

    Returns:
        date: Adjusted business day according to the roll convention 'Modified Preceding' with respect to calendar if
              the day is not already a business day. Otherwise the (unadjusted) day is returned.
    """
    prev_day = next_or_previous_business_day(day, calendar, False)
    if prev_day.month < day.month:
        return following(day, calendar)
    else:
        return prev_day


# to be used in the switcher (identical argument list)
def unadjusted(day: Union[date, datetime], _) -> date:
    """
    Leaves the day unchanged independent from the fact if it is already a business day.

    Args:
        day (Union[date, datetime]): Day to be adjusted according to the roll convention.
        _: Placeholder for calendar argument.

    Returns:
        date: Unadjusted day.
    """
    return datetime_to_date(day)


def roll_day(day: Union[date, datetime], calendar: Union[HolidayBase, str],
             business_day_convention: Union[Roll_Convention, str], start_day: Union[date, datetime] = None) -> date:
    """
    Adjusts a given day according to the specified business day convention with respect to a given calendar or if the
    given day falls on a Saturday or Sunday. For some roll conventions not only the (end) day to be adjusted but also
    the start day of a period is relevant for the adjustment of the given (end) day.

    Args:
        day (Union[date, datetime]): Day to be adjusted if it is a non-business day.
        calendar (Union[HolidayBase, str]): Holiday calendar defining non-business days (but not Saturdays and Sundays).
        business_day_convention (Union[Roll_Convention, str]): Set of rules defining how to adjust non-business days.
        start_day (Union[date, datetime], optional): Period's start day that may influence the rolling of the (end) day.
                                                     Defaults to None.

    Returns:
        date: Adjusted day.
    """
    roll_convention = roll_convention_to_string(business_day_convention)
    if start_day is not None:
        start_day = datetime_to_date(start_day)

    switcher = {
        'Unadjusted': unadjusted,
        'Following': following,
        'ModifiedFollowing': modified_following,
        'ModifiedFollowingEOM': modified_following_eom,
        'ModifiedFollowingBimonthly': modified_following_bimonthly,
        'Nearest': nearest_business_day,
        'Preceding': preceding,
        'ModifiedPreceding': modified_preceding
    }
    # Get the appropriate roll function from switcher dictionary
    roll_func = switcher.get(roll_convention, lambda: "Business day convention '" + str(business_day_convention)
                                                      + "' is not known!")
    try:
        result = roll_func(day, calendar)
    except TypeError:
        result = roll_func(day, calendar, start_day)

    return result
