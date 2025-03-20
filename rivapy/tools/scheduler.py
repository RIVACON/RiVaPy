from typing import Union, Set, List
import numpy as np
import pandas as pd
import datetime as dt
import rivapy.tools.interfaces as interfaces
from rivapy.tools.datetime_grid import DateTimeGrid


class SimpleSchedule(interfaces.FactoryObject):
    def __init__(
        self,
        start: dt.datetime,
        end: dt.datetime,
        freq: str = "1H",
        weekdays: Set[int] = None,
        hours: Set[int] = None,
        ignore_hours_for_weekdays: Set[int] = None,
        tz: str = None,
    ):
        """Simple schedule of fixed datetime points.

        Args:
                start (dt.datetime): Start of schedule (including this timepoint).
                end (dt.datetime): End of schedule (excluding this timepoint).
                freq (str, optional): Frequency of timepoints. Defaults to '1H'. See documentation for pandas.date_range for further details on freq.
                weekdays (Set[int], optional): List of integers representing the weekdays where the schedule is defined.
                                                                                Integers according to datetime weekdays (0->Monay, 1->Tuesday,...,6->Sunday).
                                                                                If None, all weekdays are used. Defaults to None.
                hours (Set[int], optional): List of hours where schedule is defined. If None, all hours are included. Defaults to None.
                ignor_hours_for_weekdays (Set[int], optional): List of days for which the hours setting is ignored and each hour is considered where the schedule is defined. Defaults to None.
                tz (str or tzinfo): Time zone name for returning localized datetime points, for example ‘Asia/Hong_Kong’.
                                                        By default, the resulting datetime points are timezone-naive. See documentation for pandas.date_range for further details on tz.
        Examples:

        .. highlight:: python
        .. code-block:: python

                >>> simple_schedule = SimpleSchedule(dt.datetime(2023,1,1), dt.datetime(2023,1,1,4,0,0), freq='1H')
                >>> simple_schedule.get_schedule()
                [datetime(2023,1,1,0,0,0), datetime(2023,1,1,1,0,0), datetime(2023,1,1,2,0,0), datetime(2023,1,1,3,0,0)]

                # We include only hours 2 and 3 into schedule
                >>> simple_schedule = SimpleSchedule(dt.datetime(2023,1,1), dt.datetime(2023,1,1,4,0,0), freq='1H', hours=[2,3])
                >>> simple_schedule.get_schedule()
                [datetime.datetime(2023, 1, 1, 2, 0), datetime.datetime(2023, 1, 1, 3, 0)]

                # We restrict further to only mondays as weekdays included
                >>> simple_schedule = SimpleSchedule(dt.datetime(2023,1,1), dt.datetime(2023,1,2,4,0,0), freq='1H', hours=[2,3], weekdays=[0])
                >>> simple_schedule.get_schedule()
                [datetime.datetime(2023, 1, 2, 2, 0), datetime.datetime(2023, 1, 2, 3, 0)]
        """
        self.start = start
        self.end = end
        self.freq = freq
        self.weekdays = weekdays
        self.hours = hours
        self.tz = tz
        self._df = None
        self.ignore_hours_for_weekdays = ignore_hours_for_weekdays

    def _to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "freq": self.freq,
            "weekdays": self.weekdays,
            "hours": self.hours,
            "tz": self.tz,
        }

    def get_schedule(self, refdate: dt.datetime = None) -> np.ndarray:
        """Return vector of datetime values belonging to the schedule.

        Args:
                refdate (dt.datetime): All schedule dates are ignored before this reference date. If None, all schedule dates are returned. Defaults to None.

        Returns:
                np.ndarray: Vector of all datetimepoints of the schedule.
        """
        d_ = pd.date_range(self.start, self.end, freq=self.freq, tz=self.tz, inclusive="left").to_pydatetime()
        if self.weekdays is not None:
            d_ = [d for d in d_ if d.weekday() in self.weekdays]
        if self.hours is not None:
            if self.ignore_hours_for_weekdays is not None:
                d_ = [d for d in d_ if (d.hour in self.hours) or (d.weekday() in self.ignore_hours_for_weekdays)]
            else:
                d_ = [d for d in d_ if d.hour in self.hours]
        if refdate is not None:
            d_ = [d for d in d_ if d >= refdate]
        return d_

    def get_df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.DataFrame(
                {"dates": pd.date_range(self.start, self.end, freq=self.freq, tz=self.tz, inclusive="left").to_pydatetime()}
            ).reset_index()
        return self._df

    def applies(self, dates: DateTimeGrid, index: bool) -> List[Union[bool, int]]:
        dates.dates

    def get_params(self) -> dict:
        """Return all params as json serializable dictionary.

        Returns:
                dict: Dictionary of all parameters.
        """
        return {
            "start": self.start,
            "end": self.end,
            "freq": self.freq,
            "weekdays": self.weekdays,
            "hours": self.hours,
            "ignore_hours_for_weekdays": self.ignore_hours_for_weekdays,
            "tz": self.tz,
        }

    @staticmethod
    def _create_sample(n_samples: int, seed: int = None, ref_date=None):
        if ref_date is None:
            ref_date = dt.datetime(1980, 1, 1)
        if seed is not None:
            np.random.seed(seed)
        result = []
        for i in range(n_samples):
            start = ref_date + dt.timedelta(days=np.random.randint(0, 100))
            end = start + +dt.timedelta(days=np.random.randint(5, 365))
            result.append(SimpleSchedule(start=start, end=end))
        return result


class PeakSchedule(SimpleSchedule):
    def __init__(self, start: dt.datetime, end: dt.datetime, tz: str = None):
        """Scheduler, which returns the peak time grid between the start and end date times.

        Args:
                start (dt.datetime): Start of schedule (including this timepoint).
                end (dt.datetime): End of schedule (excluding this timepoint).
                tz (str or tzinfo): Time zone name for returning localized datetime points, for example ‘Asia/Hong_Kong’.
                                                        By default, the resulting datetime points are timezone-naive. See documentation for pandas.date_range for further details on tz.
        Examples:

        .. highlight:: python
        .. code-block:: python

                >>> peak_schedule = PeakSchedule(dt.datetime(2023,1,5), dt.datetime(2023,1,6))
                >>> peak_schedule.get_schedule()
                [datetime.datetime(2023, 1, 5, 8, 0),
                 datetime.datetime(2023, 1, 5, 9, 0),
                 datetime.datetime(2023, 1, 5, 10, 0),
                 datetime.datetime(2023, 1, 5, 11, 0),
                 datetime.datetime(2023, 1, 5, 12, 0),
                 datetime.datetime(2023, 1, 5, 13, 0),
                 datetime.datetime(2023, 1, 5, 14, 0),
                 datetime.datetime(2023, 1, 5, 15, 0),
                 datetime.datetime(2023, 1, 5, 16, 0),
                 datetime.datetime(2023, 1, 5, 17, 0),
                 datetime.datetime(2023, 1, 5, 18, 0),
                 datetime.datetime(2023, 1, 5, 19, 0)]
        """
        super().__init__(
            start=start,
            end=end,
            freq="h",
            hours=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            weekdays=[0, 1, 2, 3, 4],
            tz=tz,
        )


class OffPeakSchedule(SimpleSchedule):
    def __init__(self, start: dt.datetime, end: dt.datetime, tz: str = None):
        """Scheduler, which returns the offpeak time grid between the start and end date times.

        Args:
                start (dt.datetime): Start of schedule (including this timepoint).
                end (dt.datetime): End of schedule (excluding this timepoint).
                tz (str or tzinfo): Time zone name for returning localized datetime points, for example ‘Asia/Hong_Kong’.
                                                        By default, the resulting datetime points are timezone-naive. See documentation for pandas.date_range for further details on tz.
        Examples:

        .. highlight:: python
        .. code-block:: python

                >>> offpeak_schedule = OffPeakSchedule(dt.datetime(2023,1,5), dt.datetime(2023,1,6))
                >>> offpeak_schedule.get_schedule()
                [datetime.datetime(2023, 1, 5, 0, 0),
                 datetime.datetime(2023, 1, 5, 1, 0),
                 datetime.datetime(2023, 1, 5, 2, 0),
                 datetime.datetime(2023, 1, 5, 3, 0),
                 datetime.datetime(2023, 1, 5, 4, 0),
                 datetime.datetime(2023, 1, 5, 5, 0),
                 datetime.datetime(2023, 1, 5, 6, 0),
                 datetime.datetime(2023, 1, 5, 7, 0),
                 datetime.datetime(2023, 1, 5, 20, 0),
                 datetime.datetime(2023, 1, 5, 21, 0),
                 datetime.datetime(2023, 1, 5, 22, 0),
                 datetime.datetime(2023, 1, 5, 23, 0)]
        """
        super().__init__(
            start=start,
            end=end,
            freq="h",
            hours=[0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23],
            ignore_hours_for_weekdays=[5, 6],
            tz=tz,
        )


class GasSchedule(SimpleSchedule):
    def __init__(self, start: dt.datetime, end: dt.datetime, tz: str = None):
        """Scheduler, which returns the gas day time grid (from 6 am to 6 am) between the start and end date times.

        Args:
                start (dt.datetime): Start of schedule (including this timepoint).
                end (dt.datetime): End of schedule (excluding this timepoint).
                tz (str or tzinfo): Time zone name for returning localized datetime points, for example ‘Asia/Hong_Kong’.
                                                        By default, the resulting datetime points are timezone-naive. See documentation for pandas.date_range for further details on tz.
        Examples:

        .. highlight:: python
        .. code-block:: python

                >>> gas_schedule = GasSchedule(dt.datetime(2023,1,5), dt.datetime(2023,1,7))
                >>> gas_schedule.get_schedule()
                [datetime.datetime(2023, 1, 5, 6, 0), datetime.datetime(2023, 1, 6, 6, 0)]
        """
        super().__init__(
            start=start,
            end=end,
            freq="h",
            hours=[6],
            weekdays=[0, 1, 2, 3, 4, 5, 6],
            tz=tz,
        )
