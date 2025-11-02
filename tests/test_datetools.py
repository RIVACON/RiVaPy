from unittest import main, TestCase

import holidays
from matplotlib.dates import relativedelta
from rivapy.tools.datetools import calc_end_day, is_business_day, roll_day, Period, Schedule, DayCounter
from rivapy.tools.enums import RollConvention, DayCounterType, RollRule
import calendar
from rivapy.tools.datetools import (
    _date_to_datetime,
    _is_ambiguous_date,
    calc_end_day,
    _is_IMM_date,
    calc_start_day,
    last_day_of_month,
    is_last_day_of_month,
    next_IMM_date,
    last_business_day_of_month,
    is_last_business_day_of_month,
    nearest_business_day,
    nearest_last_business_day_of_month,
    next_or_previous_business_day,
)
from datetime import date, datetime
from rivapy.tools.holidays_compat import DE, ECB


class DayCounterTests(TestCase):

    def test_yf(self):
        d1 = datetime(2023, 1, 1)
        d2 = datetime(2024, 1, 1)
        self.assertAlmostEqual(DayCounter.yf_Act365Fixed(d1, d2), 1.0, delta=1e-5)
        dc = DayCounter(DayCounterType.Act365Fixed)
        self.assertAlmostEqual(DayCounter.yf_Act365Fixed(d1, d2), dc.yf(d1, d2), delta=1e-5)

        d1 = datetime(2024, 12, 1)
        d2 = datetime(2025, 2, 1)
        self.assertAlmostEqual(DayCounter.yf_ActAct(d1, d2), 0.16963096040122763, delta=1e-5)
        d1 = date(2024, 1, 1)
        d2 = date(2025, 1, 1)
        self.assertAlmostEqual(DayCounter.yf_Act360(d1, d2), 1.0166666666666666, delta=1e-5)
        self.assertAlmostEqual(DayCounter.yf_Act365Fixed(d1, d2), 1.0027397260273974, delta=1e-5)
        coupon_schedule = [date(2024, 5, 1), date(2024, 11, 1)]
        self.assertEqual(DayCounter.yf_ActActICMA(date(2024, 5, 1), date(2024, 5, 31), coupon_schedule, coupon_frequency=2), 30 / 368)

        self.assertAlmostEqual(DayCounter.yf_30360ISDA(date(2025, 1, 1), date(2025, 2, 1)), 0.08333333333333333, delta=1e-5)
        self.assertAlmostEqual(DayCounter.yf_30360ISDA(date(2024, 12, 31), date(2025, 1, 31)), 0.08333333333333333, delta=1e-5)
        self.assertAlmostEqual(DayCounter.yf_30360ISDA(date(2025, 4, 29), date(2025, 5, 30)), 0.08611111111111111, delta=1e-5)
        self.assertAlmostEqual(DayCounter.yf_30360ISDA(date(2025, 4, 30), date(2025, 5, 31)), 0.08333333333333333, delta=1e-5)

        self.assertAlmostEqual(DayCounter.yf_30E360(date(2024, 12, 31), date(2025, 1, 31)), 0.08333333333333333, delta=1e-5)
        self.assertAlmostEqual(DayCounter.yf_30E360(date(2024, 12, 31), date(2025, 1, 30)), 0.08333333333333333, delta=1e-5)
        self.assertAlmostEqual(DayCounter.yf_30E360(date(2024, 12, 30), date(2025, 1, 31)), 0.08333333333333333, delta=1e-5)
        self.assertAlmostEqual(DayCounter.yf_30E360(date(2024, 12, 30), date(2025, 1, 30)), 0.08333333333333333, delta=1e-5)

        self.assertAlmostEqual(DayCounter.yf_30U360(date(2024, 2, 29), date(2025, 2, 28)), 1.0, delta=1e-5)
        self.assertAlmostEqual(DayCounter.yf_30U360(date(2024, 2, 28), date(2025, 2, 28)), 1.0, delta=1e-5)
        self.assertAlmostEqual(DayCounter.yf_30U360(date(2023, 2, 28), date(2024, 2, 28)), 0.9944444444444445, delta=1e-5)


class PeriodTests(TestCase):

    def test_period(self):
        p = Period(1975, 8, 22)
        self.assertEqual(p.years, 1975)
        self.assertEqual(p.months, 8)
        self.assertEqual(p.days, 22)
        self.assertTrue(p == Period(1975, 8, 22))
        p = Period.from_string("T/N")
        self.assertEqual(p.years, 0)
        self.assertEqual(p.months, 0)
        self.assertEqual(p.days, 1)
        p = Period.from_string("O/N")
        self.assertEqual(p.years, 0)
        self.assertEqual(p.months, 0)
        self.assertEqual(p.days, 1)
        p = Period.from_string("5D")
        self.assertEqual(p.years, 0)
        self.assertEqual(p.months, 0)
        self.assertEqual(p.days, 5)
        P = Period.from_string("3M")
        self.assertEqual(P.years, 0)
        self.assertEqual(P.months, 3)
        self.assertEqual(P.days, 0)
        p = Period.from_string("2Y")
        self.assertEqual(p.years, 2)
        self.assertEqual(p.months, 0)
        self.assertEqual(p.days, 0)
        with self.assertRaises(Exception):
            Period.from_string("1W")


class OGandOwnTests(TestCase):
    # These tests are shall reproduce the results given in [Chapter 4](https://usermanual.wiki/Document/interestrateinstrumentsandmarketconventionsguide.1425400940/view)
    # of Interest Rate Instruments and Market Conventions Guide by OpenGamma
    holidays_target2 = holidays.ECB(years=[2011, 2012])
    holidays_de = holidays.DE(years=1997)

    # OpenGamma Tests
    def test_rolls(self):
        start_date = date(2011, 8, 18)
        end_date = start_date + relativedelta(months=1)
        self.assertEqual(roll_day(end_date, self.holidays_target2, RollConvention.FOLLOWING), datetime(2011, 9, 19))
        self.assertEqual(roll_day(end_date, self.holidays_target2, RollConvention.PRECEDING), datetime(2011, 9, 16))
        start_date = date(2011, 6, 30)
        end_date = start_date + relativedelta(months=1)
        self.assertEqual(roll_day(end_date, self.holidays_target2, RollConvention.MODIFIED_FOLLOWING), datetime(2011, 7, 29))
        self.assertEqual(roll_day(end_date, self.holidays_target2, RollConvention.FOLLOWING), datetime(2011, 8, 1))
        self.assertEqual(roll_day(end_date, self.holidays_target2, RollConvention.MODIFIED_FOLLOWING), datetime(2011, 7, 29))
        start_date = date(2011, 9, 15)
        end_date = start_date + relativedelta(months=1)
        self.assertEqual(roll_day(end_date, self.holidays_target2, RollConvention.MODIFIED_FOLLOWING_BIMONTHLY), datetime(2011, 10, 14))
        self.assertEqual(roll_day(end_date, self.holidays_target2, RollConvention.FOLLOWING), datetime(2011, 10, 17))
        start_date = date(2011, 2, 28)
        end_date = start_date + relativedelta(months=1)
        self.assertEqual(roll_day(end_date, self.holidays_target2, RollConvention.MODIFIED_FOLLOWING_EOM, start_date), datetime(2011, 3, 31))
        start_date = date(2011, 4, 29)
        end_date = start_date + relativedelta(months=1)
        self.assertEqual(roll_day(end_date, self.holidays_target2, RollConvention.MODIFIED_FOLLOWING_EOM, start_date), datetime(2011, 5, 31))
        start_date = date(2012, 2, 28)
        end_date = start_date + relativedelta(months=1)
        self.assertEqual(roll_day(end_date, self.holidays_target2, RollConvention.MODIFIED_FOLLOWING_EOM, start_date), datetime(2012, 3, 28))

    # Own Tests

    @staticmethod
    def weekday(n):
        name = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
        return name.get(n, "weekday(" + str(n) + ") is an invalid day of the week!")

    def test_holidays(self):
        for holiday_date, holiday_name in sorted(self.holidays_de.items()):
            # self.assertTrue(
            #     self.weekday(holiday_date.isoweekday()) in [6, 7]
            #     or holiday_name
            #     in [
            #         "Neujahr",
            #         "Karfreitag",
            #         "Ostermontag",
            #         "Erster Mai",
            #         "Christi Himmelfahrt",
            #         "Pfingstmontag",
            #         "Tag der Deutschen Einheit",
            #         "Erster Weihnachtstag",
            #         "Zweiter Weihnachtstag",
            #     ],
            #     f"Unexpected holiday {holiday_name} on {holiday_date} ({self.weekday(holiday_date.isoweekday())})",
            # )
            self.assertTrue(
                holiday_date
                in [
                    date(1997, 1, 1),
                    date(1997, 3, 28),
                    date(1997, 3, 31),
                    date(1997, 5, 1),
                    date(1997, 5, 8),
                    date(1997, 5, 19),
                    date(1997, 10, 3),
                    date(1997, 12, 25),
                    date(1997, 12, 26),
                ]
            )


class RollingSchedulingTests(TestCase):
    def test_roll_day(self):
        holidays_de = DE()

        # business days are unchanged for all roll conventions
        roll_conventions = [roll_convention.value for roll_convention in RollConvention]
        roll_conventions.pop(2)  # remove 'ModifiedFollowingEOM' as it needs a start date
        for roll_convention in roll_conventions:
            self.assertEqual(roll_day(date(1997, 1, 2), holidays_de, roll_convention), datetime(1997, 1, 2))

        # test UNADJUSTED
        roll_convention = RollConvention.UNADJUSTED
        self.assertEqual(roll_day(datetime(1997, 5, 1), holidays_de, roll_convention), datetime(1997, 5, 1))
        self.assertEqual(roll_day(datetime(1997, 7, 6), holidays_de, roll_convention), datetime(1997, 7, 6))

        # test FOLLOWING
        roll_convention = RollConvention.FOLLOWING
        self.assertEqual(roll_day(datetime(1997, 5, 1), holidays_de, roll_convention), datetime(1997, 5, 2))
        self.assertEqual(roll_day(datetime(1997, 7, 5), holidays_de, roll_convention), datetime(1997, 7, 7))
        self.assertEqual(roll_day(datetime(1997, 5, 17), holidays_de, roll_convention), datetime(1997, 5, 20))
        self.assertEqual(roll_day(datetime(1997, 12, 25), holidays_de, roll_convention), datetime(1997, 12, 29))
        self.assertEqual(roll_day(datetime(1997, 3, 28), holidays_de, roll_convention), datetime(1997, 4, 1))

        # test MODIFIED_FOLLOWING
        roll_convention = RollConvention.MODIFIED_FOLLOWING
        self.assertEqual(roll_day(datetime(1997, 5, 1), holidays_de, roll_convention), datetime(1997, 5, 2))
        self.assertEqual(roll_day(datetime(1997, 12, 25), holidays_de, roll_convention), datetime(1997, 12, 29))
        self.assertEqual(roll_day(datetime(1997, 8, 30), holidays_de, roll_convention), datetime(1997, 8, 29))
        self.assertEqual(roll_day(datetime(1997, 8, 31), holidays_de, roll_convention), datetime(1997, 8, 29))
        self.assertEqual(roll_day(datetime(1997, 3, 28), holidays_de, roll_convention), datetime(1997, 3, 27))

        # test MODIFIED_FOLLOWING_BIMONTHLY
        roll_convention = RollConvention.MODIFIED_FOLLOWING_BIMONTHLY
        self.assertEqual(roll_day(datetime(1997, 5, 1), holidays_de, roll_convention), datetime(1997, 5, 2))
        self.assertEqual(roll_day(datetime(1997, 12, 25), holidays_de, roll_convention), datetime(1997, 12, 29))
        self.assertEqual(roll_day(datetime(1997, 8, 30), holidays_de, roll_convention), datetime(1997, 8, 29))
        self.assertEqual(roll_day(datetime(1997, 8, 31), holidays_de, roll_convention), datetime(1997, 8, 29))
        self.assertEqual(roll_day(datetime(1997, 3, 28), holidays_de, roll_convention), datetime(1997, 3, 27))
        self.assertEqual(roll_day(datetime(1997, 2, 15), holidays_de, roll_convention), datetime(1997, 2, 14))
        self.assertEqual(roll_day(datetime(1997, 2, 16), holidays_de, roll_convention), datetime(1997, 2, 17))

        # test NEAREST
        roll_convention = RollConvention.NEAREST
        self.assertEqual(roll_day(datetime(1997, 5, 1), holidays_de, roll_convention), datetime(1997, 5, 2))
        self.assertEqual(roll_day(datetime(1997, 7, 5), holidays_de, roll_convention), datetime(1997, 7, 4))
        self.assertEqual(roll_day(datetime(1997, 7, 6), holidays_de, roll_convention), datetime(1997, 7, 7))
        self.assertEqual(roll_day(datetime(1997, 5, 18), holidays_de, roll_convention), datetime(1997, 5, 20))
        self.assertEqual(roll_day(datetime(1997, 3, 29), holidays_de, roll_convention), datetime(1997, 3, 27))
        self.assertEqual(roll_day(datetime(1997, 3, 30), holidays_de, roll_convention), datetime(1997, 4, 1))

        # test PRECEDING
        roll_convention = RollConvention.PRECEDING
        self.assertEqual(roll_day(datetime(1997, 5, 1), holidays_de, roll_convention), datetime(1997, 4, 30))
        self.assertEqual(roll_day(datetime(1997, 7, 6), holidays_de, roll_convention), datetime(1997, 7, 4))
        self.assertEqual(roll_day(datetime(1997, 5, 19), holidays_de, roll_convention), datetime(1997, 5, 16))
        self.assertEqual(roll_day(datetime(1997, 12, 28), holidays_de, roll_convention), datetime(1997, 12, 24))
        self.assertEqual(roll_day(datetime(1997, 3, 31), holidays_de, roll_convention), datetime(1997, 3, 27))

        # test MODIFIED_PRECEDING
        roll_convention = RollConvention.MODIFIED_PRECEDING
        self.assertEqual(roll_day(datetime(1997, 5, 1), holidays_de, roll_convention), datetime(1997, 5, 2))
        self.assertEqual(roll_day(datetime(1997, 7, 5), holidays_de, roll_convention), datetime(1997, 7, 4))
        self.assertEqual(roll_day(datetime(1997, 7, 6), holidays_de, roll_convention), datetime(1997, 7, 4))
        self.assertEqual(roll_day(datetime(1997, 3, 2), holidays_de, roll_convention), datetime(1997, 3, 3))

        # test MODIFIED_FOLLOWING_EOM
        roll_convention = RollConvention.MODIFIED_FOLLOWING_EOM
        self.assertEqual(roll_day(datetime(1997, 3, 28), holidays_de, roll_convention, datetime(1997, 2, 28)), datetime(1997, 3, 27))
        self.assertEqual(roll_day(datetime(1997, 4, 26), holidays_de, roll_convention, datetime(1997, 3, 26)), datetime(1997, 4, 28))
        self.assertEqual(roll_day(datetime(1997, 4, 27), holidays_de, roll_convention, datetime(1997, 3, 27)), datetime(1997, 4, 30))
        self.assertEqual(roll_day(datetime(1997, 6, 29), holidays_de, roll_convention, datetime(1997, 5, 29)), datetime(1997, 6, 30))
        self.assertEqual(roll_day(datetime(1997, 7, 30), holidays_de, roll_convention, datetime(1997, 6, 30)), datetime(1997, 7, 31))
        self.assertEqual(roll_day(datetime(1997, 8, 31), holidays_de, roll_convention, datetime(1997, 7, 31)), datetime(1997, 8, 29))
        self.assertEqual(roll_day(datetime(1997, 9, 28), holidays_de, roll_convention, datetime(1997, 8, 28)), datetime(1997, 9, 29))
        self.assertEqual(roll_day(datetime(1997, 9, 29), holidays_de, roll_convention, datetime(1997, 8, 29)), datetime(1997, 9, 30))
        self.assertEqual(roll_day(datetime(1997, 11, 30), holidays_de, roll_convention, datetime(1997, 10, 30)), datetime(1997, 11, 28))
        self.assertEqual(roll_day(datetime(1997, 12, 28), holidays_de, roll_convention, datetime(1997, 11, 28)), datetime(1997, 12, 31))

    def test_schedule_generation(self):
        holidays_de = ECB()
        # test roll_out using differnt roll conventions
        self.assertEqual(
            Schedule._roll_out(datetime(2024, 1, 30), datetime(2024, 7, 31), Period(0, 1, 0), False, False, RollRule.NONE),
            [
                datetime(2024, 1, 30),
                datetime(2024, 2, 29),
                datetime(2024, 3, 29),
                datetime(2024, 4, 29),
                datetime(2024, 5, 29),
                datetime(2024, 6, 29),
                datetime(2024, 7, 29),
                datetime(2024, 7, 31),
            ],
        ),
        self.assertEqual(
            Schedule._roll_out(datetime(2024, 1, 30), datetime(2024, 7, 31), Period(0, 1, 0), False, True, RollRule.NONE),
            [
                datetime(2024, 1, 30),
                datetime(2024, 2, 29),
                datetime(2024, 3, 29),
                datetime(2024, 4, 29),
                datetime(2024, 5, 29),
                datetime(2024, 6, 29),
                datetime(2024, 7, 31),
            ],
        ),
        dates = Schedule._roll_out(datetime(2024, 7, 31), datetime(2024, 1, 30), Period(0, 1, 0), True, False, RollRule.NONE)
        dates.reverse()
        self.assertEqual(
            dates,
            [
                datetime(2024, 7, 31),
                datetime(2024, 6, 30),
                datetime(2024, 5, 30),
                datetime(2024, 4, 30),
                datetime(2024, 3, 30),
                datetime(2024, 2, 29),
                datetime(2024, 1, 30),
            ],
        ),
        dates = Schedule._roll_out(datetime(2024, 7, 31), datetime(2024, 1, 30), Period(0, 1, 0), True, True, RollRule.NONE)
        dates.reverse()
        self.assertEqual(
            dates,
            [
                datetime(2024, 7, 31),
                datetime(2024, 6, 30),
                datetime(2024, 5, 30),
                datetime(2024, 4, 30),
                datetime(2024, 3, 30),
                datetime(2024, 1, 30),
            ],
        ),
        self.assertEqual(
            Schedule._roll_out(datetime(2024, 1, 30), datetime(2024, 7, 31), Period(0, 1, 0), False, False, RollRule.EOM),
            [
                datetime(2024, 1, 31),
                datetime(2024, 2, 29),
                datetime(2024, 3, 31),
                datetime(2024, 4, 30),
                datetime(2024, 5, 31),
                datetime(2024, 6, 30),
                datetime(2024, 7, 31),
            ],
        ),
        self.assertEqual(
            Schedule._roll_out(datetime(2024, 1, 30), datetime(2024, 7, 31), Period(0, 1, 0), False, True, RollRule.EOM),
            [
                datetime(2024, 1, 31),
                datetime(2024, 2, 29),
                datetime(2024, 3, 31),
                datetime(2024, 4, 30),
                datetime(2024, 5, 31),
                datetime(2024, 6, 30),
                datetime(2024, 7, 31),
            ],
        ),
        dates = Schedule._roll_out(datetime(2024, 7, 31), datetime(2024, 1, 30), Period(0, 1, 0), True, False, RollRule.EOM)
        dates.reverse()
        self.assertEqual(
            dates,
            [
                datetime(2024, 7, 31),
                datetime(2024, 6, 30),
                datetime(2024, 5, 31),
                datetime(2024, 4, 30),
                datetime(2024, 3, 31),
                datetime(2024, 2, 29),
                datetime(2024, 1, 31),
            ],
        ),
        dates = Schedule._roll_out(datetime(2024, 7, 31), datetime(2024, 1, 30), Period(0, 1, 0), True, True, RollRule.EOM)
        dates.reverse()
        self.assertEqual(
            dates,
            [
                datetime(2024, 7, 31),
                datetime(2024, 6, 30),
                datetime(2024, 5, 31),
                datetime(2024, 4, 30),
                datetime(2024, 3, 31),
                datetime(2024, 2, 29),
                datetime(2024, 1, 31),
            ],
        ),
        self.assertEqual(
            Schedule._roll_out(datetime(2023, 11, 30), datetime(2024, 9, 30), Period(0, 3, 0), False, True, RollRule.EOM),
            [
                datetime(2023, 11, 30),
                datetime(2024, 2, 29),
                datetime(2024, 5, 31),
                datetime(2024, 9, 30),
            ],
        ),
        self.assertEqual(
            Schedule._roll_out(datetime(2024, 1, 30), datetime(2024, 7, 31), Period(0, 1, 0), False, False, RollRule.DOM),
            [
                datetime(2024, 1, 30),
                datetime(2024, 2, 29),
                datetime(2024, 3, 30),
                datetime(2024, 4, 30),
                datetime(2024, 5, 30),
                datetime(2024, 6, 30),
                datetime(2024, 7, 30),
                datetime(2024, 7, 31),
            ],
        ),
        self.assertEqual(
            Schedule._roll_out(datetime(2024, 1, 30), datetime(2024, 7, 31), Period(0, 1, 0), False, True, RollRule.DOM),
            [
                datetime(2024, 1, 30),
                datetime(2024, 2, 29),
                datetime(2024, 3, 30),
                datetime(2024, 4, 30),
                datetime(2024, 5, 30),
                datetime(2024, 6, 30),
                datetime(2024, 7, 31),
            ],
        ),
        self.assertEqual(
            Schedule._roll_out(datetime(2024, 1, 29), datetime(2024, 7, 31), Period(0, 1, 0), False, True, RollRule.DOM),
            [
                datetime(2024, 1, 29),
                datetime(2024, 2, 29),
                datetime(2024, 3, 29),
                datetime(2024, 4, 29),
                datetime(2024, 5, 29),
                datetime(2024, 6, 29),
                datetime(2024, 7, 31),
            ],
        ),
        dates = Schedule._roll_out(datetime(2024, 7, 31), datetime(2024, 1, 30), Period(0, 1, 0), True, False, RollRule.DOM)
        dates.reverse()
        self.assertEqual(
            dates,
            [
                datetime(2024, 7, 31),
                datetime(2024, 6, 30),
                datetime(2024, 5, 31),
                datetime(2024, 4, 30),
                datetime(2024, 3, 31),
                datetime(2024, 2, 29),
                datetime(2024, 1, 31),
                datetime(2024, 1, 30),
            ],
        ),
        dates = Schedule._roll_out(datetime(2024, 7, 31), datetime(2024, 1, 30), Period(0, 1, 0), True, True, RollRule.DOM)
        dates.reverse()
        self.assertEqual(
            dates,
            [
                datetime(2024, 7, 31),
                datetime(2024, 6, 30),
                datetime(2024, 5, 31),
                datetime(2024, 4, 30),
                datetime(2024, 3, 31),
                datetime(2024, 2, 29),
                datetime(2024, 1, 30),
            ],
        ),
        self.assertEqual(
            Schedule._roll_out(datetime(2023, 12, 18), datetime(2024, 7, 31), Period(0, 3, 0), False, False, RollRule.IMM),
            [
                datetime(2023, 12, 20),
                datetime(2024, 3, 20),
                datetime(2024, 6, 19),
                datetime(2024, 7, 31),
            ],
        ),
        self.assertEqual(
            Schedule._roll_out(datetime(2023, 12, 18), datetime(2024, 7, 31), Period(0, 3, 0), False, True, RollRule.IMM),
            [
                datetime(2023, 12, 20),
                datetime(2024, 3, 20),
                datetime(2024, 7, 31),
            ],
        ),
        self.assertEqual(
            Schedule._roll_out(datetime(2023, 12, 18), datetime(2024, 7, 31), Period(0, 1, 0), False, True, RollRule.IMM),
            [
                datetime(2023, 12, 20),
                datetime(2024, 3, 20),
                datetime(2024, 7, 31),
            ],
        ),
        # self.assertEqual(
        #     Schedule._roll_out(datetime(2024, 7, 31), datetime(2023, 12, 18), Period(0, 3, 0), True, False, RollRule.IMM),
        #     [
        #         datetime(2024, 6, 19),
        #         datetime(2024, 3, 20),
        #         datetime(2023, 12, 20),
        #         datetime(2023, 12, 18),
        #     ],
        # ),
        # self.assertEqual(
        #     Schedule._roll_out(datetime(2024, 7, 31), datetime(2023, 12, 18), Period(0, 3, 0), True, True, RollRule.IMM),
        #     [
        #         datetime(2024, 6, 19),
        #         datetime(2024, 3, 20),
        #         datetime(2023, 12, 18),
        #     ],
        # ),
        self.assertEqual(
            Schedule._roll_out(datetime(2023, 12, 21), datetime(2024, 2, 1), Period(0, 1, 0), False, True, RollRule.IMM),
            [],
        ),
        # generate_dates with different periods and business day conventions
        # sufficient to test with two different business day conventions? should be ok, if application of bdc is tested properly
        self.assertEqual(
            Schedule(
                datetime(2020, 8, 21), datetime(2021, 8, 21), Period(0, 3, 0), True, True, RollConvention.UNADJUSTED, holidays_de
            ).generate_dates(False),
            [datetime(2020, 8, 21), datetime(2020, 11, 21), datetime(2021, 2, 21), datetime(2021, 5, 21), datetime(2021, 8, 21)],
        )
        self.assertEqual(
            Schedule(
                datetime(2020, 8, 21), datetime(2021, 8, 21), Period(0, 3, 0), True, False, RollConvention.UNADJUSTED, holidays_de
            ).generate_dates(False),
            [datetime(2020, 8, 21), datetime(2020, 11, 21), datetime(2021, 2, 21), datetime(2021, 5, 21), datetime(2021, 8, 21)],
        )
        self.assertEqual(
            Schedule(
                datetime(2020, 8, 21), datetime(2021, 8, 21), Period(0, 3, 0), False, True, RollConvention.UNADJUSTED, holidays_de
            ).generate_dates(False),
            [datetime(2020, 8, 21), datetime(2020, 11, 21), datetime(2021, 2, 21), datetime(2021, 5, 21), datetime(2021, 8, 21)],
        )
        self.assertEqual(
            Schedule(
                datetime(2020, 8, 21), datetime(2021, 8, 21), Period(0, 3, 0), False, False, RollConvention.UNADJUSTED, holidays_de
            ).generate_dates(False),
            [datetime(2020, 8, 21), datetime(2020, 11, 21), datetime(2021, 2, 21), datetime(2021, 5, 21), datetime(2021, 8, 21)],
        )

        self.assertEqual(
            Schedule(
                datetime(2020, 8, 21), datetime(2021, 8, 21), Period(0, 5, 0), True, True, RollConvention.UNADJUSTED, holidays_de
            ).generate_dates(False),
            [datetime(2020, 8, 21), datetime(2021, 3, 21), datetime(2021, 8, 21)],
        )
        self.assertEqual(
            Schedule(
                datetime(2020, 8, 21), datetime(2021, 8, 21), Period(0, 5, 0), True, False, RollConvention.UNADJUSTED, holidays_de
            ).generate_dates(False),
            [datetime(2020, 8, 21), datetime(2020, 10, 21), datetime(2021, 3, 21), datetime(2021, 8, 21)],
        )
        self.assertEqual(
            Schedule(
                datetime(2020, 8, 21), datetime(2021, 8, 21), Period(0, 5, 0), False, True, RollConvention.UNADJUSTED, holidays_de
            ).generate_dates(False),
            [datetime(2020, 8, 21), datetime(2021, 1, 21), datetime(2021, 8, 21)],
        )
        self.assertEqual(
            Schedule(
                datetime(2020, 8, 21), datetime(2021, 8, 21), Period(0, 5, 0), False, False, RollConvention.UNADJUSTED, holidays_de
            ).generate_dates(False),
            [datetime(2020, 8, 21), datetime(2021, 1, 21), datetime(2021, 6, 21), datetime(2021, 8, 21)],
        )

        self.assertEqual(
            Schedule(
                datetime(2020, 8, 21), datetime(2021, 8, 21), Period(0, 3, 0), True, True, RollConvention.MODIFIED_FOLLOWING, holidays_de
            ).generate_dates(False),
            [datetime(2020, 8, 21), datetime(2020, 11, 23), datetime(2021, 2, 22), datetime(2021, 5, 21), datetime(2021, 8, 23)],
        )
        self.assertEqual(
            Schedule(
                datetime(2020, 8, 21), datetime(2021, 8, 21), Period(0, 3, 0), True, False, RollConvention.MODIFIED_FOLLOWING, holidays_de
            ).generate_dates(False),
            [datetime(2020, 8, 21), datetime(2020, 11, 23), datetime(2021, 2, 22), datetime(2021, 5, 21), datetime(2021, 8, 23)],
        )
        self.assertEqual(
            Schedule(
                datetime(2020, 8, 21), datetime(2021, 8, 21), Period(0, 3, 0), False, True, RollConvention.MODIFIED_FOLLOWING, holidays_de
            ).generate_dates(False),
            [datetime(2020, 8, 21), datetime(2020, 11, 23), datetime(2021, 2, 22), datetime(2021, 5, 21), datetime(2021, 8, 23)],
        )
        self.assertEqual(
            Schedule(
                datetime(2020, 8, 21), datetime(2021, 8, 21), Period(0, 3, 0), False, False, RollConvention.MODIFIED_FOLLOWING, holidays_de
            ).generate_dates(False),
            [datetime(2020, 8, 21), datetime(2020, 11, 23), datetime(2021, 2, 22), datetime(2021, 5, 21), datetime(2021, 8, 23)],
        )

        self.assertEqual(
            Schedule(
                datetime(2020, 8, 21), datetime(2021, 8, 21), Period(0, 5, 0), True, True, RollConvention.MODIFIED_FOLLOWING, holidays_de
            ).generate_dates(False),
            [datetime(2020, 8, 21), datetime(2021, 3, 22), datetime(2021, 8, 23)],
        )
        self.assertEqual(
            Schedule(
                datetime(2020, 8, 21), datetime(2021, 8, 21), Period(0, 5, 0), True, False, RollConvention.MODIFIED_FOLLOWING, holidays_de
            ).generate_dates(False),
            [datetime(2020, 8, 21), datetime(2020, 10, 21), datetime(2021, 3, 22), datetime(2021, 8, 23)],
        )
        self.assertEqual(
            Schedule(
                datetime(2020, 8, 21), datetime(2021, 8, 21), Period(0, 5, 0), False, True, RollConvention.MODIFIED_FOLLOWING, holidays_de
            ).generate_dates(False),
            [datetime(2020, 8, 21), datetime(2021, 1, 21), datetime(2021, 8, 23)],
        )
        self.assertEqual(
            Schedule(
                datetime(2020, 8, 21), datetime(2021, 8, 21), Period(0, 5, 0), False, False, RollConvention.MODIFIED_FOLLOWING, holidays_de
            ).generate_dates(False),
            [datetime(2020, 8, 21), datetime(2021, 1, 21), datetime(2021, 6, 21), datetime(2021, 8, 23)],
        )

    def test_calc_end_day(self):
        self.assertEqual(calc_end_day(datetime(2023, 1, 31), Period(0, 1, 0)), datetime(2023, 2, 28))
        self.assertEqual(calc_end_day(datetime(2023, 1, 30), Period(0, 1, 0)), datetime(2023, 2, 28))
        self.assertEqual(calc_end_day(datetime(2023, 1, 29), Period(0, 1, 0)), datetime(2023, 2, 28))
        self.assertEqual(calc_end_day(datetime(2023, 1, 28), Period(0, 1, 0)), datetime(2023, 2, 28))
        self.assertEqual(calc_end_day(datetime(2024, 1, 31), Period(0, 1, 0)), datetime(2024, 2, 29))
        self.assertEqual(calc_end_day(datetime(2024, 1, 30), Period(0, 1, 0)), datetime(2024, 2, 29))
        self.assertEqual(calc_end_day(datetime(2024, 1, 29), Period(0, 1, 0)), datetime(2024, 2, 29))
        self.assertEqual(calc_end_day(datetime(2024, 1, 28), Period(0, 1, 0)), datetime(2024, 2, 28))
        self.assertEqual(calc_end_day(datetime(2023, 2, 28), Period(0, 1, 0)), datetime(2023, 3, 28))
        self.assertEqual(calc_end_day(datetime(2023, 2, 27), Period(0, 1, 0)), datetime(2023, 3, 27))
        self.assertEqual(calc_end_day(datetime(2024, 2, 29), Period(0, 1, 0)), datetime(2024, 3, 29))
        self.assertEqual(calc_end_day(datetime(2024, 2, 28), Period(0, 1, 0)), datetime(2024, 3, 28))
        self.assertEqual(calc_end_day(datetime(2024, 2, 27), Period(0, 1, 0)), datetime(2024, 3, 27))
        roll = RollRule(RollRule.EOM)
        self.assertEqual(calc_end_day(datetime(2023, 1, 31), Period(0, 1, 0), roll_convention=roll), datetime(2023, 2, 28))
        self.assertEqual(calc_end_day(datetime(2023, 1, 30), Period(0, 1, 0), roll_convention=roll), datetime(2023, 2, 28))
        self.assertEqual(calc_end_day(datetime(2023, 1, 29), Period(0, 1, 0), roll_convention=roll), datetime(2023, 2, 28))
        self.assertEqual(calc_end_day(datetime(2023, 1, 28), Period(0, 1, 0), roll_convention=roll), datetime(2023, 2, 28))
        self.assertEqual(calc_end_day(datetime(2024, 1, 31), Period(0, 1, 0), roll_convention=roll), datetime(2024, 2, 29))
        self.assertEqual(calc_end_day(datetime(2024, 1, 30), Period(0, 1, 0), roll_convention=roll), datetime(2024, 2, 29))
        self.assertEqual(calc_end_day(datetime(2024, 1, 29), Period(0, 1, 0), roll_convention=roll), datetime(2024, 2, 29))
        self.assertEqual(calc_end_day(datetime(2024, 1, 28), Period(0, 1, 0), roll_convention=roll), datetime(2024, 2, 28))
        self.assertEqual(calc_end_day(datetime(2023, 2, 28), Period(0, 1, 0), roll_convention=roll), datetime(2023, 3, 31))
        self.assertEqual(calc_end_day(datetime(2023, 2, 27), Period(0, 1, 0), roll_convention=roll), datetime(2023, 3, 27))
        self.assertEqual(calc_end_day(datetime(2024, 2, 29), Period(0, 1, 0), roll_convention=roll), datetime(2024, 3, 31))
        self.assertEqual(calc_end_day(datetime(2024, 2, 28), Period(0, 1, 0), roll_convention=roll), datetime(2024, 3, 31))
        self.assertEqual(calc_end_day(datetime(2024, 2, 27), Period(0, 1, 0), roll_convention=roll), datetime(2024, 3, 27))
        self.assertEqual(calc_end_day(datetime(2023, 3, 31), Period(0, 1, 0), roll_convention=roll), datetime(2023, 4, 30))
        self.assertEqual(calc_end_day(datetime(2023, 3, 30), Period(0, 1, 0), roll_convention=roll), datetime(2023, 4, 30))
        self.assertEqual(calc_end_day(datetime(2023, 3, 29), Period(0, 1, 0), roll_convention=roll), datetime(2023, 4, 29))
        self.assertEqual(calc_end_day(datetime(2023, 3, 28), Period(0, 1, 0), roll_convention=roll), datetime(2023, 4, 28))
        roll = RollRule(RollRule.DOM)
        self.assertEqual(calc_end_day(datetime(2023, 1, 31), Period(0, 1, 0), roll_convention=roll), datetime(2023, 2, 28))
        self.assertEqual(calc_end_day(datetime(2023, 1, 30), Period(0, 1, 0), roll_convention=roll), datetime(2023, 2, 28))
        self.assertEqual(calc_end_day(datetime(2023, 1, 29), Period(0, 1, 0), roll_convention=roll), datetime(2023, 2, 28))
        self.assertEqual(calc_end_day(datetime(2023, 1, 28), Period(0, 1, 0), roll_convention=roll), datetime(2023, 2, 28))
        self.assertEqual(calc_end_day(datetime(2024, 1, 31), Period(0, 1, 0), roll_convention=roll), datetime(2024, 2, 29))
        self.assertEqual(calc_end_day(datetime(2024, 1, 30), Period(0, 1, 0), roll_convention=roll), datetime(2024, 2, 29))
        self.assertEqual(calc_end_day(datetime(2024, 1, 29), Period(0, 1, 0), roll_convention=roll), datetime(2024, 2, 29))
        self.assertEqual(calc_end_day(datetime(2024, 1, 28), Period(0, 1, 0), roll_convention=roll), datetime(2024, 2, 28))
        self.assertEqual(calc_end_day(datetime(2023, 2, 28), Period(0, 1, 0), roll_convention=roll), datetime(2023, 3, 28))
        self.assertEqual(calc_end_day(datetime(2023, 2, 27), Period(0, 1, 0), roll_convention=roll), datetime(2023, 3, 27))
        self.assertEqual(calc_end_day(datetime(2024, 2, 29), Period(0, 1, 0), roll_convention=roll), datetime(2024, 3, 29))
        self.assertEqual(calc_end_day(datetime(2024, 2, 28), Period(0, 1, 0), roll_convention=roll), datetime(2024, 3, 28))
        self.assertEqual(calc_end_day(datetime(2024, 2, 27), Period(0, 1, 0), roll_convention=roll), datetime(2024, 3, 27))
        self.assertEqual(calc_end_day(datetime(2023, 3, 31), Period(0, 1, 0), roll_convention=roll), datetime(2023, 4, 30))
        self.assertEqual(calc_end_day(datetime(2023, 3, 30), Period(0, 1, 0), roll_convention=roll), datetime(2023, 4, 30))
        self.assertEqual(calc_end_day(datetime(2023, 3, 29), Period(0, 1, 0), roll_convention=roll), datetime(2023, 4, 29))
        self.assertEqual(calc_end_day(datetime(2023, 3, 28), Period(0, 1, 0), roll_convention=roll), datetime(2023, 4, 28))
        roll = RollRule(RollRule.IMM)
        self.assertEqual(calc_end_day(datetime(2023, 1, 31), Period(0, 1, 0), roll_convention=roll), datetime(2023, 6, 21))
        self.assertEqual(calc_end_day(datetime(2023, 2, 28), Period(0, 1, 0), roll_convention=roll), datetime(2023, 6, 21))
        self.assertEqual(calc_end_day(datetime(2023, 3, 29), Period(0, 1, 0), roll_convention=roll), datetime(2023, 9, 20))
        self.assertEqual(calc_end_day(datetime(2023, 1, 28), Period(0, 2, 0), roll_convention=roll), datetime(2023, 6, 21))
        self.assertEqual(calc_end_day(datetime(2023, 1, 28), Period(0, 3, 0), roll_convention=roll), datetime(2023, 6, 21))
        self.assertEqual(calc_end_day(datetime(2023, 3, 15), Period(0, 3, 0), roll_convention=roll), datetime(2023, 6, 21))
        self.assertEqual(calc_end_day(datetime(2023, 3, 15), Period(0, 6, 0), roll_convention=roll), datetime(2023, 9, 20))
        self.assertEqual(calc_end_day(datetime(2023, 1, 28), Period(0, 4, 0), roll_convention=roll), datetime(2023, 9, 20))
        self.assertEqual(calc_end_day(datetime(2023, 3, 28), Period(0, 4, 0), roll_convention=roll), datetime(2023, 12, 20))
        self.assertEqual(calc_end_day(datetime(2023, 3, 28), Period(0, 6, 0), roll_convention=roll), datetime(2023, 12, 20))

    def test_is_ambiguous_date(self):
        self.assertEqual(_is_ambiguous_date(datetime(2023, 1, 31)), False)
        self.assertEqual(_is_ambiguous_date(datetime(2023, 2, 28)), True)
        self.assertEqual(_is_ambiguous_date(datetime(2024, 1, 31)), False)
        self.assertEqual(_is_ambiguous_date(datetime(2024, 2, 27)), False)
        self.assertEqual(_is_ambiguous_date(datetime(2024, 1, 30)), True)
        self.assertEqual(_is_ambiguous_date(datetime(2024, 2, 29)), True)
        self.assertEqual(_is_ambiguous_date(datetime(2024, 3, 30)), True)
        self.assertEqual(_is_ambiguous_date(datetime(2024, 3, 29)), False)
        self.assertEqual(_is_ambiguous_date(datetime(2024, 3, 31)), False)

    def test_is_IMM_date(self):
        self.assertEqual(_is_IMM_date(datetime(2023, 1, 18)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 1, 19)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 1, 20)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 1, 21)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 2, 15)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 2, 16)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 2, 17)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 2, 18)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 3, 15)), True)
        self.assertEqual(_is_IMM_date(datetime(2023, 3, 16)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 3, 17)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 3, 22)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 4, 19)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 4, 20)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 4, 21)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 5, 17)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 5, 18)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 5, 19)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 6, 21)), True)
        self.assertEqual(_is_IMM_date(datetime(2023, 6, 20)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 6, 23)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 7, 19)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 7, 20)), False)
        self.assertEqual(_is_IMM_date(datetime(2023, 7, 21)), False)

    def test_calc_start_day(self):
        self.assertEqual(calc_start_day(datetime(2023, 1, 31), Period(0, 1, 0)), datetime(2022, 12, 31))
        self.assertEqual(calc_start_day(datetime(2023, 1, 30), Period(0, 1, 0)), datetime(2022, 12, 30))
        self.assertEqual(calc_start_day(datetime(2023, 1, 29), Period(0, 1, 0)), datetime(2022, 12, 29))
        self.assertEqual(calc_start_day(datetime(2023, 1, 28), Period(0, 1, 0)), datetime(2022, 12, 28))
        self.assertEqual(calc_start_day(datetime(2023, 1, 27), Period(0, 1, 0)), datetime(2022, 12, 27))
        self.assertEqual(calc_start_day(datetime(2023, 1, 26), Period(0, 1, 0)), datetime(2022, 12, 26))
        self.assertEqual(calc_start_day(datetime(2023, 1, 1), Period(0, 0, 1)), datetime(2022, 12, 31))
        self.assertEqual(calc_start_day(datetime(2023, 1, 2), Period(0, 0, 1)), datetime(2023, 1, 1))
        bdc = RollConvention.MODIFIED_FOLLOWING
        self.assertEqual(calc_start_day(datetime(2023, 1, 31), Period(0, 1, 0), bdc), datetime(2022, 12, 31))
        self.assertEqual(calc_start_day(datetime(2023, 1, 30), Period(0, 1, 0), bdc), datetime(2022, 12, 30))
        self.assertEqual(calc_start_day(datetime(2023, 1, 29), Period(0, 1, 0), bdc), None)
        self.assertEqual(calc_start_day(datetime(2023, 1, 28), Period(0, 1, 0), bdc), None)
        self.assertEqual(calc_start_day(datetime(2023, 1, 27), Period(0, 1, 0), bdc), datetime(2022, 12, 27))
        self.assertEqual(calc_start_day(datetime(2023, 1, 26), Period(0, 1, 0), bdc), datetime(2022, 12, 26))
        self.assertEqual(calc_start_day(datetime(2023, 1, 1), Period(0, 0, 1), bdc), None)
        self.assertEqual(calc_start_day(datetime(2023, 1, 2), Period(0, 0, 1), bdc), datetime(2023, 1, 1))
        bdc = RollConvention.FOLLOWING
        self.assertEqual(calc_start_day(datetime(2023, 1, 31), Period(0, 1, 0), bdc), datetime(2022, 12, 31))
        self.assertEqual(calc_start_day(datetime(2023, 1, 30), Period(0, 1, 0), bdc), datetime(2022, 12, 30))
        self.assertEqual(calc_start_day(datetime(2023, 1, 29), Period(0, 1, 0), bdc), None)
        self.assertEqual(calc_start_day(datetime(2023, 1, 28), Period(0, 1, 0), bdc), None)
        self.assertEqual(calc_start_day(datetime(2023, 1, 27), Period(0, 1, 0), bdc), datetime(2022, 12, 27))
        self.assertEqual(calc_start_day(datetime(2023, 1, 26), Period(0, 1, 0), bdc), datetime(2022, 12, 26))
        self.assertEqual(calc_start_day(datetime(2023, 1, 1), Period(0, 0, 1), bdc), None)
        self.assertEqual(calc_start_day(datetime(2023, 1, 2), Period(0, 0, 1), bdc), datetime(2022, 12, 30))
        bdc = RollConvention.PRECEDING
        self.assertEqual(calc_start_day(datetime(2023, 1, 31), Period(0, 1, 0), bdc), datetime(2022, 12, 31))
        self.assertEqual(calc_start_day(datetime(2023, 1, 30), Period(0, 1, 0), bdc), datetime(2022, 12, 30))
        self.assertEqual(calc_start_day(datetime(2023, 1, 29), Period(0, 1, 0), bdc), None)
        self.assertEqual(calc_start_day(datetime(2023, 1, 28), Period(0, 1, 0), bdc), None)
        self.assertEqual(calc_start_day(datetime(2023, 1, 27), Period(0, 1, 0), bdc), datetime(2022, 12, 27))
        self.assertEqual(calc_start_day(datetime(2023, 1, 26), Period(0, 1, 0), bdc), datetime(2022, 12, 26))
        self.assertEqual(calc_start_day(datetime(2023, 1, 1), Period(0, 0, 1), bdc), None)
        self.assertEqual(calc_start_day(datetime(2023, 1, 2), Period(0, 0, 1), bdc), datetime(2023, 1, 1))

    def test_last_day_of_month(self):
        self.assertEqual(last_day_of_month(datetime(2023, 1, 15)), date(2023, 1, 31))
        self.assertEqual(last_day_of_month(datetime(2023, 2, 15)), date(2023, 2, 28))
        self.assertEqual(last_day_of_month(datetime(2023, 3, 15)), date(2023, 3, 31))
        self.assertEqual(last_day_of_month(datetime(2023, 4, 15)), date(2023, 4, 30))
        self.assertEqual(last_day_of_month(datetime(2023, 5, 15)), date(2023, 5, 31))
        self.assertEqual(last_day_of_month(datetime(2023, 6, 15)), date(2023, 6, 30))
        self.assertEqual(last_day_of_month(datetime(2023, 7, 15)), date(2023, 7, 31))
        self.assertEqual(last_day_of_month(datetime(2023, 8, 15)), date(2023, 8, 31))
        self.assertEqual(last_day_of_month(datetime(2023, 9, 15)), date(2023, 9, 30))
        self.assertEqual(last_day_of_month(datetime(2023, 10, 15)), date(2023, 10, 31))
        self.assertEqual(last_day_of_month(datetime(2023, 11, 15)), date(2023, 11, 30))
        self.assertEqual(last_day_of_month(datetime(2023, 12, 15)), date(2023, 12, 31))
        self.assertEqual(last_day_of_month(datetime(2024, 1, 15)), date(2024, 1, 31))
        self.assertEqual(last_day_of_month(datetime(2024, 2, 15)), date(2024, 2, 29))

    def test_is_last_day_of_month(self):
        self.assertEqual(is_last_day_of_month(datetime(2023, 1, 31)), True)
        self.assertEqual(is_last_day_of_month(datetime(2023, 2, 28)), True)
        self.assertEqual(is_last_day_of_month(datetime(2024, 2, 28)), False)
        self.assertEqual(is_last_day_of_month(datetime(2024, 2, 29)), True)
        self.assertEqual(is_last_day_of_month(datetime(2023, 3, 31)), True)
        self.assertEqual(is_last_day_of_month(datetime(2023, 4, 30)), True)
        self.assertEqual(is_last_day_of_month(datetime(2023, 4, 29)), False)

    def test_is_business_day(self):
        holidays_de = ECB()
        self.assertEqual(is_business_day(datetime(2023, 1, 2), holidays_de), True)  # New Year's Day observed
        self.assertEqual(is_business_day(datetime(2023, 1, 3), holidays_de), True)
        self.assertEqual(is_business_day(datetime(2023, 1, 7), holidays_de), False)
        self.assertEqual(is_business_day(datetime(2023, 1, 8), holidays_de), False)  # Sunday
        self.assertEqual(is_business_day(datetime(2023, 1, 9), holidays_de), True)  # Monday after New Year's Day
        self.assertEqual(is_business_day(datetime(2023, 4, 7), holidays_de), False)  # Good Friday
        self.assertEqual(is_business_day(datetime(2023, 4, 10), holidays_de), False)  # Easter Monday
        self.assertEqual(is_business_day(datetime(2023, 5, 1), holidays_de), False)  # Labour Day
        self.assertEqual(is_business_day(datetime(2023, 5, 18), holidays_de), True)  # Ascension Day
        self.assertEqual(is_business_day(datetime(2023, 5, 29), holidays_de), True)  # Whit Monday
        self.assertEqual(is_business_day(datetime(2023, 6, 8), holidays_de), True)  # Corpus Christi
        self.assertEqual(is_business_day(datetime(2023, 10, 3), holidays_de), True)  # German Unity Day
        self.assertEqual(is_business_day(datetime(2023, 12, 25), holidays_de), False)  # Christmas Day
        self.assertEqual(is_business_day(datetime(2023, 12, 26), holidays_de), False)  # Second Day of Christmas
        self.assertEqual(is_business_day(datetime(2023, 12, 27), holidays_de), True)
        self.assertEqual(is_business_day(datetime(2025, 4, 18), holidays_de), False)  # Good Friday
        holidays_de = DE()
        self.assertEqual(is_business_day(datetime(2023, 1, 2), holidays_de), True)  # New Year's Day observed
        self.assertEqual(is_business_day(datetime(2023, 1, 3), holidays_de), True)
        self.assertEqual(is_business_day(datetime(2023, 1, 7), holidays_de), False)
        self.assertEqual(is_business_day(datetime(2023, 1, 8), holidays_de), False)  # Sunday
        self.assertEqual(is_business_day(datetime(2023, 1, 9), holidays_de), True)  # Monday after New Year's Day
        self.assertEqual(is_business_day(datetime(2023, 4, 7), holidays_de), False)  # Good Friday
        self.assertEqual(is_business_day(datetime(2023, 4, 10), holidays_de), False)  # Easter Monday
        self.assertEqual(is_business_day(datetime(2023, 5, 1), holidays_de), False)  # Labour Day
        self.assertEqual(is_business_day(datetime(2023, 5, 18), holidays_de), False)  # Ascension Day
        self.assertEqual(is_business_day(datetime(2023, 5, 29), holidays_de), False)  # Whit Monday
        self.assertEqual(is_business_day(datetime(2023, 6, 8), holidays_de), True)  # Corpus Christi
        self.assertEqual(is_business_day(datetime(2023, 10, 3), holidays_de), False)  # German Unity Day
        self.assertEqual(is_business_day(datetime(2023, 12, 25), holidays_de), False)  # Christmas Day
        self.assertEqual(is_business_day(datetime(2023, 12, 26), holidays_de), False)  # Second Day of Christmas
        self.assertEqual(is_business_day(datetime(2023, 12, 27), holidays_de), True)
        self.assertEqual(is_business_day(datetime(2025, 4, 18), holidays_de), False)  # Good Friday

    def test_last_business_day_of_month(self):
        self.assertEqual(last_business_day_of_month(datetime(2023, 1, 15), ECB()), date(2023, 1, 31))
        self.assertEqual(last_business_day_of_month(datetime(2023, 2, 15), ECB()), date(2023, 2, 28))
        self.assertEqual(last_business_day_of_month(datetime(2023, 3, 15), ECB()), date(2023, 3, 31))
        self.assertEqual(last_business_day_of_month(datetime(2023, 4, 15), ECB()), date(2023, 4, 28))
        self.assertEqual(last_business_day_of_month(datetime(2023, 5, 15), ECB()), date(2023, 5, 31))
        self.assertEqual(last_business_day_of_month(datetime(2023, 6, 15), ECB()), date(2023, 6, 30))
        self.assertEqual(last_business_day_of_month(datetime(2023, 7, 15), ECB()), date(2023, 7, 31))
        self.assertEqual(last_business_day_of_month(datetime(2023, 8, 15), ECB()), date(2023, 8, 31))
        self.assertEqual(last_business_day_of_month(datetime(2023, 9, 15), ECB()), date(2023, 9, 29))
        self.assertEqual(last_business_day_of_month(datetime(2023, 10, 15), ECB()), date(2023, 10, 31))
        self.assertEqual(last_business_day_of_month(datetime(2023, 11, 15), ECB()), date(2023, 11, 30))
        self.assertEqual(last_business_day_of_month(datetime(2023, 12, 15), ECB()), date(2023, 12, 29))
        self.assertEqual(last_business_day_of_month(datetime(2024, 1, 15), ECB()), date(2024, 1, 31))
        self.assertEqual(last_business_day_of_month(datetime(2024, 2, 15), ECB()), date(2024, 2, 29))

    def test_is_last_business_day_of_month(self):
        self.assertEqual(is_last_business_day_of_month(datetime(2023, 1, 31), ECB()), True)
        self.assertEqual(is_last_business_day_of_month(datetime(2023, 2, 27), ECB()), False)
        self.assertEqual(is_last_business_day_of_month(datetime(2023, 3, 31), ECB()), True)
        self.assertEqual(is_last_business_day_of_month(datetime(2023, 4, 28), ECB()), True)
        self.assertEqual(is_last_business_day_of_month(datetime(2023, 5, 31), ECB()), True)
        self.assertEqual(is_last_business_day_of_month(datetime(2023, 6, 30), ECB()), True)
        self.assertEqual(is_last_business_day_of_month(datetime(2023, 7, 31), ECB()), True)
        self.assertEqual(is_last_business_day_of_month(datetime(2023, 8, 31), ECB()), True)
        self.assertEqual(is_last_business_day_of_month(datetime(2023, 9, 29), ECB()), True)
        self.assertEqual(is_last_business_day_of_month(datetime(2023, 10, 30), ECB()), False)
        self.assertEqual(is_last_business_day_of_month(datetime(2023, 11, 30), ECB()), True)
        self.assertEqual(is_last_business_day_of_month(datetime(2023, 12, 30), ECB()), False)
        self.assertEqual(is_last_business_day_of_month(datetime(2024, 1, 31), ECB()), True)
        self.assertEqual(is_last_business_day_of_month(datetime(2024, 2, 28), ECB()), False)

    def test_nearest_business_day(self):
        self.assertEqual(nearest_business_day(datetime(2023, 1, 1), ECB()), datetime(2023, 1, 2))
        self.assertEqual(nearest_business_day(datetime(2023, 1, 2), ECB()), datetime(2023, 1, 2))
        self.assertEqual(nearest_business_day(datetime(2023, 1, 3), ECB()), datetime(2023, 1, 3))
        self.assertEqual(nearest_business_day(datetime(2023, 1, 7), ECB()), datetime(2023, 1, 6))
        self.assertEqual(nearest_business_day(datetime(2023, 1, 8), ECB()), datetime(2023, 1, 9))
        self.assertEqual(nearest_business_day(datetime(2023, 1, 9), ECB()), datetime(2023, 1, 9))
        self.assertEqual(nearest_business_day(datetime(2023, 4, 7), ECB()), datetime(2023, 4, 6))  # Good Friday
        self.assertEqual(nearest_business_day(datetime(2023, 4, 8), ECB()), datetime(2023, 4, 6))  # Saturday
        self.assertEqual(nearest_business_day(datetime(2023, 4, 9), ECB()), datetime(2023, 4, 11))  # Easter Sunday
        self.assertEqual(nearest_business_day(datetime(2023, 4, 10), ECB()), datetime(2023, 4, 11))  # Easter Monday
        self.assertEqual(nearest_business_day(datetime(2023, 12, 23), ECB()), datetime(2023, 12, 22))
        self.assertEqual(nearest_business_day(datetime(2023, 12, 24), ECB()), datetime(2023, 12, 22))  # Sunday
        self.assertEqual(nearest_business_day(datetime(2023, 12, 25), ECB()), datetime(2023, 12, 27))  # Christmas Day
        self.assertEqual(nearest_business_day(datetime(2023, 12, 26), ECB()), datetime(2023, 12, 27))  # Second Day of Christmas
        self.assertEqual(nearest_business_day(datetime(2023, 12, 27), ECB()), datetime(2023, 12, 27))
        self.assertEqual(nearest_business_day(datetime(2023, 12, 30), ECB()), datetime(2023, 12, 29))
        self.assertEqual(nearest_business_day(datetime(2023, 12, 31), ECB()), datetime(2024, 1, 2))  # New Year's Day observed

    def test_nearest_last_business_day_of_month(self):
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 1, 15), ECB()), datetime(2023, 1, 31))
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 1, 30), ECB()), datetime(2023, 1, 31))
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 1, 31), ECB()), datetime(2023, 1, 31))
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 2, 1), ECB()), datetime(2023, 1, 31))
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 2, 14), ECB()), datetime(2023, 2, 28))
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 2, 15), ECB()), datetime(2023, 2, 28))
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 2, 27), ECB()), datetime(2023, 2, 28))
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 2, 28), ECB()), datetime(2023, 2, 28))
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 3, 1), ECB()), datetime(2023, 2, 28))
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 3, 15), ECB()), datetime(2023, 2, 28))
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 3, 15), ECB(), False), datetime(2023, 2, 28))
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 3, 16), ECB()), datetime(2023, 3, 31))
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 3, 30), ECB()), datetime(2023, 3, 31))
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 3, 31), ECB()), datetime(2023, 3, 31))
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 4, 1), ECB()), datetime(2023, 3, 31))
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 4, 13), ECB()), datetime(2023, 3, 31))
        self.assertEqual(nearest_last_business_day_of_month(datetime(2023, 4, 14), ECB()), datetime(2023, 4, 28))

    def test_next_or_previous_business_day(self):
        self.assertEqual(next_or_previous_business_day(datetime(2023, 1, 1), ECB(), True), datetime(2023, 1, 2))
        self.assertEqual(next_or_previous_business_day(datetime(2023, 1, 2), ECB(), True), datetime(2023, 1, 2))
        self.assertEqual(next_or_previous_business_day(datetime(2023, 1, 3), ECB(), True), datetime(2023, 1, 3))
        self.assertEqual(next_or_previous_business_day(datetime(2023, 1, 7), ECB(), True), datetime(2023, 1, 9))
        self.assertEqual(next_or_previous_business_day(datetime(2023, 1, 8), ECB(), True), datetime(2023, 1, 9))
        self.assertEqual(next_or_previous_business_day(datetime(2023, 1, 7), ECB(), False), datetime(2023, 1, 6))
        self.assertEqual(next_or_previous_business_day(datetime(2023, 1, 8), ECB(), False), datetime(2023, 1, 6))
        self.assertEqual(next_or_previous_business_day(datetime(2023, 1, 9), ECB(), True), datetime(2023, 1, 9))
        self.assertEqual(next_or_previous_business_day(datetime(2023, 4, 7), ECB(), True), datetime(2023, 4, 11))  # Good Friday
        self.assertEqual(next_or_previous_business_day(datetime(2023, 4, 8), ECB(), True), datetime(2023, 4, 11))  # Saturday
        self.assertEqual(next_or_previous_business_day(datetime(2023, 4, 9), ECB(), True), datetime(2023, 4, 11))  # Easter Sunday
        self.assertEqual(next_or_previous_business_day(datetime(2023, 4, 10), ECB(), True), datetime(2023, 4, 11))  # Easter Monday
        self.assertEqual(next_or_previous_business_day(datetime(2023, 4, 7), ECB(), False), datetime(2023, 4, 6))  # Good Friday
        self.assertEqual(next_or_previous_business_day(datetime(2023, 4, 8), ECB(), False), datetime(2023, 4, 6))  # Saturday
        self.assertEqual(next_or_previous_business_day(datetime(2023, 4, 9), ECB(), False), datetime(2023, 4, 6))  # Easter Sunday
        self.assertEqual(next_or_previous_business_day(datetime(2023, 4, 10), ECB(), False), datetime(2023, 4, 6))  # Easter Monday
        self.assertEqual(next_or_previous_business_day(datetime(2023, 12, 23), ECB(), True), datetime(2023, 12, 27))
        self.assertEqual(next_or_previous_business_day(datetime(2023, 12, 24), ECB(), True), datetime(2023, 12, 27))  # Sunday
        self.assertEqual(next_or_previous_business_day(datetime(2023, 12, 25), ECB(), True), datetime(2023, 12, 27))  # Christmas Day
        self.assertEqual(next_or_previous_business_day(datetime(2023, 12, 26), ECB(), False), datetime(2023, 12, 22))  # Second Day of Christmas

    def test_next_IMM_date(self):
        self.assertEqual(next_IMM_date(datetime(2023, 1, 15)), date(2023, 3, 15))
        self.assertEqual(next_IMM_date(datetime(2023, 1, 16)), date(2023, 3, 15))
        self.assertEqual(next_IMM_date(datetime(2023, 1, 17)), date(2023, 3, 15))
        self.assertEqual(next_IMM_date(datetime(2023, 1, 18)), date(2023, 3, 15))
        self.assertEqual(next_IMM_date(datetime(2023, 1, 19)), date(2023, 3, 15))
        self.assertEqual(next_IMM_date(datetime(2023, 1, 20)), date(2023, 3, 15))
        self.assertEqual(next_IMM_date(datetime(2023, 1, 21)), date(2023, 3, 15))
        self.assertEqual(next_IMM_date(datetime(2023, 2, 15)), date(2023, 3, 15))
        self.assertEqual(next_IMM_date(datetime(2023, 2, 16)), date(2023, 3, 15))
        self.assertEqual(next_IMM_date(datetime(2023, 2, 17)), date(2023, 3, 15))
        self.assertEqual(next_IMM_date(datetime(2023, 2, 18)), date(2023, 3, 15))
        self.assertEqual(next_IMM_date(datetime(2023, 2, 19)), date(2023, 3, 15))
        self.assertEqual(next_IMM_date(datetime(2023, 2, 20)), date(2023, 3, 15))
        self.assertEqual(next_IMM_date(datetime(2023, 2, 21)), date(2023, 3, 15))
        self.assertEqual(next_IMM_date(datetime(2023, 3, 14)), date(2023, 3, 15))
        self.assertEqual(next_IMM_date(datetime(2023, 3, 15)), date(2023, 6, 21))
        self.assertEqual(next_IMM_date(datetime(2023, 3, 16)), date(2023, 6, 21))
        self.assertEqual(next_IMM_date(datetime(2023, 12, 19)), date(2023, 12, 20))
        self.assertEqual(next_IMM_date(datetime(2023, 12, 20)), date(2024, 3, 20))


if __name__ == "__main__":
    main()
