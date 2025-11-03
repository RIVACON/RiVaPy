import unittest
import datetime as dt
import math

from rivapy.tools.interpolate import Interpolator
from rivapy.tools.enums import DayCounterType, InterpolationType, ExtrapolationType
from rivapy.marketdata import DiscountCurve, SurvivalCurve, EquityForwardCurve
from rivapy.tools.datetools import DayCounter


# , delta=1e-5 ?
class TestDiscountCurve(unittest.TestCase):

    # Discount Curve has
    # _init_
    # get_dates
    # get_df
    # value
    # get_pyvacon_obj
    # value -  cares about daycount convention refdates, target date, interpolation type, extrapolation type..
    #  at the moment, it is assumed the target date is correctly calculated  before input with correct business day logic/roll convention
    #  there is at the moment a potential issue with the roll convention and the given reference date, for now assume it is correct #TODO
    # plot - calls the value or value in order to plot.
    # TODO: consider how to wrap both value and value depending on if the interpolationType is a pyvacon or Rivapy construction respectively

    def setUp(self):
        """Test data, simple linear case. Extend to more robust if requested."""

        # Example discount curve, with discount factors calculated based on given rates and days to maturity
        # first example uses day count convention ACT365FIXED

        # base imformation
        self.refdate = dt.datetime(2017, 1, 1, 0, 0, 0)
        self.days_to_maturity = [1, 180, 365, 720, 3 * 365, 4 * 365, 10 * 365]
        self.rates = [-0.0065, 0.0003, 0.0059, 0.0086, 0.0101, 0.02, 0.03]
        self.rate = 0.03
        self.dates = [self.refdate + dt.timedelta(days=i) for i in self.days_to_maturity]

        self.dsc_fac_ACT365FIXED = [math.exp(-d / 365.0 * self.rate) for d in self.days_to_maturity]

    def test_init_success_and_getters(self):
        """Check initialization and getters."""
        dc = DiscountCurve(
            "test_curve",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
            DayCounterType.Act365Fixed,
        )
        self.assertEqual(dc.id, "test_curve")
        self.assertEqual(dc.get_dates()[0], self.refdate)
        self.assertEqual(dc.get_df()[0], 1.0)
        self.assertTrue(all(isinstance(d, dt.datetime) for d in dc.get_dates()))

    def test_init_invalid_inputs(self):
        """Invalid input combinations must raise where appropriate."""

        # Empty dates and dfs — must raise
        with self.assertRaises(Exception):
            DiscountCurve("x", self.refdate, [], [])

        # Length mismatch
        with self.assertRaises(Exception):
            DiscountCurve("x", self.refdate, [self.refdate + dt.timedelta(days=1)], [0.9, 0.8])

        # Non-enum arguments
        with self.assertRaises(TypeError):
            DiscountCurve(
                "x",
                self.refdate,
                [self.refdate + dt.timedelta(days=1)],
                [1.0],
                interpolation="BAD",
                extrapolation=ExtrapolationType.LINEAR,
                daycounter=DayCounterType.Act365Fixed,
            )
        with self.assertRaises(TypeError):
            DiscountCurve(
                "x",
                self.refdate,
                [self.refdate + dt.timedelta(days=1)],
                [1.0],
                interpolation=InterpolationType.LINEAR,
                extrapolation="BAD",
                daycounter=DayCounterType.Act365Fixed,
            )
        with self.assertRaises(TypeError):
            DiscountCurve(
                "x",
                self.refdate,
                [self.refdate + dt.timedelta(days=1)],
                [1.0],
                interpolation=InterpolationType.LINEAR,
                extrapolation=ExtrapolationType.LINEAR,
                daycounter="BAD",
            )

        # First date before refdate
        with self.assertRaises(Exception):
            DiscountCurve(
                "x",
                self.refdate,
                [self.refdate - dt.timedelta(days=1)],
                [1.0],
                InterpolationType.LINEAR,
                ExtrapolationType.LINEAR,
                DayCounterType.Act365Fixed,
            )

        # Instead of expecting an exception, assert correct behavior:
        dc = DiscountCurve(
            "x",
            self.refdate,
            [self.refdate + dt.timedelta(days=1)],
            [0.9],
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
            DayCounterType.Act365Fixed,
        )
        # constructor should have prepended (refdate, 1.0)
        dates = dc.get_dates()
        dfs = dc.get_df()
        self.assertEqual(dates[0], self.refdate)
        self.assertEqual(dfs[0], 1.0)
        self.assertEqual(dates[1], self.refdate + dt.timedelta(days=1))
        self.assertAlmostEqual(dfs[1], 0.9)

        # Non-monotonic or duplicate dates
        with self.assertRaises(Exception):
            DiscountCurve(
                "x",
                self.refdate,
                [self.refdate + dt.timedelta(days=1), self.refdate + dt.timedelta(days=1)],
                [1.0, 0.99],
                InterpolationType.LINEAR,
                ExtrapolationType.LINEAR,
                DayCounterType.Act365Fixed,
            )

    def test_get_dates_and_get_df(self):
        dc = DiscountCurve(
            "Test_DC_ACT365FIXED",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
            DayCounterType.Act365Fixed,
        )
        dates = dc.get_dates()
        dfs = dc.get_df()
        self.assertEqual(len(dates), len(dfs))
        self.assertEqual(dates[0], self.refdate)
        self.assertEqual(dfs[0], 1.0)

    def test_value_and_extrapolation_cases(self):
        """Replicates and extends original test_value logic."""
        dc_linear = DiscountCurve(
            "Test_DC_ACT365FIXED",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
            DayCounterType.Act365Fixed,
        )

        df1 = dc_linear.value(self.refdate, self.refdate + dt.timedelta(days=90))
        df2 = dc_linear.value(self.refdate, self.refdate + dt.timedelta(days=180))
        fwd_df = dc_linear.value(self.refdate + dt.timedelta(days=90), self.refdate + dt.timedelta(days=180))
        self.assertAlmostEqual(df1, 0.9926568878362608, delta=1e-5)
        self.assertAlmostEqual(df2, 0.9853143806626516, delta=1e-5)
        self.assertAlmostEqual(fwd_df, df2 / df1, delta=1e-5)

        # Linear extrapolation
        df_extrap1 = dc_linear.value(self.refdate, self.refdate + dt.timedelta(days=10 * 365 + 10))
        df_extrap2 = dc_linear.value(self.refdate, self.refdate + dt.timedelta(days=10 * 365 + 60))
        self.assertAlmostEqual(df_extrap1, 0.7401510872751634, delta=1e-5)
        self.assertAlmostEqual(df_extrap2, 0.7368154202423907, delta=1e-5)

        # Constant extrapolation
        dc_const = DiscountCurve(
            "Test_DC_ACT365FIXED",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.LINEAR,
            ExtrapolationType.CONSTANT,
            DayCounterType.Act365Fixed,
        )
        df_const1 = dc_const.value(self.refdate, self.refdate + dt.timedelta(days=10 * 365 + 10))
        df_const2 = dc_const.value(self.refdate, self.refdate + dt.timedelta(days=10 * 365 + 60))
        self.assertAlmostEqual(df_const1, df_const2, delta=1e-10)

        # Extrapolation NONE -> should raise ValueError
        dc_none = DiscountCurve(
            "Test_DC_ACT365FIXED",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.LINEAR,
            ExtrapolationType.NONE,
            DayCounterType.Act365Fixed,
        )
        with self.assertRaises(ValueError):
            dc_none.value(self.refdate, self.refdate + dt.timedelta(days=4000))

        # TODO:
        # Tests with other DCC

        # ActAct, LINEAR, LINEAR
        # ActAct, LINEAR, CONSTANT
        # ActAct, LINEAR, NONE
        # Act360, LINEAR, LINEAR
        # Act360, LINEAR, CONSTANT
        # Act360, LINEAR, NONE
        # 30U360, LINEAR, LINEAR
        # 30U360, LINEAR, CONSTANT
        # 30U360, LINEAR, NONE

        # TODO: TEST LIST of given dates

    def test_value_rate_and_yf(self):
        dc = DiscountCurve(
            "Test_DC_ACT365FIXED",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
            DayCounterType.Act365Fixed,
        )
        d = self.refdate + dt.timedelta(days=365)
        rate = dc.value_rate(self.refdate, d)
        df = dc.value(self.refdate, d)
        expected = -math.log(df) / DayCounter(DayCounterType.Act365Fixed).yf(self.refdate, d)
        self.assertAlmostEqual(rate, expected, delta=1e-12)

        df_yf = dc.value_yf(0.5)
        self.assertIsInstance(df_yf, float)

    def test_value_fwd_and_fwd_rate(self):
        dc = DiscountCurve(
            "Test_DC_ACT365FIXED",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
            DayCounterType.Act365Fixed,
        )

        d1 = self.refdate + dt.timedelta(days=365)
        d2 = self.refdate + dt.timedelta(days=730)
        fwd_df = dc.value_fwd(self.refdate, d1, d2)
        self.assertTrue(fwd_df < 1.0)
        fwd_rate = dc.value_fwd_rate(self.refdate, d1, d2)
        self.assertAlmostEqual(fwd_rate, -math.log(fwd_df) / DayCounter(DayCounterType.Act365Fixed).yf(d1, d2))

        # Value date > refdate triggers rebasement logic
        val_date = self.refdate + dt.timedelta(days=500)
        fwd_df2 = dc.value_fwd(val_date, d1, d2)
        self.assertIsInstance(fwd_df2, float)

        # Value date before refdate -> should raise
        with self.assertRaises(Exception):
            dc.value_fwd(self.refdate - dt.timedelta(days=1), d1, d2)

    def test_call_zero_rate_and_rate_for_dates(self):
        dc = DiscountCurve(
            "Test_DC_ACT365FIXED",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
            DayCounterType.Act365Fixed,
        )

        # Direct zero rate for year fraction
        z = dc(0.5)
        self.assertIsInstance(z, float)
        self.assertGreater(z, 0)

        # Zero rate for given date/refdate
        d = self.refdate + dt.timedelta(days=365)
        z2 = dc(0.5, self.refdate, d)
        self.assertIsInstance(z2, float)

    # --------------------- Error and edge case tests --------------------------

    def test_value_invalid_ref_before_curve_ref(self):
        dc = DiscountCurve(
            "Test_DC_ACT365FIXED",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
            DayCounterType.Act365Fixed,
        )
        with self.assertRaises(Exception):
            dc.value(self.refdate - dt.timedelta(days=1), self.refdate + dt.timedelta(days=1))

    def test_value_rate_invalid_ref_before_curve_ref(self):
        dc = DiscountCurve(
            "Test_DC_ACT365FIXED",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
            DayCounterType.Act365Fixed,
        )
        with self.assertRaises(Exception):
            dc.value_rate(self.refdate - dt.timedelta(days=1), self.refdate + dt.timedelta(days=1))

    # --------------------- Placeholder for HAGAN and plot ---------------------
    # TODO
    def test_value_HAGAN(self):
        """Basic smoke test for HAGAN interpolation type."""
        dc = DiscountCurve(
            "Test_DC_HAGAN",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.HAGAN_DF,
            ExtrapolationType.LINEAR,
            DayCounterType.Act365Fixed,
        )
        val = dc.value(self.refdate, self.refdate + dt.timedelta(days=180))
        self.assertIsInstance(val, float)

    def test_plot_value(self):
        """Plot test stub (if implemented in module)."""
        dc = DiscountCurve(
            "Test_DC_PLOT",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
            DayCounterType.Act365Fixed,
        )
        # The class itself doesn’t define plot(), but if added, ensure it runs
        if hasattr(dc, "plot"):
            dc.plot()  # Smoke test


if __name__ == "__main__":
    unittest.main()
