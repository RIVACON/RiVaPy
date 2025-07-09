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
    # rivapy_value -  cares about daycount convention refdates, target date, interpolation type, extrapolation type..
    #  at the moment, it is assumed the target date is correctly calculated  before input with correct business day logic/roll convention
    #  there is at the moment a potential issue with the roll convention and the given reference date, for now assume it is correct #TODO
    # plot - calls the value or rivapy_value in order to plot.
    # TODO: consider how to wrape both value and rivapy_value depending on if the interpolationType is a pyvacon or Rivapy construction respectively

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

        # object_id = "Test_DC_ACT365FIXED"
        # self.dsc_fac_ACT365FIXED = [math.exp(-self.rates[d] * self.days_to_maturity[d] / 365) for d in range(len(self.rates))]
        self.dsc_fac_ACT365FIXED = [math.exp(-d / 365.0 * self.rate) for d in self.days_to_maturity]
        # self.test_dc_ACT365FIXED = {}

        # dc_linear = DiscountCurve(
        #    "Test_DC_ACT365FIXED", refdate, dates, dsc_fac, InterpolationType.LINEAR, ExtrapolationType.LINEAR, DayCounterType.Act365Fixed
        # )

    def test_get_dates(self):
        """_summary_"""

        dc_linear = DiscountCurve(
            "Test_DC_ACT365FIXED",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
            DayCounterType.Act365Fixed,
        )

        expected = tuple()  # TODO

        pass

    def test_get_df(self):
        """_summary_"""
        dc_linear = DiscountCurve(
            "Test_DC_ACT365FIXED",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
            DayCounterType.Act365Fixed,
        )

        expected = tuple()  # TODO
        pass

    def test_rivapy_value(self):
        """_summary_"""

        # DDC, interpolation, extrapolation
        # ACT365FIXED, LINEAR, LINEAR
        dc_linear = DiscountCurve(
            "Test_DC_ACT365FIXED",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
            DayCounterType.Act365Fixed,
        )

        df1 = dc_linear.rivapy_value(self.refdate, self.refdate + dt.timedelta(days=90))
        df2 = dc_linear.rivapy_value(self.refdate, self.refdate + dt.timedelta(days=180))
        fwd_df = dc_linear.rivapy_value(self.refdate + dt.timedelta(days=90), self.refdate + dt.timedelta(days=180))

        self.assertAlmostEqual(df1, 0.9926568878362608, delta=1e-5)
        self.assertAlmostEqual(df2, 0.9853143806626516, delta=1e-5)
        self.assertAlmostEqual(fwd_df, df2 / df1, delta=1e-5)

        # extrapolation
        df1 = dc_linear.rivapy_value(self.refdate, self.refdate + dt.timedelta(days=10 * 365 + 10))
        df2 = dc_linear.rivapy_value(self.refdate, self.refdate + dt.timedelta(days=10 * 365 + 60))
        fwd_df = dc_linear.rivapy_value(self.refdate + dt.timedelta(days=10 * 365 + 10), self.refdate + dt.timedelta(days=10 * 365 + 60))

        self.assertAlmostEqual(df1, 0.7401510872751634, delta=1e-5)  # cf with pyvacon results
        self.assertAlmostEqual(df2, 0.7368154202423907, delta=1e-5)
        self.assertAlmostEqual(fwd_df, df2 / df1, delta=1e-5)

        # ACT365FIXED, LINEAR, CONSTANT
        dc_linear = DiscountCurve(
            "Test_DC_ACT365FIXED",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.LINEAR,
            ExtrapolationType.CONSTANT,
            DayCounterType.Act365Fixed,
        )

        # interpolation
        df1 = dc_linear.rivapy_value(self.refdate, self.refdate + dt.timedelta(days=90))
        df2 = dc_linear.rivapy_value(self.refdate, self.refdate + dt.timedelta(days=180))
        fwd_df = dc_linear.rivapy_value(self.refdate + dt.timedelta(days=90), self.refdate + dt.timedelta(days=180))

        self.assertAlmostEqual(df1, 0.9926568878362608, delta=1e-5)  # cf with pyvacon results
        self.assertAlmostEqual(df2, 0.9853143806626516, delta=1e-5)
        self.assertAlmostEqual(fwd_df, df2 / df1, delta=1e-5)

        # extrapolation
        df1 = dc_linear.rivapy_value(self.refdate, self.refdate + dt.timedelta(days=10 * 365 + 10))
        df2 = dc_linear.rivapy_value(self.refdate, self.refdate + dt.timedelta(days=10 * 365 + 60))
        fwd_df = dc_linear.rivapy_value(self.refdate + dt.timedelta(days=10 * 365 + 10), self.refdate + dt.timedelta(days=10 * 365 + 60))

        self.assertAlmostEqual(df1, 0.7408182206817179, delta=1e-5)  # cf with pyvacon results
        self.assertAlmostEqual(df2, 0.7408182206817179, delta=1e-5)
        self.assertAlmostEqual(fwd_df, df2 / df1, delta=1e-5)

        # ACT365FIXED, LINEAR, NONE -  EXPECT ERROR TO BE THROWN for EXTRAPOLATIOn
        dc_linear = DiscountCurve(
            "Test_DC_ACT365FIXED",
            self.refdate,
            self.dates,
            self.dsc_fac_ACT365FIXED,
            InterpolationType.LINEAR,
            ExtrapolationType.NONE,
            DayCounterType.Act365Fixed,
        )

        # interpolation
        df1 = dc_linear.rivapy_value(self.refdate, self.refdate + dt.timedelta(days=90))
        df2 = dc_linear.rivapy_value(self.refdate, self.refdate + dt.timedelta(days=180))
        fwd_df = dc_linear.rivapy_value(self.refdate + dt.timedelta(days=90), self.refdate + dt.timedelta(days=180))

        self.assertAlmostEqual(df1, 0.9926568878362608, delta=1e-5)  # cf with pyvacon results
        self.assertAlmostEqual(df2, 0.9853143806626516, delta=1e-5)
        self.assertAlmostEqual(fwd_df, df2 / df1, delta=1e-5)

        # extrapolation
        with self.assertRaises(ValueError):
            df1 = dc_linear.rivapy_value(self.refdate, self.refdate + dt.timedelta(days=10 * 365 + 10))
        with self.assertRaises(ValueError):
            df2 = dc_linear.rivapy_value(self.refdate, self.refdate + dt.timedelta(days=10 * 365 + 60))
        with self.assertRaises(ValueError):
            fwd_df = dc_linear.rivapy_value(self.refdate + dt.timedelta(days=10 * 365 + 10), self.refdate + dt.timedelta(days=10 * 365 + 60))

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

    def test_rivapy_value_HAGAN(self):
        """_summary_"""

        pass

    def test_plot_rivapy_value(self):
        """_summary_"""

        pass


if __name__ == "__main__":
    unittest.main()
