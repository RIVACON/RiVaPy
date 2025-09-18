import math
from unittest import main, TestCase
import datetime as dt

from matplotlib.dates import relativedelta
from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.tools.datetools import DayCounter
from rivapy.tools.enums import DayCounterType, ExtrapolationType, Instrument, InterpolationType
from rivapy.pricing.deposit_pricing import DepositPricer


class TestDepositSpecification(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDepositSpecification, self).__init__(*args, **kwargs)

    # Test to run
    # default behavior in dates
    # date validators
    # cashflow generation
    # valuation
    # _create_sample_cashflows
    # _to_dict

    fixing = dt.datetime(2024, 1, 1)
    start_tn = fixing + dt.timedelta(days=1)
    startdate = dt.datetime(2024, 1, 2)  # start after fixing [spot lag(=2) behavior]
    dep_on = DepositSpecification(obj_id="dep_on", fixing_date=fixing, start_date=fixing, end_date=dt.datetime(2024, 4, 3), rate=0.01)
    dep_on2 = DepositSpecification(obj_id="dep_on2", fixing_date=fixing, term="O/N", rate=0.01)
    dep_on3 = DepositSpecification(obj_id="dep_on3", fixing_date=fixing, start_date=fixing, term="T/N", rate=0.01)
    dep_on4 = DepositSpecification(obj_id="dep_on4", fixing_date=fixing, start_date=fixing + relativedelta(days=2), term="O/N", rate=0.01)
    dep_tn = DepositSpecification(obj_id="dep_tn", fixing_date=fixing, start_date=start_tn, end_date=dt.datetime(2024, 4, 3), rate=0.01)
    dep_tn2 = DepositSpecification(obj_id="dep_tn2", fixing_date=fixing, term="T/N", rate=0.01)
    dep_3M1 = DepositSpecification(obj_id="dep_3M1", start_date=startdate, term="3M", rate=0.01)

    def test_o_n_spot_lag_and_t_n_spot_lag(self):
        """Fixing date == start_date should set spot_lag to 0 (O/N).
        Fixing date + 1 day == start_date should set spot_lag to 1 (T/N).
        """

        # O/N: fixing == start_date -> spot_lag 0

        # DepositSpecification stores spot days in _spot_days (used in _to_dict)
        self.assertEqual(self.dep_on._spot_days, 0)
        self.assertEqual(self.dep_on2._spot_days, 0)
        with self.assertLogs(level="ERROR") as log:
            DepositSpecification(obj_id="dep_on3", fixing_date=self.fixing, start_date=self.fixing, term="T/N", rate=0.01)
            DepositSpecification(obj_id="dep_on4", fixing_date=self.fixing, start_date=self.fixing + relativedelta(days=2), term="O/N", rate=0.01)

        # T/N: start_date == fixing_date + 1 day -> spot_lag 1

        self.assertEqual(self.dep_tn._spot_days, 1)
        self.assertEqual(self.dep_tn2._spot_days, 1)
        with self.assertLogs(level="ERROR") as log:
            DepositSpecification(obj_id="dep_tn3", fixing_date=self.fixing, start_date=self.start_tn, term="O/N", rate=0.01)
            DepositSpecification(obj_id="dep_tn4", fixing_date=self.fixing, start_date=self.start_tn + relativedelta(days=2), term="T/N", rate=0.01)

    def test_fixingdate(self):
        self.assertEqual(self.dep_on._first_fixing_date, dt.datetime(2024, 1, 2))
        with self.assertLogs(level="WARNING") as log:
            DepositSpecification(obj_id="dep_fix", fixing_date=self.fixing, start_date=self.fixing, term="3M", rate=0.01)
        self.assertEqual(self.dep_3M1._first_fixing_date, dt.datetime(2023, 12, 29))

    def test_missing_dates_raises(self):
        """Neither fixing_date nor start_date provided -> ValueError"""
        with self.assertRaises(ValueError):
            DepositSpecification(obj_id="dep_missing", rate=0.01)

    def test_cashflow_amount_and_date_for_term(self):
        """Check that the cashflow equals notional + accrued interest
        computed with the deposit's day count convention.
        """
        start = dt.datetime(2024, 1, 2)  # Use a 6 month term (Period-like string supported by Period class)
        dep = DepositSpecification(obj_id="dep_6m", start_date=start, term="6M", rate=0.05, notional=1000.0, day_count_convention="30E360")
        self.assertEqual(
            DepositPricer.get_expected_cashflows(specification=dep),
            [(dt.datetime(2024, 7, 2), 25.0), (dt.datetime(2024, 1, 2), -1000.0), (dt.datetime(2024, 7, 2), 1000.0)],
        )

    def test_get_implied_simply_compounded_rate(self):
        """Check that the implied simply compounded rate equals the contract rate
        if the discount curve is flat and equals the contract rate.
        """
        start = dt.datetime(2024, 1, 2)
        dep = DepositSpecification(obj_id="dep_6m", start_date=start, term="6M", rate=0.05, notional=1000.0, day_count_convention="30E360")
        dcc = DayCounter(DayCounterType.ThirtyE360)
        from rivapy.marketdata import DiscountCurve

        # flat discount curve with 5% rate
        object_id = "TEST_CURVE"
        flat_rate = 0.05
        days_to_maturity = [1, 180, 365, 720, 3 * 365, 4 * 365, 10 * 365]
        dates = [start + dt.timedelta(days=d) for d in days_to_maturity]
        df = [math.exp(-d / 365.0 * flat_rate) for d in days_to_maturity]
        dc = DiscountCurve(
            id=object_id, refdate=start, dates=dates, df=df, interpolation=InterpolationType.LINEAR, extrapolation=ExtrapolationType.LINEAR
        )
        cont_df = dc.rivapy_valueFWD(val_date=start, d1=dep.start_date, d2=dep._end_date)
        self.assertAlmostEqual(
            DepositPricer.get_implied_simply_compounded_rate(val_date=start, specification=dep, discount_curve=dc),
            1 / dcc.yf(dep.start_date, dep._end_date) * (1 / cont_df - 1),
            places=10,
        )

    def test_create_sample_and_to_dict_and_ins_type(self):
        """_create_sample returns requested length; _to_dict contains expected keys;
        ins_type returns Instrument.DEPOSIT; get_end_date returns internal end date.
        """
        samples = DepositSpecification._create_sample(n_samples=5, seed=42)
        self.assertEqual(len(samples), 5)
        # each sample should be a dict containing key entries
        for s in samples:
            self.assertIn("fixing_date", s)
            self.assertIn("start_date", s)
            self.assertIn("maturity_date", s)
            self.assertIn("rate", s)

        # create a deposit and inspect _to_dict
        fixing = dt.datetime(2024, 2, 1)
        start = dt.datetime(2024, 2, 5)
        dep = DepositSpecification(obj_id="dep_check", fixing_date=fixing, start_date=start, term="3M", rate=0.02)
        d = dep._to_dict()
        # required keys from implementation
        expected_keys = [
            "obj_id",
            "fixing_date",
            "start_date",
            "maturity_date",
            "currency",
            "notional",
            "rate",
            "day_count_convention",
            "roll_convention",
            "spot_days",
            "business_day_convention",
            "issuer",
            "securitization_level",
            "payment_days",
        ]
        for k in expected_keys:
            self.assertIn(k, d)

        # ins_type
        self.assertEqual(dep.ins_type(), Instrument.DEPOSIT)

        # get_end_date returns the internal _end_date (implementation uses _end_date)
        self.assertEqual(dep.get_end_date(), dep._end_date)


#######################################################
# Tests for Pricing


if __name__ == "__main__":
    main()
