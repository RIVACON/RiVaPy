# hnguyen, 2024-09-08
# unit tests for DepositSpecification class
import unittest
import datetime as dt
from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.tools.datetools import DayCounter
from rivapy.tools.enums import DayCounterType, Instrument


class TestDepositSpecification(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDepositSpecification, self).__init__(*args, **kwargs)

    def test_o_n_spot_lag_and_t_n_spot_lag(self):
        """Fixing date == start_date should set spot_lag to 0 (O/N).
        Fixing date + 1 day == start_date should set spot_lag to 1 (T/N).
        """
        fixing = dt.datetime(2024, 6, 3)
        # O/N: fixing == start_date -> spot_lag 0
        dep_on = DepositSpecification(
            obj_id="dep_on",
            fixing_date=fixing,
            start_date=fixing,
            rate=0.01,
        )
        # DepositSpecification stores spot days in _spot_days (used in _to_dict)
        self.assertEqual(dep_on._spot_days, 0)

        # T/N: start_date == fixing_date + 1 day -> spot_lag 1
        fixing = dt.datetime(2024, 6, 3)
        start_tn = fixing + dt.timedelta(days=1)
        dep_tn = DepositSpecification(
            obj_id="dep_tn",
            fixing_date=fixing,
            start_date=start_tn,
            rate=0.01,
        )
        self.assertEqual(dep_tn._spot_days, 1)

    def test_missing_dates_raises(self):
        """Neither fixing_date nor start_date provided -> ValueError"""
        with self.assertRaises(ValueError):
            DepositSpecification(obj_id="dep_missing", rate=0.01)

    def test_cashflow_amount_and_date_for_term(self):
        """Check that the single maturity cashflow equals notional + accrued interest
        computed with the deposit's day count convention (default ACT/360).
        """
        fixing = dt.datetime(2024, 1, 2)
        start = dt.datetime(2024, 1, 4)  # start after fixing [spot lag(=2) behavior]
        # Use a 6 month term (Period-like string supported by class)

        # TODO
        # dep = DepositSpecification(
        #     obj_id="dep_6m",
        #     fixing_date=fixing,
        #     start_date=start,
        #     term="6M",
        #     rate=0.05,
        #     notional=1000.0,
        #     day_count_convention=DayCounterType.ACT360,  # default but explicit here
        # )

        # TODO
        # 2025.09.09 HN currentls throws error due to cal_end_day
        # if roll_conv == RollRule.EOM.value and _is_ambiguous_date(start_date):  in line 888 in datetools.py
        #
        # there is no check for is roll_conv is EOM and date is NOT ambiguous. will trigger exception and no end date is set
        # what is then the desired output? start date in question used was 2025.01.04 -> not ambiguous
        # should be 2025.07.04 as end date?

        # # The deposit should produce a single cashflow at maturity with interest + notional.
        # cfs = dep.cashflows # this is assuming same construction as bonds, hoowever, deposits.cashflows is not yet inplemented #TODO
        # # Expect one cashflow
        # self.assertEqual(len(cfs), 1)

        # payment_date, amount = cfs[0]

        # # Deposit's end_date and maturity_date are set by constructor; get start and end used
        # sd = dep.start_date
        # ed = dep.end_date
        # md = dep.maturity_date

        # # Check payment date equals maturity_date (maturity / payment convention)
        # self.assertEqual(payment_date, md)

        # # Compute expected interest: year fraction * rate * notional using ACT/360
        # yf = DayCounter.yf_Act360(d1=sd, d2=ed)
        # expected_interest = yf * dep.rate * dep.notional
        # expected_amount = dep.notional + expected_interest

        # # allow small floating rounding tolerance
        # self.assertAlmostEqual(expected_amount, amount, places=10)

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
    unittest.main()
