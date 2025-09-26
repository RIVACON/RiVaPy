# hnguyen, 2024-09-08
# unit tests for Forward rate agreement specification class
import math
from holidays import ECB  # ??

import unittest
import datetime as dt
from matplotlib import dates
import numpy as np
from rivapy.instruments.fra_specifications import ForwardRateAgreementSpecification
from rivapy.marketdata.curves import DiscountCurve
from rivapy.pricing.fra_pricing import ForwardRateAgreementPricer
from rivapy.pricing.bond_pricing import DeterministicCashflowPricer
from rivapy.tools.datetools import DayCounter, roll_day
from rivapy.tools.enums import (
    DayCounterType,
    InterpolationType,
    RollConvention,
    SecuritizationLevel,
    Currency,
    Rating,
    Instrument,
)


class TestForwardRateAgreementSpecification(unittest.TestCase):

    # Set up FRA
    ccy = "EUR"
    fra_rate = 0.04
    ref_date = dt.datetime(2023, 1, 28)
    start_date = dt.datetime(2023, 7, 28)
    end_date = dt.datetime(2023, 10, 28)

    mat_date = ref_date + dt.timedelta(days=365)

    fra = ForwardRateAgreementSpecification(
        obj_id="dummy_id",
        trade_date=ref_date,
        notional=1000.0,
        rate=fra_rate,
        start_date=start_date,
        end_date=end_date,
        udlID="dummy_underlying_index",
        rate_start_date=start_date,
        rate_end_date=end_date,
        day_count_convention="Act360",
        rate_day_count_convention="Act360",
        currency=ccy,
        spot_lag=1,
        payment_days=1,
        issuer="dummy_issuer",
        securitization_level="NONE",
    )

    def setUp(self):
        """Common setup for tests"""
        self.trade_date = dt.date(2024, 1, 1)
        self.maturity_date = dt.date(2024, 12, 31)
        self.start_date = dt.date(2024, 3, 1)
        self.end_date = dt.date(2024, 9, 1)

        self.fra = ForwardRateAgreementSpecification(
            obj_id="fra_001",
            trade_date=self.trade_date,
            maturity_date=self.maturity_date,
            notional=1_000_000,
            rate=0.02,
            start_date=self.start_date,
            end_date=self.end_date,
            udlID="EURIBOR_3M",  # at the moment, dummy, not used in pricing YET
            rate_start_date=self.start_date - dt.timedelta(days=2),
            rate_end_date=self.end_date - dt.timedelta(days=2),
            calendar=ECB(years=range(2024, 2025)),
            currency="EUR",
            issuer="TestBank",
            securitization_level=SecuritizationLevel.NONE,
            rating=Rating.A,
            payment_days=2,
        )

    def test_initialization(self):
        self.assertEqual(self.fra.obj_id, "fra_001")
        self.assertEqual(self.fra.notional, 1_000_000)
        self.assertEqual(self.fra.rate, 0.02)
        self.assertEqual(self.fra.trade_date, self.trade_date)
        self.assertEqual(self.fra.maturity_date, self.maturity_date)
        self.assertEqual(self.fra.start_date, self.start_date)
        self.assertEqual(self.fra.end_date, self.end_date)
        self.assertEqual(self.fra.currency, "EUR")
        self.assertEqual(self.fra.issuer, "TestBank")
        self.assertEqual(self.fra.securitization_level, "NONE")
        self.assertEqual(self.fra.rating, "A")
        self.assertEqual(self.fra.payment_days, 2)

    def test_to_dict(self):
        d = self.fra._to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["obj_id"], "fra_001")
        self.assertEqual(d["currency"], "EUR")
        self.assertEqual(d["notional"], 1_000_000)

    def test_property_setters(self):
        self.fra.rate = 0.05
        self.assertEqual(self.fra.rate, 0.05)

        self.fra.notional = 5000
        self.assertEqual(self.fra.notional, 5000)

        self.fra.issuer = "NewBank"
        self.assertEqual(self.fra.issuer, "NewBank")

        self.fra.rating = Rating.B
        self.assertEqual(self.fra.rating, "B")

        self.fra.currency = Currency.USD  # should be fine despite the warning
        self.assertEqual(self.fra.currency, "USD")

    def test_invalid_notional_raises(self):
        with self.assertRaises(ValueError):
            self.fra.notional = -1000

    def test_ins_type(self):
        self.assertEqual(self.fra.ins_type(), Instrument.FRA)

    def test_get_end_date_alias(self):
        self.assertEqual(self.fra.get_end_date(), self.maturity_date)

    def test_create_sample_reproducibility(self):
        samples1 = ForwardRateAgreementSpecification._create_sample(3, seed=42, ref_date=self.trade_date)
        samples2 = ForwardRateAgreementSpecification._create_sample(3, seed=42, ref_date=self.trade_date)

        self.assertEqual(len(samples1), 3)
        self.assertEqual(len(samples2), 3)
        # Same seed → same first instrument spec
        self.assertEqual(samples1[0]["notional"], samples2[0]["notional"])
        self.assertEqual(samples1[0]["currency"], samples2[0]["currency"])

    def test_create_sample_structure(self):
        samples = ForwardRateAgreementSpecification._create_sample(2, seed=1, ref_date=self.trade_date)
        self.assertIsInstance(samples, list)
        self.assertIn("trade_date", samples[0])
        self.assertIn("maturity_date", samples[0])
        self.assertIn("currency", samples[0])


#######################################################
# Tests for Pricing
def test_fra_cf_implied_rate(self):
    # setting up necessary curves
    # discount curve
    object_id = "TEST_DC"
    dsc_rate = 0.01
    days_to_maturity = [1, 180, 365, 720, 3 * 365, 4 * 365, 10 * 365]
    dates = [self.ref_date + dt.timedelta(days=d) for d in days_to_maturity]
    df = [math.exp(-d / 365.0 * dsc_rate) for d in days_to_maturity]
    dc = DiscountCurve(id=object_id, refdate=self.ref_date, dates=dates, df=df, interpolation=InterpolationType.LINEAR)

    # Fixing curve
    object_id = "TEST_fwd"
    fwd_rate = 0.05
    fwd_df = [math.exp(-d / 365.0 * fwd_rate) for d in days_to_maturity]
    fwd_dc = DiscountCurve(id=object_id, refdate=self.ref_date, dates=dates, df=fwd_df, interpolation=InterpolationType.LINEAR)

    fra_pricer = ForwardRateAgreementPricer(self.ref_date, self.fra, dc, fwd_dc)

    # Manually calculate expected cashflows 'manually' for comparison
    dcc_rate = DayCounter(fwd_dc.daycounter)
    fwdrateDF = fwd_dc.value_fwd(self.ref_date, self.fra._rate_start_date, self.fra._rate_end_date)
    dt_rate = dcc_rate.yf(self.fra._rate_start_date, self.fra._rate_end_date)
    fwdrate = (1.0 / fwdrateDF - 1) / dt_rate

    # using instrument daycount convention to calculate delta t for cf amount calculation and discounting
    dcc = DayCounter(self.fra.day_count_convention)
    dt = dcc.yf(self.fra._start_date, self.fra._end_date)
    amount = self.fra._notional * (fwdrate - self.fra._rate) * dt
    cf = amount / (1 + fwdrate * dt)

    fair_rate = (1.0 / fwdrateDF - 1) / dt_rate

    self.assertEqual(fra_pricer._fra_spec, self.fra)
    self.assertEqual(fra_pricer._val_date, self.ref_date)
    self.assertEqual(fra_pricer._discount_curve, dc)
    self.assertEqual(fra_pricer._forward_curve, fwd_dc)

    self.assertEqual(
        fra_pricer.get_expected_cashflows(),
        [(roll_day(self.fra._start_date, self.fra._calendar, self.fra._business_day_convention, settle_days=self.fra._payment_days), cf)],
    )
    self.assertEqual(fra_pricer.compute_fair_rate(self.ref_date, self.fra, fwd_dc), fair_rate)


if __name__ == "__main__":
    unittest.main()
