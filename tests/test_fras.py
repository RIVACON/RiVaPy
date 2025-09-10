# hnguyen, 2024-09-08
# unit tests for Forward rate agreement specification class
from holidays import ECB  # ??

import unittest
import datetime as dt
import numpy as np
from rivapy.instruments.fra_specifications import ForwardRateAgreementSpecification
from rivapy.tools.datetools import DayCounter
from rivapy.tools.enums import (
    DayCounterType,
    RollConvention,
    SecuritizationLevel,
    Currency,
    Rating,
    Instrument,
)


class TestForwardRateAgreementSpecification(unittest.TestCase):
    def setUp(self):
        """Common setup for tests"""
        self.issue_date = dt.date(2024, 1, 1)
        self.maturity_date = dt.date(2024, 12, 31)
        self.start_date = dt.date(2024, 3, 1)
        self.end_date = dt.date(2024, 9, 1)

        self.fra = ForwardRateAgreementSpecification(
            obj_id="fra_001",
            issue_date=self.issue_date,
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
        self.assertEqual(self.fra.issue_date, self.issue_date)
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
        samples1 = ForwardRateAgreementSpecification._create_sample(3, seed=42, ref_date=self.issue_date)
        samples2 = ForwardRateAgreementSpecification._create_sample(3, seed=42, ref_date=self.issue_date)

        self.assertEqual(len(samples1), 3)
        self.assertEqual(len(samples2), 3)
        # Same seed → same first instrument spec
        self.assertEqual(samples1[0]["notional"], samples2[0]["notional"])
        self.assertEqual(samples1[0]["currency"], samples2[0]["currency"])

    def test_create_sample_structure(self):
        samples = ForwardRateAgreementSpecification._create_sample(2, seed=1, ref_date=self.issue_date)
        self.assertIsInstance(samples, list)
        self.assertIn("issue_date", samples[0])
        self.assertIn("maturity_date", samples[0])
        self.assertIn("currency", samples[0])


#######################################################
# Tests for Pricing


if __name__ == "__main__":
    unittest.main()
