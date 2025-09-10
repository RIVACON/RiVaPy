import unittest
from datetime import datetime, timedelta

from rivapy.instruments.ir_swap_specification import (
    IrSwapLegSpecification,
    IrFixedLegSpecification,
    IrFloatLegSpecification,
    IrOISLegSpecification,
    InterestRateSwapSpecification,
)
from rivapy.instruments.notional_structure import ConstNotionalStructure
from rivapy.tools.enums import DayCounterType, IrLegType, Currency, RollConvention, SecuritizationLevel, Rating


class TestIrSwapLegSpecification(unittest.TestCase):
    def setUp(self):
        """General setup for tests with some default values e.g., dates and notional"""
        self.start_dates = [datetime(2024, 1, 1)]
        self.end_dates = [datetime(2025, 1, 1)]
        self.pay_dates = [datetime(2025, 1, 1)]
        self.notional = 1000.0

    def test_init_and_properties(self):
        """Test for the initialization and properties of the IrSwapLegSpecification class"""
        leg = IrSwapLegSpecification(
            obj_id="leg1",
            notional=self.notional,
            start_dates=self.start_dates,
            end_dates=self.end_dates,
            pay_dates=self.pay_dates,
            currency="EUR",
            day_count_convention=DayCounterType.ThirtyU360,
        )
        self.assertEqual(leg.obj_id, "leg1")
        self.assertEqual(leg.currency, "EUR")
        self.assertEqual(leg.start_dates, self.start_dates)
        self.assertEqual(leg.end_dates, self.end_dates)
        self.assertEqual(leg.pay_dates, self.pay_dates)
        self.assertIsInstance(leg.notional_structure, ConstNotionalStructure)

    def test_notional_structure_setter(self):
        """Test for use of NotionalStructure class used in IrSwapLegSpecification.
        For more spepcific tests of NotionalStructure, see its own test class.

        #TODO add uses cases of different notional structures
        """
        ns = ConstNotionalStructure(5000.0)
        leg = IrSwapLegSpecification(
            obj_id="leg2",
            notional=ns,
            start_dates=self.start_dates,
            end_dates=self.end_dates,
            pay_dates=self.pay_dates,
            currency="USD",
        )
        self.assertIs(leg.notional_structure, ns)


class TestIrFixedLegSpecification(unittest.TestCase):
    """Similar to TestIrSwapLegSpecification but for fixed leg specific

    Args:
        unittest (_type_): _description_
    """

    def setUp(self):
        self.start_dates = [datetime(2024, 1, 1)]
        self.end_dates = [datetime(2025, 1, 1)]
        self.pay_dates = [datetime(2025, 1, 1)]
        self.notional = 1000.0

    def test_fixed_leg(self):
        leg = IrFixedLegSpecification(
            fixed_rate=0.01,
            obj_id="fixed_leg",
            notional=self.notional,
            start_dates=self.start_dates,
            end_dates=self.end_dates,
            pay_dates=self.pay_dates,
            currency="EUR",
        )
        self.assertEqual(leg.leg_type, IrLegType.FIXED)
        self.assertAlmostEqual(leg.fixed_rate, 0.01)
        self.assertEqual(leg.udl_id, "")


class TestIrFloatLegSpecification(unittest.TestCase):
    """Similar to TestIrSwapLegSpecification but for float leg specific

    Args:
        unittest (_type_): _description_
    """

    def setUp(self):
        self.start_dates = [datetime(2024, 1, 1)]
        self.end_dates = [datetime(2025, 1, 1)]
        self.pay_dates = [datetime(2025, 1, 1)]
        self.reset_dates = [datetime(2024, 1, 1)]
        self.rate_start_dates = [datetime(2024, 1, 1)]
        self.rate_end_dates = [datetime(2025, 1, 1)]
        self.notional = 1000.0

    def test_float_leg(self):
        leg = IrFloatLegSpecification(
            obj_id="float_leg",
            notional=self.notional,
            reset_dates=self.reset_dates,
            start_dates=self.start_dates,
            end_dates=self.end_dates,
            rate_start_dates=self.rate_start_dates,
            rate_end_dates=self.rate_end_dates,
            pay_dates=self.pay_dates,
            currency="USD",
            udl_id="SOFR",
            fixing_id="SOFR_FIX",
            spread=0.002,
        )
        self.assertEqual(leg.leg_type, IrLegType.FLOAT)
        self.assertEqual(leg.udl_id, "SOFR")
        self.assertEqual(leg.fixing_id, "SOFR_FIX")
        self.assertAlmostEqual(leg.spread, 0.002)
        self.assertEqual(leg.reset_dates, self.reset_dates)


class TestInterestRateSwapSpecification(unittest.TestCase):
    def setUp(self):
        self.start_dates = [datetime(2024, 1, 1)]
        self.end_dates = [datetime(2025, 1, 1)]
        self.pay_dates = [datetime(2025, 1, 1)]
        self.notional = 1000.0
        self.fixed_leg = IrFixedLegSpecification(
            fixed_rate=0.01,
            obj_id="fixed_leg",
            notional=self.notional,
            start_dates=self.start_dates,
            end_dates=self.end_dates,
            pay_dates=self.pay_dates,
            currency="EUR",
        )
        self.float_leg = IrFloatLegSpecification(
            obj_id="float_leg",
            notional=self.notional,
            reset_dates=self.start_dates,
            start_dates=self.start_dates,
            end_dates=self.end_dates,
            rate_start_dates=self.start_dates,
            rate_end_dates=self.end_dates,
            pay_dates=self.pay_dates,
            currency="USD",
            udl_id="SOFR",
            fixing_id="SOFR_FIX",
            spread=0.002,
        )

    def test_swap_specification(self):
        issue_date = datetime(2024, 1, 1)
        maturity_date = datetime(2025, 1, 1)
        spec = InterestRateSwapSpecification(
            obj_id="swap1",
            notional=self.notional,
            issue_date=issue_date,
            maturity_date=maturity_date,
            pay_leg=self.fixed_leg,
            receive_leg=self.float_leg,
            currency="EUR",
            day_count_convention=DayCounterType.ThirtyU360,
            business_day_convention=RollConvention.FOLLOWING,
            issuer="TestIssuer",
            securitization_level=SecuritizationLevel.NONE,
            rating=Rating.NONE,
        )
        self.assertEqual(spec.obj_id, "swap1")
        self.assertEqual(spec.issue_date, issue_date)
        self.assertEqual(spec.maturity_date, maturity_date)
        self.assertEqual(spec.pay_leg, self.fixed_leg)
        self.assertEqual(spec.receive_leg, self.float_leg)
        self.assertEqual(spec.currency, "EUR")
        self.assertEqual(spec.issuer, "TestIssuer")
        self.assertEqual(spec.securitization_level, SecuritizationLevel.to_string(SecuritizationLevel.NONE))
        self.assertEqual(spec.rating, Rating.to_string(Rating.NONE))
        self.assertIsInstance(spec.notional_structure, ConstNotionalStructure)

    def test_get_fixed_and_float_leg(self):
        spec = InterestRateSwapSpecification(
            obj_id="swap2",
            notional=self.notional,
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2025, 1, 1),
            pay_leg=self.fixed_leg,
            receive_leg=self.float_leg,
        )
        self.assertIs(spec.get_fixed_leg(), self.fixed_leg)
        self.assertIs(spec.get_float_leg(), self.float_leg)

    def test_get_fixed_leg_error(self):
        # Both legs fixed should raise
        fixed_leg2 = IrFixedLegSpecification(
            fixed_rate=0.01,
            obj_id="fixed_leg2",
            notional=self.notional,
            start_dates=self.start_dates,
            end_dates=self.end_dates,
            pay_dates=self.pay_dates,
            currency="EUR",
        )
        spec = InterestRateSwapSpecification(
            obj_id="swap3",
            notional=self.notional,
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2025, 1, 1),
            pay_leg=self.fixed_leg,
            receive_leg=fixed_leg2,
        )
        with self.assertRaises(ValueError):
            spec.get_fixed_leg()

    def test_get_float_leg_error(self):
        # Both legs fixed should raise
        fixed_leg2 = IrFixedLegSpecification(
            fixed_rate=0.01,
            obj_id="fixed_leg2",
            notional=self.notional,
            start_dates=self.start_dates,
            end_dates=self.end_dates,
            pay_dates=self.pay_dates,
            currency="EUR",
        )
        spec = InterestRateSwapSpecification(
            obj_id="swap4",
            notional=self.notional,
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2025, 1, 1),
            pay_leg=self.fixed_leg,
            receive_leg=fixed_leg2,
        )
        with self.assertRaises(ValueError):
            spec.get_float_leg()


#######################################################
# Tests for Pricing


if __name__ == "__main__":
    unittest.main()
