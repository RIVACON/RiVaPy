import unittest
from datetime import date
from rivapy.instruments.rewrite import FixedRateBond, DiscountCurve
from rivapy.tools.datetools import Schedule, Period, DayCounterType

class TestFixedRateBondPricing(unittest.TestCase):

    def test_simple_annual_bond_on_coupon_date(self):
        """
        Scenario: A 2-year bond valued exactly one year before maturity, on a coupon payment date.
        Accrued interest must be zero in this case.
        """
        valuation_date = date(2024, 1, 15)
        issue_date = date(2023, 1, 15)
        maturity_date = date(2025, 1, 15)
        
        # --- Manual Calculation ---
        # Future cash flows from the valuation date:
        # - Coupon on 2025-01-15: 1000 * 0.05 = 50
        # - Notional on 2025-01-15: 1000
        # Total cash flow at maturity: 1050
        #
        # Discounting:
        # - Time to maturity (Act/365 Fixed): (2025-01-15 - 2024-01-15).days / 365 = 365 / 365 = 1.0 year
        # - Discount rate: 3%
        # - Dirty Price = 1050 / (1 + 0.03)^1 = 1019.4174757
        #
        # Accrued Interest:
        # - On a coupon date, the accrued interest is reset to 0.
        #
        # Clean Price = Dirty Price - Accrued Interest = 1019.4174757
        
        expected_dirty_price = 1019.417476
        expected_accrued = 0.0
        expected_clean_price = 1019.417476

        # --- Setup ---
        schedule = Schedule(start_day=issue_date, end_day=maturity_date, time_period=Period(years=1))
        bond = FixedRateBond(obj_id="test_bond_1", schedule=schedule, notional=1000.0, currency="EUR",
                             issue_date=issue_date, coupon_rate=0.05, 
                             accrual_day_counter_type=DayCounterType.ActActICMA)
        
        curve = DiscountCurve(valuation_date=valuation_date, flat_rate=0.03, 
                              day_counter_type=DayCounterType.Act365Fixed)

        # --- Tests ---
        self.assertAlmostEqual(bond.compute_accrued_interest(valuation_date), expected_accrued, places=6)
        self.assertAlmostEqual(bond.compute_dirty_price(curve), expected_dirty_price, places=6)
        self.assertAlmostEqual(bond.compute_clean_price(curve), expected_clean_price, places=6)

    def test_pricing_mid_period(self):
        """
        Scenario: A 2-year bond valued between two coupon dates.
        This test verifies the correct calculation of accrued interest.
        """
        valuation_date = date(2023, 7, 15)
        issue_date = date(2023, 1, 15)
        maturity_date = date(2025, 1, 15)
        
        # --- Manual Calculation ---
        # Future cash flows: (50 on 2024-01-15) and (1050 on 2025-01-15)
        #
        # Discounting (Act/365 Fixed, 3% Rate):
        # - Time to CF1 (2024-01-15): 184 days -> 184/365 = 0.5041 years. PV = 50 / (1.03)^0.5041 = 49.2611
        # - Time to CF2 (2025-01-15): 550 days -> 550/365 = 1.5068 years. PV = 1050 / (1.03)^1.5068 = 1004.1585
        # - Dirty Price = 49.2611 + 1004.1585 = 1053.4196
        #
        # Accrued Interest (Act/Act ICMA):
        # - Coupon period: 2023-01-15 to 2024-01-15 (365 days)
        # - Accrued days: 2023-01-15 to 2023-07-15 = 181 days
        # - Year Fraction = 181 / 365 = 0.49589
        # - Accrued Interest = 1000 * 0.05 * 0.49589 = 24.7945
        #
        # Clean Price = 1053.4196 - 24.7945 = 1028.6251
        
        expected_dirty_price = 1053.419602
        expected_accrued = 24.794521
        expected_clean_price = 1028.625081

        # --- Setup ---
        schedule = Schedule(start_day=issue_date, end_day=maturity_date, time_period=Period(years=1))
        bond = FixedRateBond(obj_id="test_bond_2", schedule=schedule, notional=1000.0, currency="EUR",
                             issue_date=issue_date, coupon_rate=0.05, 
                             accrual_day_counter_type=DayCounterType.ActActICMA)
        
        curve = DiscountCurve(valuation_date=valuation_date, flat_rate=0.03, 
                              day_counter_type=DayCounterType.Act365Fixed)

        # --- Tests ---
        self.assertAlmostEqual(bond.compute_accrued_interest(valuation_date), expected_accrued, places=6)
        self.assertAlmostEqual(bond.compute_dirty_price(curve), expected_dirty_price, places=6)
        self.assertAlmostEqual(bond.compute_clean_price(curve), expected_clean_price, places=6)

    def test_semi_annual_bond_mid_period(self):
        """
        Scenario: A bond with semi-annual coupons, priced between payment dates.
        """
        valuation_date = date(2024, 4, 1)
        issue_date = date(2023, 1, 1)
        maturity_date = date(2025, 1, 1)
        
        # --- Manual Calculation ---
        # Coupons are paid on Jan 1 and Jul 1. Coupon per period = 1000 * 0.05 / 2 = 25
        # Future cash flows from 2024-04-01:
        # - CF1: 25 on 2024-07-01
        # - CF2: 1025 on 2025-01-01
        #
        # Discounting (Act/365 Fixed, 3% Rate):
        # - Time to CF1: (2024-07-01 - 2024-04-01).days / 365 = 91 / 365 = 0.2493 years. PV = 25 / (1.03)^0.2493 = 24.8159
        # - Time to CF2: (2025-01-01 - 2024-04-01).days / 365 = 275 / 365 = 0.7534 years. PV = 1025 / (1.03)^0.7534 = 1002.2618
        # - Dirty Price = 24.8159 + 1002.2618 = 1027.0777
        #
        # Accrued Interest (Act/Act ICMA):
        # - Current coupon period: 2024-01-01 to 2024-07-01 (182 days)
        # - Accrued days: 2024-01-01 to 2024-04-01 = 91 days
        # - Year Fraction = 91 / (182 * 2) = 0.25 (Days in period * Freq)
        # - Accrued = 1000 * 0.05 * 0.25 = 12.5
        #
        # Clean Price = 1027.0777 - 12.5 = 1014.5777
        
        expected_dirty_price = 1027.077708
        expected_accrued = 12.5
        expected_clean_price = 1014.577708

        # --- Setup ---
        schedule = Schedule(start_day=issue_date, end_day=maturity_date, time_period=Period(months=6))
        bond = FixedRateBond(obj_id="test_bond_semi", schedule=schedule, notional=1000.0, currency="EUR",
                             issue_date=issue_date, coupon_rate=0.05, 
                             accrual_day_counter_type=DayCounterType.ActActICMA)
        
        curve = DiscountCurve(valuation_date=valuation_date, flat_rate=0.03, 
                              day_counter_type=DayCounterType.Act365Fixed)

        # --- Tests ---
        self.assertAlmostEqual(bond.compute_accrued_interest(valuation_date), expected_accrued, places=6)
        self.assertAlmostEqual(bond.compute_dirty_price(curve), expected_dirty_price, places=6)
        self.assertAlmostEqual(bond.compute_clean_price(curve), expected_clean_price, places=6)

    def test_pricing_at_maturity(self):
        """
        Scenario: A bond valued exactly on its maturity date.
        """
        valuation_date = date(2025, 1, 15)
        issue_date = date(2023, 1, 15)
        maturity_date = date(2025, 1, 15)
        
        # --- Manual Calculation ---
        # At maturity, the bond pays its final coupon and the notional. No discounting is needed.
        # - Dirty Price = Notional + Final Coupon = 1000 + (1000 * 0.05) = 1050
        #
        # Accrued Interest: The full final coupon has accrued.
        # - Coupon period: 2024-01-15 to 2025-01-15 (365 days)
        # - Accrued days: 365
        # - Accrued Interest = 1000 * 0.05 * (365/365) = 50
        #
        # Clean Price = Dirty Price - Accrued = 1050 - 50 = 1000
        
        expected_dirty_price = 1050.0
        expected_accrued = 50.0
        expected_clean_price = 1000.0

        # --- Setup ---
        schedule = Schedule(start_day=issue_date, end_day=maturity_date, time_period=Period(years=1))
        bond = FixedRateBond(obj_id="test_bond_maturity", schedule=schedule, notional=1000.0, currency="EUR",
                             issue_date=issue_date, coupon_rate=0.05, 
                             accrual_day_counter_type=DayCounterType.ActActICMA)
        
        curve = DiscountCurve(valuation_date=valuation_date, flat_rate=0.03, 
                              day_counter_type=DayCounterType.Act365Fixed)

        # --- Tests ---
        self.assertAlmostEqual(bond.compute_accrued_interest(valuation_date), expected_accrued, places=6)
        self.assertAlmostEqual(bond.compute_dirty_price(curve), expected_dirty_price, places=6)
        self.assertAlmostEqual(bond.compute_clean_price(curve), expected_clean_price, places=6)

    def test_pricing_after_maturity(self):
        """
        Scenario: A bond valued after it has already matured.
        All prices and accrued interest should be zero.
        """
        valuation_date = date(2026, 1, 15)
        issue_date = date(2023, 1, 15)
        maturity_date = date(2025, 1, 15)
        
        # --- Manual Calculation ---
        # The bond has already matured, so there are no future cash flows.
        # All values should be zero.
        
        expected_dirty_price = 0.0
        expected_accrued = 0.0
        expected_clean_price = 0.0

        # --- Setup ---
        schedule = Schedule(start_day=issue_date, end_day=maturity_date, time_period=Period(years=1))
        bond = FixedRateBond(obj_id="test_bond_expired", schedule=schedule, notional=1000.0, currency="EUR",
                             issue_date=issue_date, coupon_rate=0.05, 
                             accrual_day_counter_type=DayCounterType.ActActICMA)
        
        curve = DiscountCurve(valuation_date=valuation_date, flat_rate=0.03, 
                              day_counter_type=DayCounterType.Act365Fixed)

        # --- Tests ---
        self.assertAlmostEqual(bond.compute_accrued_interest(valuation_date), expected_accrued, places=6)
        self.assertAlmostEqual(bond.compute_dirty_price(curve), expected_dirty_price, places=6)
        self.assertAlmostEqual(bond.compute_clean_price(curve), expected_clean_price, places=6)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)