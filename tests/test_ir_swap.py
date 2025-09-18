import unittest
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import math


from rivapy.marketdata.curves import DiscountCurve
from rivapy.instruments.ir_swap_specification import (
    IrSwapLegSpecification,
    IrFixedLegSpecification,
    IrFloatLegSpecification,
    IrOISLegSpecification,
    InterestRateSwapSpecification,
)
from rivapy.instruments.notional_structure import ConstNotionalStructure
from rivapy.tools.enums import DayCounterType, IrLegType, Currency, RollConvention, SecuritizationLevel, Rating, InterpolationType, ExtrapolationType
from rivapy.pricing.interest_rate_swap_pricing import get_projected_notionals, InterestRateSwapPricer as Pricer



#of class InterestRateSwapPricer as Pricer
# #non-static methods are:
#price
# #static methods are
# _populate_cashflow_fixed
# _populate_cashflow_float
# _populate_cashflow_ois
# price_leg_pricing_data # not fully implemented - test later
# price_leg
# compute_swap_rate
# compute_swap_spread # not yett implemented
# compute_basis_spread # not yet implemented




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
    """Full swap with both leg tests

    Args:
        unittest (_type_): _description_
    """

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

class TestIRSwapSpecificationPricing(unittest.TestCase):
    """Test suite for pricing functionality of interest rate swaps.
    Note OIS is modelled after a plain IRS except with its own pricing function
    for the OIS/float leg.

    Args:
        unittest (_type_): _description_
    """

    @staticmethod
    def temp_ois_scheduler(start_dates: list, end_dates: list):

        # CONSIDER USING A SCHEDULER FUNCTION ONCE IT IS FINISHED
        daily_rate_start_dates = []  # 2D list: coupon i -> list of daily starts
        daily_rate_end_dates = []  # 2D list: coupon i -> list of daily ends
        daily_rate_reset_dates = []  # 2D list: coupon i -> list of reset dates
        pay_dates = []  # 1D list: one pay date per coupon

        for i in range(len(start_dates)):

            # for this test we keep it simple and ignore conventions e.g. business day or so. i.e just take every day
            num_days = (end_dates[i] - start_dates[i]).days
            daily_schedule = [start_dates[i] + timedelta(days=j) for j in range(num_days)]

            # Build start/end date pairs for accrual periods
            starts = daily_schedule[:-1]  # all except last
            ends = daily_schedule[1:]  # all except first

            daily_rate_start_dates.append(starts)
            daily_rate_end_dates.append(ends)

            # 4. Compute reset dates (fixing lag applied to each start)
            # resets = [add_business_days(start, fixingLag, rateHolidays)
            #           for start in starts]
            # assume simple case reset date is the same as start date
            resets = starts  # reset dates are equal to start dates if spot lag is 0.
            daily_rate_reset_dates.append(resets)

            # Compute payment date for the coupon
            # pay_date = add_business_days(end_dates[i], payLag, holidays)
            # assume simple case, pay date is end date
            pay_date = end_dates[i]
            pay_dates.append(pay_date)

        return [daily_rate_start_dates, daily_rate_end_dates, daily_rate_reset_dates, pay_dates]


    def setUp(self):
        """Setup of default values used throughout pricing
        """
        # by hand calculations to compare to
        # #discount curve
        # ttm = [0.5, 1.0, 1.5]
        # rates = [0.1, 0.105, 0.11]
        # N = 100 # Notional
        # m = 2 # compounding frequency i.e. every half year


        # ref_date = datetime(2017, 1, 1)

        # days_to_maturity = [180, 360, 540]
        # dates = [ref_date + timedelta(days=d) for d in days_to_maturity]
        # df = [math.exp(-r * t) for r, t in zip(rates, ttm)]




        # Discount curve - we use these discount factors to get the present values of both the fixed and floating leg as well as
        object_id = "TEST_DC"
        refdatedc = datetime(2017, 1, 1)
        days_to_maturity = [180, 360, 540]
        dates_dc = [refdatedc + timedelta(days=d) for d in days_to_maturity]
        # discount factors from constant rate
        rates = [0.10, 0.105, 0.11]
        df = [math.exp(-r * d / 360) for r, d in zip(rates, days_to_maturity)]
        dc = DiscountCurve(
            id=object_id, refdate=refdatedc, dates=dates_dc, df=df, interpolation=InterpolationType.LINEAR, extrapolation=ExtrapolationType.LINEAR
        )



        dcc = "Act360"
        ccy = "EUR"
        # Create the vectors defining the statdates, enddates, paydates and reset dates
        refdate = datetime(2017, 1, 1)
        days_to_maturity = [0, 180, 360, 540]
        dates = [ refdate + timedelta(d) for d in days_to_maturity]

        startdates = dates[:-1]
        enddates = dates[1:]
        paydates = enddates
        resetdates = startdates
        
        fixed_leg = IrFixedLegSpecification(
            fixed_rate=0.08,
            obj_id="dummy_fixed_leg",
            notional=100.0,
            start_dates=startdates,
            end_dates=enddates,
            pay_dates=paydates,
            currency=ccy,
            day_count_convention=dcc,
        )

        spread = 0.00
        ns = ConstNotionalStructure(100.0)

        float_leg = IrFloatLegSpecification(
            obj_id="dummy_float_leg",
            notional=ns,
            reset_dates=resetdates,
            start_dates=startdates,
            end_dates=enddates,
            rate_start_dates=startdates,
            rate_end_dates=enddates,
            pay_dates=paydates,
            currency=ccy,
            udl_id="test_udl_id",
            fixing_id="test_fixing_id",
            day_count_convention=dcc,
            spread=spread,
        )

        #in my other example, was an OIS with 6M tenor, and 6M maturity. with only 1 "interval"
        #here we will have more if using these startdates and enddates
        res = TestIRSwapSpecificationPricing.temp_ois_scheduler(startdates, enddates)
        daily_rate_start_dates = res[0]  # 2D list: coupon i -> list of daily starts
        daily_rate_end_dates = res[1]  # 2D list: coupon i -> list of daily ends
        daily_rate_reset_dates = res[2]  # 2D list: coupon i -> list of reset dates
        ois_pay_dates = res[3]

        ois_leg = IrOISLegSpecification(
            obj_id="dummy_ois_leg",
            notional=ns,
            rate_reset_dates=daily_rate_reset_dates, #reflects the overnight nature of OIS
            start_dates=startdates, # same as before
            end_dates=enddates, #same as before
            rate_start_dates=daily_rate_start_dates,#reflects the overnight nature of OIS
            rate_end_dates=daily_rate_end_dates,#reflects the overnight nature of OIS
            pay_dates=ois_pay_dates,
            currency=ccy,
            udl_id="test_udl_id",
            fixing_id="test_fixing_id",
            day_count_convention=dcc,
            rate_day_count_convention=dcc,
            spread=spread,
        )


        maturity_date = refdate + timedelta(600)
        # ir_swap = InterestRateSwapSpecification('TEST_SWAP', 'DBK', 'COLLATERALIZED', 'EUR', paydates[-1], fixedleg, floatleg)
        ir_swap = InterestRateSwapSpecification(
            obj_id="dummy_swap_6m",
            notional=ns,
            issue_date=refdate,
            maturity_date=maturity_date,
            pay_leg=fixed_leg,
            receive_leg=float_leg,
            currency=ccy,
            day_count_convention=dcc,
            issuer="dummy_issuer",
            securitization_level="COLLATERALIZED",
        )

        oi_swap = InterestRateSwapSpecification(
            obj_id="dummy_ois_6M",
            notional=ns,
            issue_date=refdate,
            maturity_date=maturity_date,
            pay_leg=fixed_leg,
            receive_leg=ois_leg,
            currency=ccy,
            day_count_convention=dcc,
            issuer="dummy_issuer",
            securitization_level="COLLATERALIZED",
        )
        self.refdate = refdate
        self.start_dates = startdates
        self.end_dates = enddates
        self.reset_dates = resetdates
        self.pay_dates = paydates
        self.maturity_date = maturity_date

        self.day_count_convention = dcc
        self.ccy = ccy
        self.dc = dc
        self.dc_rates = rates
        self.dc_dates =dates_dc
        self.dc_df = df

        self.ns = ns

        self.fixed_leg = fixed_leg
        self.float = float_leg
        self.ois_leg = ois_leg
        self.ir_swap = ir_swap
        self.oi_swap = oi_swap

    def get_projected_notionals(self):
        pass

    def populatechflow_fixed(self):
        pass

    def popoulate_cashflow_float(self):
        pass

    def populate_cashflow_ois(self):
        pass

    def price(self):
        pass


    def price_leg_pricing_data(self): #using the pricing data container structure
        pass   

    def price_leg(self):
        pass



    def compute_swap_rate(self):
        pass


# TODO once implemented: computeSwapSpread, computeBasisSpread





if __name__ == "__main__":
    unittest.main()
