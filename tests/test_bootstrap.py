# 2025.09.09 Bootstrapping without pyvacon
import unittest
import math
from datetime import date, datetime, timedelta

from rivapy.marketdata.bootstrapping_2025 import (
    bootstrap_curve,
    error_fn,
    find_bracket,
    get_quote,
)
from rivapy.marketdata.curves import DiscountCurve
from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.instruments.fra_specifications import ForwardRateAgreementSpecification
from rivapy.instruments.ir_swap_specification import (
    InterestRateSwapSpecification,
    IrFixedLegSpecification,
    IrFloatLegSpecification,
)
from rivapy.tools.enums import DayCounterType, InterpolationType, ExtrapolationType, Instrument


# Minimal instrument specification classes for testing
class DummyDepositSpec(DepositSpecification):
    def __init__(self, end_date=None, start_date=None, ref_date=None):
        """Setting up base deposit specification for tests.
        For now, as O/N deposit with 1 day accrual.
        """
        ##########################################
        # setting up depoist
        # calculation date
        if ref_date is None:
            ref_date = datetime(2019, 8, 31)

        # start date of the accrual period with spot lag equal to 2 days
        if start_date is None:
            start_date = ref_date + timedelta(days=2)

        # end date of the accrual period is 1 day after startdate
        if end_date is None:
            end_date = start_date + timedelta(days=1)

        super().__init__(
            obj_id="dummy_deposit",
            issuer="dummy_issuer",
            currency="EUR",
            fixing_date=ref_date,
            start_date=start_date,
            maturity_date=end_date,
            notional=100.0,
            rate=0.01,
            day_count_convention="Act360",
        )


class DummyFRASpec:
    def __init__(self, end_date):
        self._end_date = end_date

    def get_end_date(self):
        return self._end_date

    def ins_type(self):
        return Instrument.FRA


class DummyIRSSpec:
    def __init__(self, end_date):
        self._end_date = end_date

    def get_end_date(self):
        return self._end_date

    def ins_type(self):
        return Instrument.IRS

    def get_float_leg(self):
        return None

    def get_fixed_leg(self):
        return None


class TestBootstrapCurve(unittest.TestCase):

    def test_input_length_assertion(self):
        """Test that providing different lengths of instruments and quotes raises an AssertionError."""
        with self.assertRaises(AssertionError):
            bootstrap_curve(
                ref_date=date(2024, 1, 1),
                curve_id="curve1",
                day_count_convention=DayCounterType.ThirtyU360,
                instruments=[DummyDepositSpec(end_date=datetime(2025, 1, 1))],
                quotes=[],
            )

    def test_duplicate_end_dates(self):
        """Test that duplicate end dates in instruments raises an error.
        This is a design choice for the moment to keep bootstrapping logic simple.
        """

        inst1 = DummyDepositSpec(date(2025, 1, 1))
        inst2 = DummyDepositSpec(date(2025, 1, 1))
        with self.assertRaises(Exception) as cm:
            bootstrap_curve(
                ref_date=datetime(2024, 1, 1),
                curve_id="curve1",
                day_count_convention=DayCounterType.ThirtyU360,
                instruments=[inst1, inst2],
                quotes=[0.01, 0.02],
            )
        self.assertIn("Duplicate expiry date", str(cm.exception))

    def test_successful_bootstrap(self):
        """Test if bootstrap_curve runs without error and returns a DiscountCurve
        Does not check for correctness of the curve.
        """
        # here this end_date is being saved into the maturity date..., and the end date is calculated internally in deposit spec
        # inst = DummyDepositSpec(ref_date=datetime(2024, 1, 1), start_date=datetime(2024, 1, 1), end_date=datetime(2025, 1, 1))
        # here we input an end date that would coincide with the maturity date by desgien
        inst = DummyDepositSpec(ref_date=datetime(2024, 1, 1), start_date=datetime(2024, 1, 1), end_date=datetime(2024, 1, 3))
        result = bootstrap_curve(
            ref_date=datetime(2024, 1, 1),
            curve_id="curve1",
            day_count_convention=DayCounterType.ThirtyU360,
            instruments=[inst],
            quotes=[0.01],
        )

        self.assertIsInstance(result, DiscountCurve)
        self.assertEqual(result.get_dates()[0], datetime(2024, 1, 1))
        self.assertEqual(result.get_dates()[1], datetime(2024, 1, 3))


class TestErrorFn(unittest.TestCase):
    """Tests on the error function used for the brentq solver used for iteratively solving for discount factors
    regardless of instrument type.

    Args:
        unittest (_type_): _description_
    """

    def test_error_fn_returns_float(self):
        """Test function output type check"""
        inst = DummyDepositSpec(datetime(2024, 1, 3))
        dfs = [1.0, 0.99]
        yc_dates = [datetime(2024, 1, 1), datetime(2024, 1, 3)]
        curves = {
            "discount_curve": DiscountCurve(
                "test", datetime(2024, 1, 1), yc_dates, dfs, InterpolationType.LINEAR, ExtrapolationType.LINEAR, DayCounterType.ThirtyU360
            )
        }
        result = error_fn(
            df_val=0.98,
            index=1,
            dfs=dfs.copy(),
            yc_dates=yc_dates,
            instrument_spec=inst,
            ref_date=datetime(2024, 1, 1),
            ref_quote=0.01,
            curves=curves,
            interpolation_type=InterpolationType.LINEAR,
            extrapolation_type=ExtrapolationType.LINEAR,
        )
        self.assertIsInstance(result, float)


class TestFindBracket(unittest.TestCase):
    """Test for the helper function used in initial testing for bretnq solver.
    Helper function was made for the case that inital guesses
    were far off from potential root values to find better initial brackets.

    Args:
        unittest (_type_): _description_
    """

    def test_find_bracket_success(self):
        """Caveat is that the inital guess should not produce a boundary value that is also the intended root."""

        def fake_error_fn(x, *args):
            return x - 1

        lower, upper = find_bracket(fake_error_fn, 1.5)
        self.assertLess(lower, 1)
        self.assertGreater(upper, 1)

    def test_find_bracket_failure(self):
        def fake_error_fn(x, *args):
            return 1

        with self.assertRaises(RuntimeError):
            find_bracket(fake_error_fn, 2)


class TestGetQuote(unittest.TestCase):
    """Testing get_quote for different instrument types

    Args:
        unittest (_type_): _description_
    """

    def test_get_quote_deposit(self):
        refdate = datetime(2024, 1, 1)
        inst = DummyDepositSpec(ref_date=refdate, start_date=refdate, end_date=datetime(2024, 1, 3))
        days_to_maturity = [1, 180, 365, 720, 3 * 365, 4 * 365, 10 * 365]
        dates = [refdate + timedelta(days=d) for d in days_to_maturity]
        flat_rate = 0.025
        df = [math.exp(-d / 365.0 * flat_rate) for d in days_to_maturity]
        curve = DiscountCurve(
            id="dummy discount curve",
            refdate=refdate,
            dates=dates,
            df=df,
            interpolation=InterpolationType.LINEAR,
            extrapolation=ExtrapolationType.LINEAR,
        )
        result = get_quote(
            ref_date=refdate,
            instrument_spec=inst,
            curve_dict={"discount_curve": curve},
        )
        # self.assertEqual(result, 0.01)  # TODO input expected calculated value, 0.01 is dummy
        self.assertAlmostEqual(result, 0.024508664452316253, delta=1e-8)

    def test_get_quote_fra(self):  # TODO -> verify correctness of the FRA specificaiton first... changes were made
        pass
        # inst = DummyFRASpec(date(2025, 1, 1))
        # curve = DiscountCurve(
        #     "test",
        #     date(2024, 1, 1),
        #     [date(2024, 1, 1), date(2025, 1, 1)],
        #     [1.0, 0.99],
        #     InterpolationType.LINEAR,
        #     ExtrapolationType.LINEAR,
        #     DayCounterType.ThirtyU360,
        # )
        # result = get_quote(
        #     ref_date=date(2024, 1, 1),
        #     instrument_spec=inst,
        #     curve_dict={"discount_curve": curve},
        # )
        # self.assertEqual(result, 0.02)  # TODO input expected calculated value, 0.02 is dummy

    def test_get_quote_irs(self):
        inst = DummyIRSSpec(date(2025, 1, 1))
        curve = DiscountCurve(
            "test",
            date(2024, 1, 1),
            [date(2024, 1, 1), date(2025, 1, 1)],
            [1.0, 0.99],
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
            DayCounterType.ThirtyU360,
        )
        result = get_quote(
            ref_date=date(2024, 1, 1),
            instrument_spec=inst,
            curve_dict={"discount_curve": curve, "fixing_curve": curve},
        )
        self.assertEqual(result, 0.03)  # TODO input expected calculated value, 0.03 is dummy


class TestBootstrapCurveIntegration(unittest.TestCase):
    """Test for various combination of instruments supplied to the bootstrapper

    Args:
        unittest (_type_): _description_
    """

    def setUp(self):
        self.ref_date = date(2024, 1, 1)
        self.curve_id = "EUR_DISC"
        self.day_count = DayCounterType.ThirtyU360
        self.interp = InterpolationType.LINEAR
        self.extrap = ExtrapolationType.LINEAR


#     def test_deposit_bootstrap(self):
#         # Minimal deposit: 6M, 2% rate
#         end_date = self.ref_date + timedelta(days=182)
#         deposit = DepositSpecification(
#             obj_id="dep1",
#             notional=1_000_000,
#             start_date=self.ref_date,
#             end_date=end_date,
#             currency="EUR",
#             day_count_convention=self.day_count,
#             rate=0.02,
#         )
#         curve = bootstrap_curve(
#             ref_date=self.ref_date,
#             curve_id=self.curve_id,
#             day_count_convention=self.day_count,
#             instruments=[deposit],
#             quotes=[0.02],
#             interpolation_type=self.interp,
#             extrapolation_type=self.extrap,
#         )
#         self.assertIsInstance(curve, DiscountCurve)
#         self.assertEqual(curve.dates[0], self.ref_date)
#         self.assertEqual(curve.dates[1], end_date)
#         implied_rate = curve.get_zero_rate(end_date)
#         self.assertAlmostEqual(implied_rate, 0.02, places=4)

#     def test_fra_bootstrap(self):
#         # Minimal FRA: 6M-12M, 2.5% rate
#         start_date = self.ref_date + timedelta(days=182)
#         end_date = self.ref_date + timedelta(days=365)
#         fra = ForwardRateAgreementSpecification(
#             obj_id="fra1",
#             notional=1_000_000,
#             start_date=start_date,
#             end_date=end_date,
#             currency="EUR",
#             day_count_convention=self.day_count,
#             rate=0.025,
#         )
#         # Need a deposit for the first period to anchor the curve
#         deposit = DepositSpecification(
#             obj_id="dep1",
#             notional=1_000_000,
#             start_date=self.ref_date,
#             end_date=start_date,
#             currency="EUR",
#             day_count_convention=self.day_count,
#             rate=0.02,
#         )
#         curve = bootstrap_curve(
#             ref_date=self.ref_date,
#             curve_id=self.curve_id,
#             day_count_convention=self.day_count,
#             instruments=[deposit, fra],
#             quotes=[0.02, 0.025],
#             interpolation_type=self.interp,
#             extrapolation_type=self.extrap,
#         )
#         self.assertIsInstance(curve, DiscountCurve)
#         self.assertEqual(curve.dates[0], self.ref_date)
#         self.assertEqual(curve.dates[1], start_date)
#         self.assertEqual(curve.dates[2], end_date)

#     def test_irs_bootstrap(self):
#         # Minimal IRS: 1Y, fixed 3%, float 6M reset
#         start_date = self.ref_date
#         end_date = self.ref_date + timedelta(days=365)
#         pay_dates = [end_date]
#         fixed_leg = IrFixedLegSpecification(
#             fixed_rate=0.03,
#             obj_id="fixed_leg",
#             notional=1_000_000,
#             start_dates=[start_date],
#             end_dates=[end_date],
#             pay_dates=pay_dates,
#             currency="EUR",
#             day_count_convention=self.day_count,
#         )
#         float_leg = IrFloatLegSpecification(
#             obj_id="float_leg",
#             notional=1_000_000,
#             reset_dates=[start_date],
#             start_dates=[start_date],
#             end_dates=[end_date],
#             rate_start_dates=[start_date],
#             rate_end_dates=[end_date],
#             pay_dates=pay_dates,
#             currency="EUR",
#             udl_id="EURIBOR6M",
#             fixing_id="EURIBOR6M",
#             day_count_convention=self.day_count,
#             rate_day_count_convention=self.day_count,
#             spread=0.0,
#         )
#         irs = InterestRateSwapSpecification(
#             obj_id="swap1",
#             notional=1_000_000,
#             issue_date=start_date,
#             maturity_date=end_date,
#             pay_leg=fixed_leg,
#             receive_leg=float_leg,
#             currency="EUR",
#             day_count_convention=self.day_count,
#         )
#         # Need a deposit to anchor the curve
#         deposit = DepositSpecification(
#             obj_id="dep1",
#             notional=1_000_000,
#             start_date=self.ref_date,
#             end_date=self.ref_date + timedelta(days=182),
#             currency="EUR",
#             day_count_convention=self.day_count,
#             rate=0.02,
#         )
#         curve = bootstrap_curve(
#             ref_date=self.ref_date,
#             curve_id=self.curve_id,
#             day_count_convention=self.day_count,
#             instruments=[deposit, irs],
#             quotes=[0.02, 0.03],
#             interpolation_type=self.interp,
#             extrapolation_type=self.extrap,
#         )
#         self.assertIsInstance(curve, DiscountCurve)
#         self.assertIn(end_date, curve.dates)

#     def test_deposit_fra_irs_combination(self):
#         # Deposit, FRA, IRS in sequence
#         d1 = self.ref_date + timedelta(days=182)
#         d2 = self.ref_date + timedelta(days=365)
#         d3 = self.ref_date + timedelta(days=730)
#         deposit = DepositSpecification(
#             obj_id="dep1",
#             notional=1_000_000,
#             start_date=self.ref_date,
#             end_date=d1,
#             currency="EUR",
#             day_count_convention=self.day_count,
#             rate=0.02,
#         )
#         fra = ForwardRateAgreementSpecification(
#             obj_id="fra1",
#             notional=1_000_000,
#             start_date=d1,
#             end_date=d2,
#             currency="EUR",
#             day_count_convention=self.day_count,
#             rate=0.025,
#         )
#         # IRS 1Y from d2 to d3
#         pay_dates = [d3]
#         fixed_leg = IrFixedLegSpecification(
#             fixed_rate=0.03,
#             obj_id="fixed_leg",
#             notional=1_000_000,
#             start_dates=[d2],
#             end_dates=[d3],
#             pay_dates=pay_dates,
#             currency="EUR",
#             day_count_convention=self.day_count,
#         )
#         float_leg = IrFloatLegSpecification(
#             obj_id="float_leg",
#             notional=1_000_000,
#             reset_dates=[d2],
#             start_dates=[d2],
#             end_dates=[d3],
#             rate_start_dates=[d2],
#             rate_end_dates=[d3],
#             pay_dates=pay_dates,
#             currency="EUR",
#             udl_id="EURIBOR6M",
#             fixing_id="EURIBOR6M",
#             day_count_convention=self.day_count,
#             rate_day_count_convention=self.day_count,
#             spread=0.0,
#         )
#         irs = InterestRateSwapSpecification(
#             obj_id="swap1",
#             notional=1_000_000,
#             issue_date=d2,
#             maturity_date=d3,
#             pay_leg=fixed_leg,
#             receive_leg=float_leg,
#             currency="EUR",
#             day_count_convention=self.day_count,
#         )
#         curve = bootstrap_curve(
#             ref_date=self.ref_date,
#             curve_id=self.curve_id,
#             day_count_convention=self.day_count,
#             instruments=[deposit, fra, irs],
#             quotes=[0.02, 0.025, 0.03],
#             interpolation_type=self.interp,
#             extrapolation_type=self.extrap,
#         )
#         self.assertIsInstance(curve, DiscountCurve)
#         self.assertIn(d1, curve.dates)
#         self.assertIn(d2, curve.dates)
#         self.assertIn(d3, curve.dates)

#     def test_multicurve_bootstrap(self):
#         # Deposit for discount, FRA for forward
#         d1 = self.ref_date + timedelta(days=182)
#         d2 = self.ref_date + timedelta(days=365)
#         deposit = DepositSpecification(
#             obj_id="dep1",
#             notional=1_000_000,
#             start_date=self.ref_date,
#             end_date=d1,
#             currency="EUR",
#             day_count_convention=self.day_count,
#             rate=0.02,
#         )
#         fra = ForwardRateAgreementSpecification(
#             obj_id="fra1",
#             notional=1_000_000,
#             start_date=d1,
#             end_date=d2,
#             currency="EUR",
#             day_count_convention=self.day_count,
#             rate=0.025,
#         )
#         # First, bootstrap discount curve
#         discount_curve = bootstrap_curve(
#             ref_date=self.ref_date,
#             curve_id="DISC",
#             day_count_convention=self.day_count,
#             instruments=[deposit],
#             quotes=[0.02],
#             interpolation_type=self.interp,
#             extrapolation_type=self.extrap,
#         )
#         # Now, bootstrap forward curve using FRA and discount curve
#         curves = {"discount_curve": discount_curve}
#         forward_curve = bootstrap_curve(
#             ref_date=self.ref_date,
#             curve_id="FWD",
#             day_count_convention=self.day_count,
#             instruments=[fra],
#             quotes=[0.025],
#             curves=curves,
#             interpolation_type=self.interp,
#             extrapolation_type=self.extrap,
#         )
#         self.assertIsInstance(forward_curve, DiscountCurve)
#         self.assertIn(d2, forward_curve.dates)

#     def test_multicurve_with_deposit_raises(self):
#         # Deposit in multicurve should raise
#         d1 = self.ref_date + timedelta(days=182)
#         deposit = DepositSpecification(
#             obj_id="dep1",
#             notional=1_000_000,
#             start_date=self.ref_date,
#             end_date=d1,
#             currency="EUR",
#             day_count_convention=self.day_count,
#             rate=0.02,
#         )
#         discount_curve = bootstrap_curve(
#             ref_date=self.ref_date,
#             curve_id="DISC",
#             day_count_convention=self.day_count,
#             instruments=[deposit],
#             quotes=[0.02],
#             interpolation_type=self.interp,
#             extrapolation_type=self.extrap,
#         )
#         curves = {"discount_curve": discount_curve}
#         with self.assertRaises(Exception) as cm:
#             bootstrap_curve(
#                 ref_date=self.ref_date,
#                 curve_id="FWD",
#                 day_count_convention=self.day_count,
#                 instruments=[deposit],
#                 quotes=[0.02],
#                 curves=curves,
#                 interpolation_type=self.interp,
#                 extrapolation_type=self.extrap,
#             )
#         self.assertIn("Deposits cannot be used in multicurve bootstrapping", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
