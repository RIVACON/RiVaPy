# 2025.09.09 Bootstrapping without pyvacon
import unittest
import sys

from tests.setup_logging import setup_logging_for_tests

# Configure logging once per test module
setup_logging_for_tests("tests/rivapy_test.log")
import logging

logger = logging.getLogger("rivapy.tests.test_bootstrap")

import math
import pandas as pd
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

from rivapy.marketdata.bootstrapping import (
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
    IrOISLegSpecification,
    InterestRateBasisSwapSpecification,
)
from rivapy.tools.enums import DayCounterType, InterpolationType, ExtrapolationType, Instrument
from rivapy.instruments.components import ConstNotionalStructure
from rivapy.tools.datetools import DayCounter, Period, Schedule, calc_end_day, calc_start_day


# for specification from file tests
import rivapy.instruments.specification_from_csv as sfc
from holidays import HolidayBase as _HolidayBase
from holidays import EuropeanCentralBank as _ECB


# Helper functions
def tolerance_from_quote(q: float) -> float:
    """
    Determine an appropriate delta for assertAlmostEqual
    based on the number of decimals in the quote.
    """
    s = format(q, "f").rstrip("0").rstrip(".")
    decimals = len(s.split(".")[1]) if "." in s else 0
    delta = 0.5 * 10 ** (-decimals)
    return delta


def deep_equal_OLD(obj1, obj2):
    if type(obj1) != type(obj2):
        return False
    if hasattr(obj1, "__dict__") and hasattr(obj2, "__dict__"):
        return all(deep_equal(v, obj2.__dict__[k]) for k, v in obj1.__dict__.items())
    if isinstance(obj1, (list, tuple)):
        return all(deep_equal(x, y) for x, y in zip(obj1, obj2))
    return obj1 == obj2


def deep_equal(obj1, obj2, path="root"):
    """Recursively compare two objects and print where they differ."""
    if type(obj1) != type(obj2):
        print(f"Type mismatch at {path}: {type(obj1)} != {type(obj2)}")
        return False

    # handle objects with __dict__ (custom classes)
    if hasattr(obj1, "__dict__") and hasattr(obj2, "__dict__"):
        all_equal = True
        keys1, keys2 = set(obj1.__dict__.keys()), set(obj2.__dict__.keys())

        for key in keys1 | keys2:
            if key not in obj1.__dict__:
                print(f"Missing key {path}.{key} in obj1")
                all_equal = False
                continue
            if key not in obj2.__dict__:
                print(f"Missing key {path}.{key} in obj2")
                all_equal = False
                continue

            if not deep_equal(obj1.__dict__[key], obj2.__dict__[key], f"{path}.{key}"):
                all_equal = False
        return all_equal

    # handle lists and tuples
    if isinstance(obj1, (list, tuple)):
        all_equal = True
        for i, (x, y) in enumerate(zip(obj1, obj2)):
            if not deep_equal(x, y, f"{path}[{i}]"):
                all_equal = False
        if len(obj1) != len(obj2):
            print(f"Length mismatch at {path}: {len(obj1)} != {len(obj2)}")
            all_equal = False
        return all_equal

    # base case: primitive comparison
    if obj1 != obj2:
        print(f"Value mismatch at {path}: {obj1} != {obj2}")
        return False

    return True


# Minimal instrument specification classes for testing
class DummyDepositSpec(DepositSpecification):
    def __init__(self, maturity_date=None, issue_date=None, ref_date=None):
        """Setting up base deposit specification for tests.
        For now, as O/N deposit with 1 day accrual.
        """
        ##########################################
        # setting up deposit
        # calculation date
        if ref_date is None:
            ref_date = datetime(2019, 8, 31)

        # start date of the accrual period with spot lag equal to 2 days
        if issue_date is None:
            issue_date = ref_date + timedelta(days=2)

        # end date of the accrual period is 1 day after startdate
        if maturity_date is None:
            maturity_date = issue_date + timedelta(days=1)

        super().__init__(
            obj_id="dummy_deposit",
            issuer="dummy_issuer",
            currency="EUR",
            issue_date=issue_date,
            maturity_date=maturity_date,
            notional=100.0,
            rate=0.01,
            day_count_convention="Act360",
        )


class TestBootstrapCurveFunctions(unittest.TestCase):

    def test_input_length_assertion(self):
        """Test that providing different lengths of instruments and quotes raises an AssertionError."""
        with self.assertRaises(AssertionError):
            bootstrap_curve(
                ref_date=date(2024, 1, 1),
                curve_id="curve1",
                day_count_convention=DayCounterType.ThirtyU360,
                instruments=[DummyDepositSpec(maturity_date=datetime(2025, 1, 1))],
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
        inst = DummyDepositSpec(ref_date=datetime(2024, 1, 1), issue_date=datetime(2024, 1, 1), maturity_date=datetime(2024, 1, 3))
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

        ARGS = ()
        lower, upper = find_bracket(fake_error_fn, 1.5, ARGS)
        self.assertLess(lower, 1)
        self.assertGreater(upper, 1)

    def test_find_bracket_failure(self):
        def fake_error_fn(x, *args):
            return 1

        ARGS = ()
        with self.assertRaises(RuntimeError):
            find_bracket(fake_error_fn, 2, ARGS)


class TestGetQuote(unittest.TestCase):
    """Testing get_quote for different instrument types

    Args:
        unittest (_type_): _description_
    """

    def test_get_quote_deposit(self):
        refdate = datetime(2024, 1, 1)
        inst = DummyDepositSpec(ref_date=refdate, issue_date=refdate, maturity_date=datetime(2024, 1, 3))
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
        # self.assertEqual(result, 0.01)
        self.assertAlmostEqual(result, 0.024508664452316253, delta=1e-8)

    def test_get_quote_fra(self):

        ref_date = datetime(2023, 1, 28)
        start_date = datetime(2023, 7, 28)  # 6mx3m
        end_date = datetime(2023, 10, 28)
        inst = ForwardRateAgreementSpecification(
            obj_id="dummy_id",
            trade_date=ref_date,
            # maturity_date=mat_date,
            notional=1000.0,
            rate=0.04,
            start_date=start_date,
            end_date=end_date,
            udlID="dummy_underlying_index",
            rate_start_date=start_date,
            rate_end_date=end_date,
            day_count_convention="Act360",
            rate_day_count_convention="Act360",
            currency="EUR",
            spot_days=1,
            payment_days=1,
            issuer="dummy_issuer",
            securitization_level="NONE",
        )

        # fwd curve
        object_id = "TEST_fwd"
        fwd_rate = 0.05
        days_to_maturity = [1, 180, 365, 720, 3 * 365, 4 * 365, 10 * 365]
        dates = [ref_date + timedelta(days=d) for d in days_to_maturity]
        fwd_df = [math.exp(-d / 365.0 * fwd_rate) for d in days_to_maturity]
        fwd_dc = DiscountCurve(
            id=object_id,
            refdate=ref_date,
            dates=dates,
            df=fwd_df,
            interpolation=InterpolationType.LINEAR,
            extrapolation=ExtrapolationType.LINEAR,
            daycounter=DayCounterType.ACT360,
        )

        # note that since we are getting to the FRA pricing through the 'get_quoute' function which
        # is in the framework of curve bootstrapping a single curve, where the discounting,
        # and forecast of forward is done with the same curve and built into the usage of FRAs in
        # context of single curve bootstrapping here
        # proper unit tests of the pricing or fair rate computation of FRAs is to be done in the FRAs section.
        result = get_quote(
            ref_date=ref_date,
            instrument_spec=inst,
            curve_dict={"discount_curve": fwd_dc},
        )
        self.assertAlmostEqual(result, 0.049315806932066504, delta=1e-8)

    def test_get_quote_irs(self):

        # 1Y maturity with quartlery payment
        ref_date = datetime(2019, 8, 31)
        start_dates = [ref_date + relativedelta(months=3 * i) for i in range(4)]

        # reset dates are equal to start dates if spot lag is 0.
        reset_dates = start_dates

        # the end dates of the accral periods
        end_dates = [x + relativedelta(months=3) for x in start_dates]
        pay_dates = end_dates
        ns = ConstNotionalStructure(100.0)
        spread = 0.00

        # float leg spec
        float_leg = IrFloatLegSpecification(
            obj_id="dummy_float_leg",
            notional=ns,
            reset_dates=reset_dates,
            start_dates=start_dates,
            end_dates=end_dates,
            rate_start_dates=start_dates,
            rate_end_dates=end_dates,
            pay_dates=pay_dates,
            currency="EUR",
            udl_id="test_udl_id",
            fixing_id="test_fixing_id",
            day_count_convention="Act365Fixed",
            spread=spread,
        )

        # # definition of the fixed leg
        fixed_leg = IrFixedLegSpecification(
            fixed_rate=0.01,
            obj_id="dummy_fixed_leg",
            notional=100.0,
            start_dates=start_dates,
            end_dates=end_dates,
            pay_dates=pay_dates,
            currency="EUR",
            day_count_convention="Act365Fixed",
        )

        # # definition of the IR swap
        ir_swap = InterestRateSwapSpecification(
            obj_id="3M_SWAP",
            notional=ns,
            issue_date=ref_date,
            maturity_date=pay_dates[-1],
            pay_leg=fixed_leg,
            receive_leg=float_leg,
            currency="EUR",
            day_count_convention="Act365Fixed",
            issuer="dummy_issuer",
            securitization_level="COLLATERALIZED",
        )

        curve = DiscountCurve(
            "test",
            ref_date,
            [ref_date, ref_date + relativedelta(years=1)],
            [1.0, 0.99],
            InterpolationType.LINEAR,
            ExtrapolationType.LINEAR,
            DayCounterType.ThirtyU360,
        )
        result = get_quote(
            ref_date=ref_date,
            instrument_spec=ir_swap,
            curve_dict={"discount_curve": curve, "fixing_curve": curve},
        )  # in the context of bootstrapping a single curve, we make the assumption that discount curve and fixing curve are the same
        # in case of irswap- compute swap rate implies calculating the annuity =>
        print(result)
        self.assertAlmostEqual(result, 0.010090861477238481, delta=1e-8)


class TestBootstrapCurveInstruments(unittest.TestCase):
    """Test for various combination of instruments supplied to the bootstrapper

    Args:
        unittest (_type_): _description_
    """

    def setUp(self):
        self.ref_date = datetime(2019, 8, 31)
        self.curve_id = "EUR_DISC"
        self.day_count = DayCounterType.Act365Fixed
        self.interp = InterpolationType.LINEAR
        self.extrap = ExtrapolationType.LINEAR

    def test_deposit_bootstrap(self):
        # Minimal deposit: 6M, 2% rate
        logger.debug("Creating 1 deposit instrument")
        issue_date = self.ref_date + timedelta(days=2)  # spot lag of 2 days
        maturity_date = issue_date + timedelta(days=1)  # 1 day after startdate
        deposit = DepositSpecification(
            obj_id="dep1",
            notional=100,
            issue_date=issue_date,
            maturity_date=maturity_date,
            currency="EUR",
            day_count_convention=self.day_count,
            rate=0.01,  # rate different from "market quote" to ensure that rate here is NOT used in deposit bootstrapping
        )
        logger.debug("Running single curve bootstrapping")
        curve = bootstrap_curve(
            ref_date=self.ref_date,
            curve_id=self.curve_id,
            day_count_convention=self.day_count,
            instruments=[deposit],
            quotes=[0.025],
            interpolation_type=self.interp,
            extrapolation_type=self.extrap,
        )
        # print(curve.get_dates())
        logger.debug("Checking assertions")
        self.assertIsInstance(curve, DiscountCurve)
        self.assertEqual(curve.get_dates()[0], self.ref_date)
        self.assertEqual(curve.get_dates()[1], maturity_date)

        # the discount curve needs to be able to get the same market quote for the instrument
        model_quote = get_quote(self.ref_date, deposit, {"discount_curve": curve})
        self.assertAlmostEqual(model_quote, 0.025, delta=1e-8)  # not good enough, what is going on? not enough data points?
        logger.debug("TestBootstrapCurveInstruments.test_deposit_bootstrap completed")

    def test_fra_bootstrap(self):
        # Minimal FRA: 6Mx3M, 2.5% rate
        # start_date = self.ref_date + relativedelta(months=6)
        # end_date = self.ref_date + relativedelta(months=9)
        ref_date = datetime(2023, 1, 28)
        issue_date = datetime(2023, 7, 28)
        maturity_date = datetime(2023, 10, 28)
        fra = ForwardRateAgreementSpecification(
            obj_id="dummy_id",
            trade_date=ref_date,
            # maturity_date=mat_date,
            notional=1000.0,
            rate=0.025,
            start_date=issue_date,
            end_date=maturity_date,
            udlID="dummy_underlying_index",
            rate_start_date=issue_date,
            rate_end_date=maturity_date,
            day_count_convention=self.day_count,
            rate_day_count_convention=self.day_count,
            currency="EUR",
            spot_days=1,
            payment_days=1,
            issuer="dummy_issuer",
            securitization_level="NONE",
        )

        # Need a deposit for the first period to anchor the curve # i.e. in front of the FRA
        deposit = DepositSpecification(
            obj_id="dep1",
            notional=1000.0,
            issue_date=ref_date,
            maturity_date=issue_date,
            currency="EUR",
            day_count_convention=self.day_count,
            rate=0.02,
        )

        curve = bootstrap_curve(
            ref_date=self.ref_date,
            curve_id=self.curve_id,
            day_count_convention=self.day_count,
            instruments=[deposit, fra],
            quotes=[0.02, 0.025],
            interpolation_type=self.interp,
            extrapolation_type=self.extrap,
        )

        self.assertIsInstance(curve, DiscountCurve)
        self.assertEqual(curve.get_dates()[0], self.ref_date)
        self.assertEqual(curve.get_dates()[1], issue_date)
        self.assertEqual(curve.get_dates()[2], maturity_date)

    def test_irs_bootstrap(self):
        # test that at least, the bootstrap completes
        # Minimal IRS: 1Y, fixed 3%, float 6M reset
        start_date = self.ref_date
        end_date = self.ref_date + timedelta(days=365)
        pay_dates = [end_date]
        ns = ConstNotionalStructure(100.0)
        fixed_leg = IrFixedLegSpecification(
            fixed_rate=0.03,
            obj_id="fixed_leg",
            notional=ns,
            start_dates=[start_date],
            end_dates=[end_date],
            pay_dates=pay_dates,
            currency="EUR",
            day_count_convention=self.day_count,
        )
        float_leg = IrFloatLegSpecification(
            obj_id="float_leg",
            notional=ns,
            reset_dates=[start_date],
            start_dates=[start_date],
            end_dates=[end_date],
            rate_start_dates=[start_date],
            rate_end_dates=[end_date],
            pay_dates=pay_dates,
            currency="EUR",
            udl_id="EURIBOR6M",
            fixing_id="EURIBOR6M",
            day_count_convention=self.day_count,
            rate_day_count_convention=self.day_count,
            spread=0.0,
        )
        irs = InterestRateSwapSpecification(
            obj_id="swap1",
            notional=ns,
            issue_date=start_date,
            maturity_date=end_date,
            pay_leg=fixed_leg,
            receive_leg=float_leg,
            currency="EUR",
            day_count_convention=self.day_count,
        )
        # Need a deposit to anchor the curve
        deposit = DepositSpecification(
            obj_id="dep1",
            notional=1_000_000,
            issue_date=self.ref_date,
            maturity_date=self.ref_date + timedelta(days=182),
            currency="EUR",
            day_count_convention=self.day_count,
            rate=0.02,
        )
        curve = bootstrap_curve(
            ref_date=self.ref_date,
            curve_id=self.curve_id,
            day_count_convention=self.day_count,
            instruments=[deposit, irs],
            quotes=[0.02, 0.03],
            interpolation_type=self.interp,
            extrapolation_type=self.extrap,
        )
        self.assertIsInstance(curve, DiscountCurve)
        self.assertIn(end_date, curve.get_dates())

        # test that it reproduces the model quotes
        model_quotes = []
        # pricing_params = {"fixing_grace_period": 0.0, "set_rate": True, "desired_rate": 1.0}
        curves_dict = {"discount_curve": curve, "fixing_curve": curve}

        for i in range(len([deposit, irs])):

            model_quote = get_quote(self.ref_date, [deposit, irs][i], curves_dict)
            # print(model_quote)
            model_quotes.append(model_quote)

        self.assertAlmostEqual(model_quotes[0], 0.02, delta=1e-8)
        self.assertAlmostEqual(model_quotes[1], 0.03, delta=1e-8)

    def test_deposit_fra_irs_combination(self):
        # Deposit, FRA, IRS in sequence
        # d1 = self.ref_date + timedelta(days=182)
        # d2 = self.ref_date + timedelta(days=365)
        # d3 = self.ref_date + timedelta(days=730)

        ref_date = datetime(2023, 1, 28)
        # start_date = datetime(2023, 7, 28)
        # end_date = datetime(2023, 10, 28)

        d1 = datetime(2023, 7, 28)
        d2 = datetime(2023, 10, 28)
        d3 = datetime(2024, 10, 28)

        deposit = DepositSpecification(
            obj_id="dep1",
            notional=1_000_000,
            issue_date=ref_date,
            maturity_date=d1,
            currency="EUR",
            day_count_convention=self.day_count,
            rate=0.02,
        )
        fra = fra = ForwardRateAgreementSpecification(
            obj_id="dummy_id",
            trade_date=ref_date,
            # maturity_date=mat_date,
            notional=1_000_000.0,
            rate=0.025,
            start_date=d1,
            end_date=d2,
            udlID="dummy_underlying_index",
            rate_start_date=d1,
            rate_end_date=d2,
            day_count_convention=self.day_count,
            rate_day_count_convention=self.day_count,
            currency="EUR",
            spot_days=1,
            payment_days=1,
            issuer="dummy_issuer",
            securitization_level="NONE",
        )
        # IRS 1Y from d2 to d3
        ns = ConstNotionalStructure(1_000_000.0)
        pay_dates = [d3]
        fixed_leg = IrFixedLegSpecification(
            fixed_rate=0.03,
            obj_id="fixed_leg",
            notional=ns,
            start_dates=[d2],
            end_dates=[d3],
            pay_dates=pay_dates,
            currency="EUR",
            day_count_convention=self.day_count,
        )
        float_leg = IrFloatLegSpecification(
            obj_id="float_leg",
            notional=ns,
            reset_dates=[d2],
            start_dates=[d2],
            end_dates=[d3],
            rate_start_dates=[d2],
            rate_end_dates=[d3],
            pay_dates=pay_dates,
            currency="EUR",
            udl_id="EURIBOR6M",
            fixing_id="EURIBOR6M",
            day_count_convention=self.day_count,
            rate_day_count_convention=self.day_count,
            spread=0.0,
        )
        irs = InterestRateSwapSpecification(
            obj_id="swap1",
            notional=ns,
            issue_date=d2,
            maturity_date=d3,
            pay_leg=fixed_leg,
            receive_leg=float_leg,
            currency="EUR",
            day_count_convention=self.day_count,
        )

        instruments = [deposit, fra, irs]
        quotes = [0.02, 0.025, 0.03]
        curve = bootstrap_curve(
            ref_date=ref_date,
            curve_id=self.curve_id,
            day_count_convention=self.day_count,
            instruments=instruments,
            quotes=quotes,
            interpolation_type=self.interp,
            extrapolation_type=self.extrap,
        )
        self.assertIsInstance(curve, DiscountCurve)
        self.assertIn(d1, curve.get_dates())
        self.assertIn(d2, curve.get_dates())
        self.assertIn(d3, curve.get_dates())

        # test that it reproduces the model quotes
        model_quotes = []
        # pricing_params = {"fixing_grace_period": 0.0, "set_rate": True, "desired_rate": 1.0}
        curves_dict = {"discount_curve": curve, "fixing_curve": curve}

        for i in range(len(instruments)):

            model_quote = get_quote(ref_date, instruments[i], curves_dict)
            # print(model_quote)
            model_quotes.append(model_quote)

        self.assertAlmostEqual(model_quotes[0], quotes[0], delta=1e-8)
        self.assertAlmostEqual(model_quotes[1], quotes[1], delta=1e-8)
        self.assertAlmostEqual(model_quotes[2], quotes[2], delta=1e-8)

    def test_many_instruments(self):
        """Case where lots of instruments are given"""
        # calculation date
        ref_date = datetime(2019, 8, 31)  # refdate = dt.datetime(2017, 8, 31)
        # start date of the accrual period with spot lag equal to 2 days
        start_date = ref_date + timedelta(days=2)

        end_date_deposits = [
            start_date + timedelta(days=1),
            start_date + timedelta(days=7),
            start_date + timedelta(days=30),
            start_date + timedelta(days=60),
            start_date + timedelta(days=90),
            start_date + timedelta(days=181),  # 180 - error due to BCC roll oveer mismatch between start and end date... look into! #TODO
            start_date + timedelta(days=270),
        ]

        # Provide a list of market quotes from which to derive the discount curve
        quotes_deposits = [0.025, 0.028, 0.0283, 0.029, 0.0305, 0.0315, 0.0348]

        # List where the multiple deposit specfications are stored
        multiple_deposits = []

        for i in range(len(quotes_deposits)):
            print(i)
            temp_deposit = DepositSpecification(
                obj_id="DEPOSIT_" + str(i + 1),
                issuer="dummy_issuer",
                currency="EUR",
                issue_date=start_date,
                maturity_date=end_date_deposits[i],
                notional=100.0,
                rate=quotes_deposits[i],
                day_count_convention="Act365Fixed",
            )
            multiple_deposits.append(temp_deposit)

        # setting up swaps
        start_dates = [ref_date + relativedelta(months=3 * i) for i in range(4)]
        # reset dates are equal to start dates if spot lag is 0.
        reset_dates = start_dates
        # the end dates of the accral periods
        end_dates = [x + relativedelta(months=3) for x in start_dates]
        # the actual payment dates of the cashflows may differ from the end of the accrual period (e.g. OIS).
        # in the standard case these two sets of dates coincide
        pay_dates = end_dates

        ns = ConstNotionalStructure(100.0)
        spread = 0.00
        # # definition of the floating leg
        float_leg = IrFloatLegSpecification(
            obj_id="dummy_float_leg",
            notional=ns,
            reset_dates=reset_dates,
            start_dates=start_dates,
            end_dates=end_dates,
            rate_start_dates=start_dates,
            rate_end_dates=end_dates,
            pay_dates=pay_dates,
            currency="EUR",
            udl_id="test_udl_id",
            fixing_id="test_fixing_id",
            day_count_convention="Act365Fixed",
            spread=spread,
        )

        # # definition of the fixed leg
        # Note that a fixed rate is given for the specification as it is required.
        # However, for the creation of the bootrstrapped curve, the market quotes are used as the target swap par rate
        fixed_leg = IrFixedLegSpecification(
            fixed_rate=0.01,
            obj_id="dummy_fixed_leg",
            notional=100.0,
            start_dates=start_dates,
            end_dates=end_dates,
            pay_dates=pay_dates,
            currency="EUR",
            day_count_convention="Act365Fixed",
        )
        # # definition of the IR swap
        ir_swap = InterestRateSwapSpecification(
            obj_id="3M_SWAP",
            notional=ns,
            issue_date=ref_date,
            maturity_date=pay_dates[-1],
            pay_leg=fixed_leg,
            receive_leg=float_leg,
            currency="EUR",
            day_count_convention="Act365Fixed",
            issuer="dummy_issuer",
            securitization_level="COLLATERALIZED",
        )

        # 2Y maturity 3M swap
        start_dates2 = [ref_date + relativedelta(months=3 * i) for i in range(4 * 2)]
        reset_dates2 = start_dates2
        end_dates2 = [x + relativedelta(months=3) for x in start_dates2]
        pay_dates2 = end_dates2

        ns = ConstNotionalStructure(100.0)
        spread = 0.00

        # # definition of the floating leg
        float_leg2 = IrFloatLegSpecification(
            obj_id="dummy_float_leg2",
            notional=ns,
            reset_dates=reset_dates2,
            start_dates=start_dates2,
            end_dates=end_dates2,
            rate_start_dates=start_dates2,
            rate_end_dates=end_dates2,
            pay_dates=pay_dates2,
            currency="EUR",
            udl_id="test_udl_id",
            fixing_id="test_fixing_id",
            day_count_convention="Act365Fixed",
            spread=spread,
        )

        # # definition of the fixed leg
        fixed_leg2 = IrFixedLegSpecification(
            fixed_rate=0.01,
            obj_id="dummy_fixed_leg2",
            notional=100.0,
            start_dates=start_dates2,
            end_dates=end_dates2,
            pay_dates=pay_dates2,
            currency="EUR",
            day_count_convention="Act365Fixed",
        )

        # # definition of the IR swap
        ir_swap2 = InterestRateSwapSpecification(
            obj_id="3M_SWAP2",
            notional=ns,
            issue_date=ref_date,
            maturity_date=pay_dates2[-1],
            pay_leg=fixed_leg2,
            receive_leg=float_leg2,
            currency="EUR",
            day_count_convention="Act365Fixed",
            issuer="dummy_issuer",
            securitization_level="COLLATERALIZED",
        )

        # 3Y maturity 3M swap

        start_dates3 = [ref_date + relativedelta(months=3 * i) for i in range(4 * 3)]
        reset_dates3 = start_dates3
        end_dates3 = [x + relativedelta(months=3) for x in start_dates3]
        pay_dates3 = end_dates3
        ns = ConstNotionalStructure(100.0)
        spread = 0.00

        # # definition of the floating leg
        float_leg3 = IrFloatLegSpecification(
            obj_id="dummy_float_leg3",
            notional=ns,
            reset_dates=reset_dates3,
            start_dates=start_dates3,
            end_dates=end_dates3,
            rate_start_dates=start_dates3,
            rate_end_dates=end_dates3,
            pay_dates=pay_dates3,
            currency="EUR",
            udl_id="test_udl_id",
            fixing_id="test_fixing_id",
            day_count_convention="Act365Fixed",
            spread=spread,
        )

        # # definition of the fixed leg
        fixed_leg3 = IrFixedLegSpecification(
            fixed_rate=0.01,
            obj_id="dummy_fixed_leg3",
            notional=100.0,
            start_dates=start_dates3,
            end_dates=end_dates3,
            pay_dates=pay_dates3,
            currency="EUR",
            day_count_convention="Act365Fixed",
        )

        # # definition of the IR swap
        ir_swap3 = InterestRateSwapSpecification(
            obj_id="3M_SWAP3",
            notional=ns,
            issue_date=ref_date,
            maturity_date=pay_dates3[-1],
            pay_leg=fixed_leg3,
            receive_leg=float_leg3,
            currency="EUR",
            day_count_convention="Act365Fixed",
            issuer="dummy_issuer",
            securitization_level="COLLATERALIZED",
        )

        # organising the swaps and their taarget market quotes
        multiple_swaps = [ir_swap, ir_swap2, ir_swap3]
        quotes_swaps = [0.05, 0.06, 0.07]

        instruments_both = multiple_deposits + multiple_swaps
        quotes_both = quotes_deposits + quotes_swaps

        boot_curve = bootstrap_curve(
            ref_date,
            "bootstrapped_DC",
            DayCounterType.Act365Fixed,
            instruments_both,
            quotes_both,
            interpolation_type=InterpolationType.LINEAR_LOG,
            extrapolation_type=ExtrapolationType.LINEAR_LOG,
        )

        model_quotes = []
        # pricing_params = {"fixing_grace_period": 0.0, "set_rate": True, "desired_rate": 1.0}
        curves_dict = {"discount_curve": boot_curve, "fixing_curve": boot_curve}

        for i in range(len(instruments_both)):
            model_quote = get_quote(ref_date, instruments_both[i], curves_dict)
            # print(model_quote)
            model_quotes.append(model_quote)

        for i in range(len(quotes_both)):

            self.assertAlmostEqual(model_quotes[i], quotes_both[i], delta=1e-8)

    def test_ois_multicurve_bootstrap(self):
        """Test for multicurve boot strapping.
        Create a discount curve based on 3M tenor, and create a discount curve from OIS instruments
        Create a forward curve based off this input curve and other instruments
        """
        ref_date = self.ref_date

        # setting up OIS
        ######################################################################
        # 1M Maturity, 1 M tenor

        start_dates = [ref_date + relativedelta(months=1 * i) for i in range(1)]

        # the end dates of the accrual periods
        end_dates = [x + relativedelta(months=1) for x in start_dates]

        # the actual payment dates of the cashflows may differ from the end of the accrual period (e.g. OIS).
        # in the standard case these two sets of dates coincide
        # pay_dates = end_dates

        print(end_dates)
        ns = ConstNotionalStructure(100.0)
        spread = 0.00

        # the difference here is that rate_date arrays are excpected to be 2 dimensional, i.e. keep track of the daily resetting per accrual period
        res = IrOISLegSpecification.ois_scheduler_2D(start_dates, end_dates)

        daily_rate_start_dates = res[0]  # 2D list: coupon i -> list of daily starts
        daily_rate_end_dates = res[1]  # 2D list: coupon i -> list of daily ends
        daily_rate_reset_dates = res[2]  # 2D list: coupon i -> list of reset dates
        pay_dates = res[3]

        float_leg = IrOISLegSpecification(
            obj_id="dummy_float_leg",
            notional=ns,
            rate_reset_dates=daily_rate_reset_dates,
            start_dates=start_dates,
            end_dates=end_dates,
            rate_start_dates=daily_rate_start_dates,
            rate_end_dates=daily_rate_end_dates,
            pay_dates=pay_dates,
            currency="EUR",
            udl_id="test_udl_id",
            fixing_id="test_fixing_id",
            day_count_convention="Act365Fixed",
            rate_day_count_convention="Act365Fixed",
            spread=spread,
        )

        # # definition of the fixed leg
        # Note that a fixed rate is given for the specification as it is required.
        # However, for the creation of the bootrstrapped curve, the market quotes are used as the target swap par rate
        fixed_leg = IrFixedLegSpecification(
            fixed_rate=0.01,
            obj_id="dummy_fixed_leg",
            notional=100.0,
            start_dates=start_dates,
            end_dates=end_dates,
            pay_dates=pay_dates,
            currency="EUR",
            day_count_convention="Act365Fixed",
        )

        # # definition of the IR swap
        ois_swap_1M = InterestRateSwapSpecification(
            obj_id="1M_SWAP",
            notional=ns,
            issue_date=ref_date,
            maturity_date=pay_dates[-1],
            pay_leg=fixed_leg,
            receive_leg=float_leg,
            currency="EUR",
            day_count_convention="Act365Fixed",
            issuer="dummy_issuer",
            securitization_level="COLLATERALIZED",
        )

        # 3M maturity 3M underlying tenor swap, i.e. the floating leg is reset every 3M
        # since it is OIS

        # start dates of the accrual periods corresponding to the tenor of the underlying index (3 months). The spot lag is set to 0.
        start_dates = [ref_date + relativedelta(months=3 * i) for i in range(1)]
        end_dates = [x + relativedelta(months=3) for x in start_dates]
        ns = ConstNotionalStructure(100.0)
        spread = 0.00
        res = IrOISLegSpecification.ois_scheduler_2D(start_dates, end_dates)

        daily_rate_start_dates = res[0]  # 2D list: coupon i -> list of daily starts
        daily_rate_end_dates = res[1]  # 2D list: coupon i -> list of daily ends
        daily_rate_reset_dates = res[2]  # 2D list: coupon i -> list of reset dates
        pay_dates = res[3]

        float_leg = IrOISLegSpecification(
            obj_id="dummy_float_leg",
            notional=ns,
            rate_reset_dates=daily_rate_reset_dates,
            start_dates=start_dates,
            end_dates=end_dates,
            rate_start_dates=daily_rate_start_dates,
            rate_end_dates=daily_rate_end_dates,
            pay_dates=pay_dates,
            currency="EUR",
            udl_id="test_udl_id",
            fixing_id="test_fixing_id",
            day_count_convention="Act365Fixed",
            rate_day_count_convention="Act365Fixed",
            spread=spread,
        )

        # # definition of the fixed leg
        # Note that a fixed rate is given for the specification as it is required.
        # However, for the creation of the bootrstrapped curve, the market quotes are used as the target swap par rate
        fixed_leg = IrFixedLegSpecification(
            fixed_rate=0.01,
            obj_id="dummy_fixed_leg",
            notional=100.0,
            start_dates=start_dates,
            end_dates=end_dates,
            pay_dates=pay_dates,
            currency="EUR",
            day_count_convention="Act365Fixed",
        )

        # # definition of the IR swap
        ois_swap_3M = InterestRateSwapSpecification(
            obj_id="3M_SWAP",
            notional=ns,
            issue_date=ref_date,
            maturity_date=pay_dates[-1],
            pay_leg=fixed_leg,
            receive_leg=float_leg,
            currency="EUR",
            day_count_convention="Act365Fixed",
            issuer="dummy_issuer",
            securitization_level="COLLATERALIZED",
        )

        # 6M Maturity, 6M tenor
        start_dates = [ref_date + relativedelta(months=6 * i) for i in range(1)]
        end_dates = [x + relativedelta(months=6) for x in start_dates]
        ns = ConstNotionalStructure(100.0)
        spread = 0.00
        res = IrOISLegSpecification.ois_scheduler_2D(start_dates, end_dates)

        daily_rate_start_dates = res[0]  # 2D list: coupon i -> list of daily starts
        daily_rate_end_dates = res[1]  # 2D list: coupon i -> list of daily ends
        daily_rate_reset_dates = res[2]  # 2D list: coupon i -> list of reset dates
        pay_dates = res[3]

        float_leg = IrOISLegSpecification(
            obj_id="dummy_float_leg",
            notional=ns,
            rate_reset_dates=daily_rate_reset_dates,
            start_dates=start_dates,
            end_dates=end_dates,
            rate_start_dates=daily_rate_start_dates,
            rate_end_dates=daily_rate_end_dates,
            pay_dates=pay_dates,
            currency="EUR",
            udl_id="test_udl_id",
            fixing_id="test_fixing_id",
            day_count_convention="Act365Fixed",
            rate_day_count_convention="Act365Fixed",
            spread=spread,
        )

        # # definition of the fixed leg

        fixed_leg = IrFixedLegSpecification(
            fixed_rate=0.01,
            obj_id="dummy_fixed_leg",
            notional=100.0,
            start_dates=start_dates,
            end_dates=end_dates,
            pay_dates=pay_dates,
            currency="EUR",
            day_count_convention="Act365Fixed",
        )

        # # definition of the IR swap
        ois_swap_6M = InterestRateSwapSpecification(
            obj_id="6M_SWAP",
            notional=ns,
            issue_date=ref_date,
            maturity_date=pay_dates[-1],
            pay_leg=fixed_leg,
            receive_leg=float_leg,
            currency="EUR",
            day_count_convention="Act365Fixed",
            issuer="dummy_issuer",
            securitization_level="COLLATERALIZED",
        )

        # 9M Maturity, 9M tenor
        start_dates = [ref_date + relativedelta(months=9 * i) for i in range(1)]
        end_dates = [x + relativedelta(months=9) for x in start_dates]
        ns = ConstNotionalStructure(100.0)
        spread = 0.00
        res = IrOISLegSpecification.ois_scheduler_2D(start_dates, end_dates)

        daily_rate_start_dates = res[0]  # 2D list: coupon i -> list of daily starts
        daily_rate_end_dates = res[1]  # 2D list: coupon i -> list of daily ends
        daily_rate_reset_dates = res[2]  # 2D list: coupon i -> list of reset dates
        pay_dates = res[3]

        float_leg = IrOISLegSpecification(
            obj_id="dummy_float_leg",
            notional=ns,
            rate_reset_dates=daily_rate_reset_dates,
            start_dates=start_dates,
            end_dates=end_dates,
            rate_start_dates=daily_rate_start_dates,
            rate_end_dates=daily_rate_end_dates,
            pay_dates=pay_dates,
            currency="EUR",
            udl_id="test_udl_id",
            fixing_id="test_fixing_id",
            day_count_convention="Act365Fixed",
            rate_day_count_convention="Act365Fixed",
            spread=spread,
        )

        # # definition of the fixed leg
        fixed_leg = IrFixedLegSpecification(
            fixed_rate=0.01,
            obj_id="dummy_fixed_leg",
            notional=100.0,
            start_dates=start_dates,
            end_dates=end_dates,
            pay_dates=pay_dates,
            currency="EUR",
            day_count_convention="Act365Fixed",
        )

        # # definition of the IR swap
        ois_swap_9M = InterestRateSwapSpecification(
            obj_id="39_SWAP",
            notional=ns,
            issue_date=ref_date,
            maturity_date=pay_dates[-1],
            pay_leg=fixed_leg,
            receive_leg=float_leg,
            currency="EUR",
            day_count_convention="Act365Fixed",
            issuer="dummy_issuer",
            securitization_level="COLLATERALIZED",
        )

        #####################################################
        # combine the multiple instruments to be given to the bootstrapper
        instruments_ois = [ois_swap_1M, ois_swap_3M, ois_swap_6M, ois_swap_9M]
        quotes_ois = [-0.00358, -0.00358, -0.00358, -0.00357]  # taken from the .csv as example

        # bootsrap OIS discount curve
        boot_curve_ois = bootstrap_curve(
            ref_date,
            "bootstrapped_ois_DC",
            DayCounterType.Act365Fixed,
            instruments_ois,
            quotes_ois,
            interpolation_type=InterpolationType.LINEAR_LOG,
            extrapolation_type=ExtrapolationType.LINEAR_LOG,
        )

        # setting up 3M tenor instruments
        ##########################################
        # 1Y IRS
        # 1Y maturity 3M swap, i.e. the floating leg is reset every 3M
        # start dates of the accrual periods corresponding to the tenor of the underlying index (3 months). The spot lag is set to 0.
        start_dates = [ref_date + relativedelta(months=3 * i) for i in range(4)]
        reset_dates = start_dates
        end_dates = [x + relativedelta(months=3) for x in start_dates]
        pay_dates = end_dates

        print(end_dates)
        ns = ConstNotionalStructure(100.0)
        spread = 0.00

        # # definition of the floating leg
        float_leg_1Y = IrFloatLegSpecification(
            obj_id="dummy_float_leg",
            notional=ns,
            reset_dates=reset_dates,
            start_dates=start_dates,
            end_dates=end_dates,
            rate_start_dates=start_dates,
            rate_end_dates=end_dates,
            pay_dates=pay_dates,
            currency="EUR",
            udl_id="test_udl_id",
            fixing_id="test_fixing_id",
            day_count_convention="Act365Fixed",
            spread=spread,
        )

        # # definition of the fixed leg
        fixed_leg_1Y = IrFixedLegSpecification(
            fixed_rate=0.01,
            obj_id="dummy_fixed_leg",
            notional=100.0,
            start_dates=start_dates,
            end_dates=end_dates,
            pay_dates=pay_dates,
            currency="EUR",
            day_count_convention="Act365Fixed",
        )

        # # definition of the IR swap
        irs_1Y = InterestRateSwapSpecification(
            obj_id="3M_SWAP_1Y",
            notional=ns,
            issue_date=ref_date,
            maturity_date=pay_dates[-1],
            pay_leg=fixed_leg_1Y,
            receive_leg=float_leg_1Y,
            currency="EUR",
            day_count_convention="Act365Fixed",
            issuer="dummy_issuer",
            securitization_level="COLLATERALIZED",
        )

        ##########################################
        # 2 Yr IRS
        # 2Y maturity 3M swap, i.e. the floating leg is reset every 3M

        start_dates = [ref_date + relativedelta(months=3 * i) for i in range(4 * 2)]
        reset_dates = start_dates
        end_dates = [x + relativedelta(months=3) for x in start_dates]
        pay_dates = end_dates
        ns = ConstNotionalStructure(100.0)
        spread = 0.00

        # # definition of the floating leg
        float_leg_2Y = IrFloatLegSpecification(
            obj_id="dummy_float_leg",
            notional=ns,
            reset_dates=reset_dates,
            start_dates=start_dates,
            end_dates=end_dates,
            rate_start_dates=start_dates,
            rate_end_dates=end_dates,
            pay_dates=pay_dates,
            currency="EUR",
            udl_id="test_udl_id",
            fixing_id="test_fixing_id",
            day_count_convention="Act365Fixed",
            spread=spread,
        )

        # # definition of the fixed leg
        fixed_leg_2Y = IrFixedLegSpecification(
            fixed_rate=0.01,
            obj_id="dummy_fixed_leg",
            notional=100.0,
            start_dates=start_dates,
            end_dates=end_dates,
            pay_dates=pay_dates,
            currency="EUR",
            day_count_convention="Act365Fixed",
        )

        # # definition of the IR swap
        irs_2Y = InterestRateSwapSpecification(
            obj_id="3M_SWAP_2Y",
            notional=ns,
            issue_date=ref_date,
            maturity_date=pay_dates[-1],
            pay_leg=fixed_leg_2Y,
            receive_leg=float_leg_2Y,
            currency="EUR",
            day_count_convention="Act365Fixed",
            issuer="dummy_issuer",
            securitization_level="COLLATERALIZED",
        )

        instruments_3M = [irs_1Y, irs_2Y]
        quotes_3M = [-0.003204, -0.002615]

        # bootstrap forward curve
        euribor3MCurve = bootstrap_curve(
            ref_date,
            "euribor3M_DC",
            DayCounterType.Act365Fixed,
            instruments_3M,
            quotes_3M,
            curves={"discount_curve": boot_curve_ois},
            interpolation_type=InterpolationType.LINEAR_LOG,
            extrapolation_type=ExtrapolationType.LINEAR_LOG,
        )

        # assertions

        model_quotes = []
        # pricing_params = {"fixing_grace_period": 0.0, "set_rate": True, "desired_rate": 1.0}
        curves_dict = {"discount_curve": euribor3MCurve, "fixing_curve": euribor3MCurve}

        for i in range(len(instruments_3M)):
            model_quote = get_quote(ref_date, instruments_3M[i], curves_dict)
            # print(model_quote)
            model_quotes.append(model_quote)

        for i in range(len(quotes_3M)):
            self.assertAlmostEqual(model_quotes[i], quotes_3M[i], places=4)

    def test_multicurve_with_deposit_raises(self):
        # Deposit in multicurve should raise
        d1 = self.ref_date + timedelta(days=182)
        deposit = DepositSpecification(
            obj_id="dep1",
            notional=1_000_000,
            issue_date=self.ref_date,
            maturity_date=d1,
            currency="EUR",
            day_count_convention=self.day_count,
            rate=0.02,
        )
        discount_curve = bootstrap_curve(
            ref_date=self.ref_date,
            curve_id="DISC",
            day_count_convention=self.day_count,
            instruments=[deposit],
            quotes=[0.02],
            interpolation_type=self.interp,
            extrapolation_type=self.extrap,
        )
        curves = {"discount_curve": discount_curve}
        with self.assertRaises(Exception) as cm:
            bootstrap_curve(
                ref_date=self.ref_date,
                curve_id="FWD",
                day_count_convention=self.day_count,
                instruments=[deposit],
                quotes=[0.02],
                curves=curves,
                interpolation_type=self.interp,
                extrapolation_type=self.extrap,
            )
        self.assertIn("Deposits cannot be used in multicurve bootstrapping", str(cm.exception))


class TestAutomaticInstrumentCreation(unittest.TestCase):
    """Helper functions used to create instrument speficiations from dataframes.

    Args:
        unittest (_type_): _description_
    """

    def setUp(self):
        """_summary_"""
        # set directory and file name for Input Quotes
        dirName = "./notebooks/marketdata"  # "./"
        fileName = "/inputQuotes_includeFRAs.csv"  # "/inputQuotes.csv"
        # fileName = "/multi_dates_tbs.csv"  # "/inputQuotes.csv"

        df = pd.read_csv(dirName + fileName, sep=";", decimal=",")
        column_names = list(df.columns)

        self.quotes_df = df
        self.column_names = column_names

    def test_create_deposits_from_df(self):
        """ """
        df = self.quotes_df.copy()
        df_deposits = df[df["Instrument"] == "DEPOSIT"]

        example_dep = df_deposits.iloc[0]
        input_data = example_dep.copy()

        # these inputs must be given by user
        refDate = datetime(2019, 3, 1)
        holidays = _ECB()

        # the following is read for every instrument type
        instr = input_data["Instrument"]
        fixDayCount = input_data["DayCountFixed"]
        floatDayCount = input_data["DayCountFloat"]
        basisDayCount = input_data["DayCountBasis"]
        maturity = input_data["Maturity"]
        tenor = input_data["UnderlyingTenor"]
        underlyingPayFreq = input_data["UnderlyingPaymentFrequency"]
        basisTenor = input_data["BasisTenor"]
        basisPayFreq = input_data["BasisPaymentFrequency"]
        fixPayFreq = input_data["PaymentFrequencyFixed"]
        rollConvFloat = input_data["RollConventionFloat"]
        rollConvFix = input_data["RollConventionFixed"]
        rollConvBasis = input_data["RollConventionBasis"]
        spotLag = input_data["SpotLag"]  # expect form "1D", i.e 1 day
        parRate = float(input_data["Quote"])
        currency = input_data["Currency"]
        label = instr + "_" + maturity

        #######################
        # so from the file, we know the TERM for sure, the MATURITY for sure, and we give the REFERENCE DATE

        # our deposit spepcificaiton can be created using the refdate, spotlag, and MATURITY to calculate the term, start, end_date

        dep_spec = DepositSpecification(
            obj_id=label,
            issue_date=refDate,
            currency=currency,
            # notional: float = 100.0, # we let notional default to 100
            rate=parRate,
            term=maturity,
            day_count_convention=floatDayCount,
            business_day_convention=rollConvFloat,
            # roll_convention: _Union[RollRule, str] = RollRule.EOM, # leave as default
            spot_days=int(spotLag[:-1]),  # make assumption it is always given in DAYS convert -> int
            calendar=holidays,
            issuer="dummy_issuer",
            securitization_level="NONE",
        )

        dep_spec2 = sfc.make_deposit_spec(input_data, refDate, holidays)

        self.assertIsInstance(dep_spec, DepositSpecification)
        self.assertIsInstance(dep_spec2, DepositSpecification)
        # self.assertEqual(dep_spec.__dict__, dep_spec2.__dict__)

    def test_create_IRS_from_df(self):
        """ """
        df = self.quotes_df.copy()
        df_irs = df[df["Instrument"] == "IRS"]

        example_irs = df_irs.iloc[0]
        input_data = example_irs.copy()

        # these inputs must be given by user
        refDate = datetime(2019, 3, 1)
        holidays = _ECB()

        # the following is read for every instrument type
        instr = input_data["Instrument"]
        fixDayCount = input_data["DayCountFixed"]
        floatDayCount = input_data["DayCountFloat"]
        basisDayCount = input_data["DayCountBasis"]
        maturity = input_data["Maturity"]
        underlyingIndex = input_data["UnderlyingIndex"]
        tenor = input_data["UnderlyingTenor"]
        underlyingPayFreq = input_data["UnderlyingPaymentFrequency"]
        basisTenor = input_data["BasisTenor"]
        basisPayFreq = input_data["BasisPaymentFrequency"]
        fixPayFreq = input_data["PaymentFrequencyFixed"]
        rollConvFloat = input_data["RollConventionFloat"]
        rollConvFix = input_data["RollConventionFixed"]
        rollConvBasis = input_data["RollConventionBasis"]
        spotLag = input_data["SpotLag"]
        parRate = float(input_data["Quote"])
        currency = input_data["Currency"]
        label = instr + "_" + maturity

        spot_date = calc_end_day(start_day=refDate, term=spotLag, business_day_convention=rollConvFix, calendar=holidays)
        expiry = calc_end_day(spot_date, maturity, rollConvFix, holidays)

        # start_day = calc_start_day(ref)
        # end_day = calc_end_day()
        # generate_dates
        fix_schedule = Schedule(
            start_day=spot_date, end_day=expiry, time_period=fixPayFreq, business_day_convention=rollConvFix, calendar=holidays, ref_date=refDate
        ).generate_dates(False)

        # fix_schedule = get_schedule(self.refDate, self.maturity, pay_freq, roll_conv, self.holidays, spot_days)
        fix_start_dates = fix_schedule[:-1]
        fix_end_dates = fix_schedule[1:]
        fix_pay_dates = fix_end_dates

        flt_schedule = Schedule(
            start_day=spot_date,
            end_day=expiry,
            time_period=underlyingPayFreq,
            business_day_convention=rollConvFloat,
            calendar=holidays,
            ref_date=refDate,
        ).generate_dates(False)

        # flt_schedule = get_schedule(self.refDate, self.maturity, pay_freq, roll_conv, self.holidays, spot_days)
        flt_start_dates = flt_schedule[:-1]
        flt_end_dates = flt_schedule[1:]
        flt_pay_dates = flt_end_dates

        flt_reset_schedule = Schedule(
            start_day=spot_date, end_day=expiry, time_period=tenor, business_day_convention=rollConvFloat, calendar=holidays, ref_date=refDate
        ).generate_dates(False)

        # flt_reset_schedule = get_schedule(self.refDate, self.maturity, reset_freq, roll_conv, self.holidays, spot_days)
        flt_reset_dates = flt_reset_schedule[:-1]

        print(flt_reset_dates)

        # start_dates3 = [ref_date + relativedelta(months=3*i) for i in range(4*3)]
        # reset_dates3 = start_dates3
        # end_dates3 = [x + relativedelta(months=3) for x in start_dates3]
        # pay_dates3 = end_dates3
        ns = ConstNotionalStructure(100.0)
        spread = 0.00

        # # definition of the floating leg
        float_leg = IrFloatLegSpecification(
            obj_id=label + "_float_leg",
            notional=ns,
            reset_dates=flt_reset_dates,
            start_dates=flt_start_dates,
            end_dates=flt_end_dates,
            rate_start_dates=flt_start_dates,
            rate_end_dates=flt_end_dates,
            pay_dates=flt_pay_dates,
            currency=currency,
            udl_id=underlyingIndex,
            fixing_id="test_fixing_id",
            day_count_convention=floatDayCount,
            spread=spread,
        )

        # # definition of the fixed leg
        fixed_leg = IrFixedLegSpecification(
            fixed_rate=parRate,
            obj_id=label + "_fixed_leg3",
            notional=100.0,
            start_dates=fix_start_dates,
            end_dates=fix_end_dates,
            pay_dates=fix_pay_dates,
            currency=currency,
            day_count_convention=fixDayCount,
        )

        # get expiry of swap (cannot be before last paydate of legs)
        # spot_date = get_end_date(self.refDate, self.spotLag)
        # expiry = get_end_date(spot_date, self.maturity)
        # # definition of the IR swap
        ir_swap = InterestRateSwapSpecification(
            obj_id=label,
            notional=ns,
            issue_date=refDate,
            maturity_date=expiry,
            pay_leg=fixed_leg,
            receive_leg=float_leg,
            currency=currency,
            day_count_convention=floatDayCount,
            issuer="dummy_issuer",
            securitization_level="COLLATERALIZED",
        )

        ir_swap2 = sfc.make_irswap_spec(input_data, refDate, holidays)
        self.maxDiff = None
        self.assertIsInstance(ir_swap, InterestRateSwapSpecification)
        self.assertIsInstance(ir_swap2, InterestRateSwapSpecification)
        self.assertTrue(deep_equal(ir_swap, ir_swap2))

    def test_create_OIS_from_df(self):
        """ """
        df = self.quotes_df.copy()
        df_ois = df[df["Instrument"] == "OIS"]

        example_ois = df_ois.iloc[0]
        input_data = example_ois.copy()

        # these inputs must be given by user
        refDate = datetime(2019, 3, 1)
        holidays = _ECB()

        # the following is read for every instrument type
        instr = input_data["Instrument"]
        fixDayCount = input_data["DayCountFixed"]
        floatDayCount = input_data["DayCountFloat"]
        basisDayCount = input_data["DayCountBasis"]
        maturity = input_data["Maturity"]
        underlyingIndex = input_data["UnderlyingIndex"]
        tenor = input_data["UnderlyingTenor"]
        underlyingPayFreq = input_data["UnderlyingPaymentFrequency"]
        basisTenor = input_data["BasisTenor"]
        basisPayFreq = input_data["BasisPaymentFrequency"]
        fixPayFreq = input_data["PaymentFrequencyFixed"]
        rollConvFloat = input_data["RollConventionFloat"]
        rollConvFix = input_data["RollConventionFixed"]
        rollConvBasis = input_data["RollConventionBasis"]
        spotLag = input_data["SpotLag"]
        parRate = float(input_data["Quote"])
        currency = input_data["Currency"]
        label = instr + "_" + maturity

        # get swap leg schedule # assume same for both fix and float legs?
        # we use the helper function with spotlag in place of maturity to effctively shift the date
        spot_date = calc_end_day(start_day=refDate, term=spotLag, business_day_convention=rollConvFix, calendar=holidays)
        expiry = calc_end_day(spot_date, maturity, rollConvFix, holidays)
        expiry_unadjusted = calc_end_day(start_day=spot_date, term=maturity, calendar=holidays)

        # start_day = calc_start_day(ref)
        # end_day = calc_end_day()
        # generate_dates
        fix_schedule = Schedule(
            start_day=spot_date,
            end_day=expiry_unadjusted,  # expiry,
            time_period=fixPayFreq,
            business_day_convention=rollConvFix,
            calendar=holidays,
            ref_date=refDate,
        ).generate_dates(False)

        # fix_schedule = get_schedule(self.refDate, self.maturity, pay_freq, roll_conv, self.holidays, spot_days)
        fix_start_dates = fix_schedule[:-1]
        fix_end_dates = fix_schedule[1:]
        fix_pay_dates = fix_end_dates

        flt_schedule = Schedule(
            start_day=spot_date,
            end_day=expiry_unadjusted,  # expiry,
            time_period=underlyingPayFreq,
            business_day_convention=rollConvFloat,
            calendar=holidays,
            ref_date=refDate,
        ).generate_dates(False)

        flt_start_dates = flt_schedule[:-1]
        flt_end_dates = flt_schedule[1:]
        flt_pay_dates = flt_end_dates

        flt_reset_schedule = Schedule(
            start_day=spot_date, end_day=expiry, time_period=tenor, business_day_convention=rollConvFloat, calendar=holidays, ref_date=refDate
        ).generate_dates(False)

        flt_reset_dates = flt_reset_schedule[:-1]

        res = IrOISLegSpecification.ois_scheduler_2D(flt_start_dates, flt_end_dates)

        daily_rate_start_dates = res[0]  # 2D list: coupon i -> list of daily starts
        daily_rate_end_dates = res[1]  # 2D list: coupon i -> list of daily ends
        daily_rate_reset_dates = res[2]  # 2D list: coupon i -> list of reset dates
        daily_rate_pay_dates = res[3]

        # print(flt_reset_dates)
        # print(daily_rate_reset_dates)

        ns = ConstNotionalStructure(100.0)
        spread = 0.00

        # # definition of the floating leg

        ois_leg = IrOISLegSpecification(
            obj_id=label + "_float_leg",
            notional=ns,
            rate_reset_dates=daily_rate_reset_dates,
            start_dates=flt_start_dates,
            end_dates=flt_end_dates,
            rate_start_dates=daily_rate_start_dates,
            rate_end_dates=daily_rate_end_dates,
            pay_dates=daily_rate_pay_dates,
            currency=currency,
            udl_id=underlyingIndex,
            fixing_id="test_fixing_id",
            day_count_convention=floatDayCount,
            rate_day_count_convention=floatDayCount,
            spread=spread,
        )

        # # definition of the fixed leg
        fixed_leg = IrFixedLegSpecification(
            fixed_rate=parRate,
            obj_id=label + "_fixed_leg3",
            notional=100.0,
            start_dates=fix_start_dates,
            end_dates=fix_end_dates,
            pay_dates=fix_pay_dates,
            currency=currency,
            day_count_convention=fixDayCount,
        )

        # # definition of the IR swap
        oi_swap = InterestRateSwapSpecification(
            obj_id=label,
            notional=ns,
            issue_date=refDate,
            maturity_date=expiry,
            pay_leg=fixed_leg,
            receive_leg=ois_leg,
            currency=currency,
            day_count_convention=floatDayCount,
            issuer="dummy_issuer",
            securitization_level="COLLATERALIZED",
        )

        oi_swap2 = sfc.make_ois_spec(input_data, refDate, holidays)
        self.maxDiff = None
        self.assertIsInstance(oi_swap, InterestRateSwapSpecification)
        self.assertIsInstance(oi_swap2, InterestRateSwapSpecification)
        self.assertTrue(deep_equal(oi_swap, oi_swap2))

    def test_create_TBS_from_df(self):
        """
        Create a tenor basis swap (TBS) specification:
        - Pay short floating leg
        - Receive long floating leg
        - Pay fixed spread leg (represents market quote)
        """
        df = self.quotes_df.copy()
        df_irs = df[df["Instrument"] == "TBS"]

        example_irs = df_irs.iloc[0]
        row = example_irs.copy()

        # these inputs must be given by user
        refDate = datetime(2019, 3, 1)
        holidays = _ECB()

        # --- Extract general fields ---
        instr = row["Instrument"]
        currency = row["Currency"]
        maturity = row["Maturity"]
        spot_lag = row["SpotLag"]
        roll_conv = row["RollConventionFloat"]
        fixDayCount = row["DayCountFloat"]
        floatDayCount = row["DayCountFloat"]
        basisDayCount = row["DayCountBasis"]
        rollConvFix = row["RollConventionFixed"]
        rollConvBasis = row["RollConventionBasis"]
        # --- Long (receive) leg info ---
        long_index = row["UnderlyingIndex"]
        long_tenor = row["UnderlyingTenor"]
        long_freq = row["UnderlyingPaymentFrequency"]

        # --- Short (pay) leg info ---
        short_index = row["UnderlyingIndex"]
        short_tenor = row["UnderlyingTenorShort"]
        short_freq = row["UnderlyingPaymentFrequencyShort"]

        # --- Spread (basis quote) ---
        spread_rate = float(row["Quote"]) / 10000.0  # e.g. 8.5 bps -> 0.00085

        # --- Spot and maturity dates ---
        spot_date = calc_end_day(refDate, spot_lag, roll_conv, holidays)
        expiry = calc_end_day(spot_date, maturity, roll_conv, holidays)
        label = f"{instr}_{maturity}"

        ns = ConstNotionalStructure(100.0)

        # --------------------------------------------
        # PAY FLOATING LEG (short tenor, pays basis)
        short_schedule = Schedule(
            start_day=spot_date,
            end_day=expiry,
            time_period=short_freq,
            business_day_convention=roll_conv,
            calendar=holidays,
            ref_date=refDate,
        ).generate_dates(False)

        short_start = short_schedule[:-1]
        short_end = short_schedule[1:]
        short_pay = short_end
        short_reset = Schedule(
            start_day=spot_date,
            end_day=expiry,
            time_period=short_tenor,
            business_day_convention=roll_conv,
            calendar=holidays,
            ref_date=refDate,
        ).generate_dates(False)[:-1]

        pay_leg = IrFloatLegSpecification(
            obj_id=label + "_pay_leg",
            notional=ns,
            reset_dates=short_reset,
            start_dates=short_start,
            end_dates=short_end,
            rate_start_dates=short_start,
            rate_end_dates=short_end,
            pay_dates=short_pay,
            currency=currency,
            udl_id=short_index,
            fixing_id="test_fixing_id",
            day_count_convention=floatDayCount,
            spread=float(row["Quote"]),  # this is the quoted basis
        )

        # --------------------------------------------
        # RECEIVE FLOATING LEG (long tenor)

        long_schedule = Schedule(
            start_day=spot_date,
            end_day=expiry,
            time_period=long_freq,
            business_day_convention=roll_conv,
            calendar=holidays,
            ref_date=refDate,
        ).generate_dates(False)

        long_start = long_schedule[:-1]
        long_end = long_schedule[1:]
        long_pay = long_end
        long_reset = Schedule(
            start_day=spot_date,
            end_day=expiry,
            time_period=long_tenor,
            business_day_convention=roll_conv,
            calendar=holidays,
            ref_date=refDate,
        ).generate_dates(False)[:-1]

        receive_leg = IrFloatLegSpecification(
            obj_id=label + "_receive_leg",
            notional=ns,
            reset_dates=long_reset,
            start_dates=long_start,
            end_dates=long_end,
            rate_start_dates=long_start,
            rate_end_dates=long_end,
            pay_dates=long_pay,
            currency=currency,
            udl_id=long_index,
            fixing_id="test_fixing_id",
            day_count_convention=floatDayCount,
            spread=0.0,
        )

        # --------------------------------------------
        #  FIXED SPREAD LEG
        # The spread leg represents the fixed +x bps cashflows applied to the pay leg
        spread_schedule = Schedule(
            start_day=spot_date,
            end_day=expiry,
            time_period=short_freq,  # same freq as short leg
            business_day_convention=rollConvFix,
            calendar=holidays,
            ref_date=refDate,
        ).generate_dates(False)

        spread_start = spread_schedule[:-1]
        spread_end = spread_schedule[1:]
        spread_pay = spread_end

        spread_leg = IrFixedLegSpecification(
            fixed_rate=spread_rate,
            obj_id=label + "_spread_leg",
            notional=100.0,
            start_dates=spread_start,
            end_dates=spread_end,
            pay_dates=spread_pay,
            currency=currency,
            day_count_convention=fixDayCount,
        )

        # --------------------------------------------
        # Combine into full TBS object
        basis_swap = InterestRateBasisSwapSpecification(
            obj_id=label,
            notional=ns,
            issue_date=refDate,
            maturity_date=expiry,
            pay_leg=pay_leg,
            receive_leg=receive_leg,
            spread_leg=spread_leg,
            currency=currency,
            day_count_convention=floatDayCount,
            issuer="dummy_issuer",
            securitization_level="COLLATERALIZED",
        )

        basis_swap2 = sfc.make_basis_swap_spec(row, refDate, holidays)

        self.maxDiff = None
        self.assertIsInstance(basis_swap, InterestRateBasisSwapSpecification)
        self.assertIsInstance(basis_swap2, InterestRateBasisSwapSpecification)
        self.assertTrue(deep_equal(basis_swap, basis_swap2))

    def test_create_FRA_from_df(self):
        """Assuming a 3Mx6M Forward rate agreement instrument"""
        df = self.quotes_df.copy()
        df_fra = df[df["Instrument"] == "FRA"]

        example_fra = df_fra.iloc[0]
        input_data = example_fra.copy()
        self.assertEqual(input_data["Maturity"], "3Mx6M")
        self.assertEqual(input_data["SpotLag"], "2D")
        # these inputs must be given by user
        refDate = datetime(2019, 4, 1)
        holidays = _ECB()
        # spot_days = 2
        start_date = datetime(2019, 4 + 3, 3)
        end_date = datetime(2019, 4 + 3 + 3, 3)
        label = input_data["Instrument"] + "_" + input_data["Maturity"]

        fra_spec = ForwardRateAgreementSpecification(
            obj_id=label,
            trade_date=refDate,
            notional=100,
            rate=float(input_data["Quote"]),
            start_date=start_date,
            end_date=end_date,
            udlID=input_data["UnderlyingIndex"],
            rate_start_date=start_date,
            rate_end_date=end_date,
            # maturity_date=,
            day_count_convention=input_data["DayCountFixed"],
            business_day_convention=input_data["RollConventionFixed"],
            rate_day_count_convention=input_data["DayCountFloat"],
            rate_business_day_convention=input_data["RollConventionFloat"],
            calendar=holidays,
            currency=input_data["Currency"],
            # payment_days: int = 0,
            spot_days=int(input_data["SpotLag"][:-1]),
            # start_period: int = None,
            # end_period: int = None,
            # ir_index: str = None,
            # issuer: str = None,
        )

        fra_spec2 = sfc.make_fra_spec(input_data, refDate, holidays)
        self.assertIsInstance(fra_spec, ForwardRateAgreementSpecification)
        self.assertIsInstance(fra_spec2, ForwardRateAgreementSpecification)
        self.assertEqual(fra_spec.start_date, fra_spec2.start_date)
        self.assertEqual(fra_spec.end_date, fra_spec2.end_date)
        # self.assertEqual(fra_spec.__dict__, fra_spec2.__dict__)

    def test_bootstrap_deposits_from_df(self):
        """Test of the bootstrap function using deposit specifications
        parsed from a datafram of expected format

        Assumption is that the deposits are already ordered by maturity...

        """
        # these inputs must be given by user
        refDate = datetime(2019, 3, 1)
        holidays = _ECB()

        df = self.quotes_df.copy()
        df_ins = df[df["Instrument"] == "DEPOSIT"]

        ins_spec = sfc.load_specifications_from_pd(df_ins, refDate, holidays)
        ins_quotes = df_ins["Quote"].tolist()

        print("--------------DEBUG PARSING")
        print(ins_quotes[0])
        print(df_ins["DayCountFixed"][0])
        print(df_ins)
        print("--------------Starting bootstrapper")
        curve = bootstrap_curve(
            ref_date=refDate,
            curve_id="dc_deposits",
            day_count_convention=df_ins["DayCountFixed"][0],  # taken the first entry and assume is valid for all other deposits
            instruments=ins_spec,
            quotes=ins_quotes,
            interpolation_type=InterpolationType.LINEAR,
            extrapolation_type=ExtrapolationType.LINEAR,
        )
        # print(curve.get_dates())
        self.assertIsInstance(curve, DiscountCurve)
        self.assertEqual(curve.get_dates()[0], refDate)

        # the discount curve needs to be able to get the same market quote for the instrument
        for i in range(len(ins_spec)):
            model_quote = get_quote(refDate, ins_spec[i], {"discount_curve": curve})
            self.assertAlmostEqual(model_quote, ins_quotes[i], delta=1e-6)
            # per_diff = (model_quote - deposit_quotes[i]) / deposit_quotes[i] * 100
            # print(f"model: {model_quote} market: {deposit_quotes[i]} perdiff: {per_diff}")

        # self.assertEqual(1, 1)

    def test_bootstrap_FRAs_from_df(self):
        """Test of the bootstrap function using deposit specifications
        parsed from a datafram of expected format

        Assumption is that the deposits are already ordered by maturity...

        """
        # these inputs must be given by user
        refDate = datetime(2019, 3, 1)
        holidays = _ECB()

        df = self.quotes_df.copy()
        df_ins = df[df["Instrument"] == "FRA"]

        ins_spec = sfc.load_specifications_from_pd(df_ins, refDate, holidays)
        ins_quotes = df_ins["Quote"].tolist()

        print("--------------DEBUG PARSING")
        print(ins_quotes[0])
        print(df_ins["DayCountFixed"].tolist()[0])
        print(df_ins)
        print("--------------Starting bootstrapper")
        curve = bootstrap_curve(
            ref_date=refDate,
            curve_id="dc_deposits",
            day_count_convention=df_ins["DayCountFixed"].tolist()[0],  # taken the first entry and assume is valid for all other deposits
            instruments=ins_spec,
            quotes=ins_quotes,
            interpolation_type=InterpolationType.LINEAR,
            extrapolation_type=ExtrapolationType.LINEAR,
        )
        # print(curve.get_dates())
        self.assertIsInstance(curve, DiscountCurve)
        self.assertEqual(curve.get_dates()[0], refDate)

        # the discount curve needs to be able to get the same market quote for the instrument
        for i in range(len(ins_spec)):
            model_quote = get_quote(refDate, ins_spec[i], {"discount_curve": curve})
            self.assertAlmostEqual(model_quote, ins_quotes[i], delta=1e-6)
            # per_diff = (model_quote - deposit_quotes[i]) / deposit_quotes[i] * 100
            # print(f"model: {model_quote} market: {deposit_quotes[i]} perdiff: {per_diff}")

        # self.assertEqual(1, 1)

    def test_bootstrap_ois_from_df(self):
        """Test of the bootstrap function using ois specifications
        parsed from a datafram of expected format

        Assumption is that the ois are already ordered by maturity...

        """
        logger.debug(f"--------------------------------------------------------")
        logger.debug("test_bootstrap_ois_from_df start")
        # these inputs must be given by user
        refDate = datetime(2019, 3, 1)
        holidays = _ECB()

        df = self.quotes_df.copy()
        df_ins = df[df["Instrument"] == "OIS"]

        logger.debug("CSV loaded")

        min_i = 0
        max_i = 18  # 19-25 problematic?
        min_i2 = 26
        max_i2 = len(df_ins)
        ins_spec = sfc.load_specifications_from_pd(df_ins.iloc[np.r_[min_i:max_i, min_i2:max_i2]], refDate, holidays)
        # ins_quotes = df_ins["Quote"].tolist()[min_i:max_i]
        ins_quotes = df_ins["Quote"].tolist()[min_i:max_i] + df_ins["Quote"].tolist()[min_i2:max_i2]

        logger.debug("instrument specifications created")

        # print("--------------DEBUG PARSING")
        # print(ins_quotes[0])
        # print(df_ins["DayCountFixed"].tolist()[0])
        # print(df_ins.iloc[min_i:max_i].copy())
        # print(len(ins_spec), len(ins_quotes))
        # print(len(df_ins["Quote"].tolist()))
        # print(len(df_ins.iloc[np.r_[min_i:max_i, min_i2:max_i2]]))
        # for i in range(len(ins_quotes)):
        #     print(i, ins_quotes[i])

        print("--------------Starting bootstrapper")
        logger.debug(f"starting bootstrapper of {len(ins_quotes)} instruments")
        curve = bootstrap_curve(
            ref_date=refDate,
            curve_id="OIS_estr",
            day_count_convention=df_ins["DayCountFixed"].tolist()[0],  # taken the first entry and assume is valid for all other deposits
            instruments=ins_spec,
            quotes=ins_quotes,
            interpolation_type=InterpolationType.LINEAR_LOG,
            extrapolation_type=ExtrapolationType.LINEAR_LOG,
        )
        # print(curve.get_dates())
        self.assertIsInstance(curve, DiscountCurve)
        self.assertEqual(curve.get_dates()[0], refDate)
        logger.debug("bootstrapped curve dates matched")
        # the discount curve needs to be able to get the same market quote for the instrument
        for i in range(len(ins_spec)):
            model_quote = get_quote(refDate, ins_spec[i], {"discount_curve": curve, "fixing_curve": curve})
            self.assertAlmostEqual(model_quote, ins_quotes[i], delta=1e-6)
            # per_diff = (model_quote - deposit_quotes[i]) / deposit_quotes[i] * 100
            # print(f"model: {model_quote} market: {deposit_quotes[i]} perdiff: {per_diff}")

        logger.debug(f"asserted market quote matched -done")
        logger.debug(f"--------------------------------------------------------")
        # self.assertEqual(1, 1)

    def test_multicurve_bootstrap_ois_3M(self):
        """Test of the bootstrap function in the context of multicurve bootstrapping
        using ois and irs specifications parsed from a datafram of expected format

        """

        # these inputs must be given by user
        refDate = datetime(2019, 3, 1)
        holidays = _ECB()

        ###############################
        # PREPARE discount curve
        df = self.quotes_df.copy()
        df_ins = df[df["Instrument"] == "OIS"]

        min_i = 0
        max_i = 17  # up to 3 years ...

        ins_spec = sfc.load_specifications_from_pd(df_ins.iloc[min_i:max_i], refDate, holidays)
        ins_quotes = df_ins["Quote"].tolist()[min_i:max_i]

        print("--------------DEBUG PARSING")
        print(ins_quotes[0])
        print(df_ins["DayCountFixed"].tolist()[0])
        print(df_ins.iloc[min_i:max_i].copy())
        print("--------------Starting bootstrapper")
        curve_ois = bootstrap_curve(
            ref_date=refDate,
            curve_id="OIS_estr",
            day_count_convention=df_ins["DayCountFixed"].tolist()[0],  # taken the first entry and assume is valid for all other deposits
            instruments=ins_spec,
            quotes=ins_quotes,
            interpolation_type=InterpolationType.LINEAR,
            extrapolation_type=ExtrapolationType.LINEAR,
        )
        # print(curve.get_dates())
        self.assertIsInstance(curve_ois, DiscountCurve)
        self.assertEqual(curve_ois.get_dates()[0], refDate)

        # the discount curve needs to be able to get the same market quote for the instrument
        for i in range(len(ins_spec)):
            model_quote = get_quote(refDate, ins_spec[i], {"discount_curve": curve_ois, "fixing_curve": curve_ois})
            self.assertAlmostEqual(model_quote, ins_quotes[i], delta=1e-6)
            # per_diff = (model_quote - deposit_quotes[i]) / deposit_quotes[i] * 100
            # print(f"model: {model_quote} market: {deposit_quotes[i]} perdiff: {per_diff}")

        # self.assertEqual(1, 1)

        ##################################################
        # select for 3M instruments
        min_i = 0
        max_i = -1

        # df_ins_3M = df[(df["UnderlyingIndex"] == "EURIBOR") & (df["UnderlyingTenor"] == "3M")]
        df_ins_3M = df[(df["UnderlyingIndex"] == "EURIBOR") & (df["UnderlyingTenor"] == "3M") & (df["Instrument"] == "IRS")]
        ins_spec_3M = sfc.load_specifications_from_pd(df_ins_3M.iloc[min_i:max_i], refDate, holidays)
        ins_quotes_3M = df_ins_3M["Quote"].tolist()[min_i:max_i]

        # bootstrap forward curve
        euribor3MCurve = bootstrap_curve(
            refDate,
            "euribor3M_DC",
            DayCounterType.Act365Fixed,
            ins_spec_3M,
            ins_quotes_3M,
            curves={"discount_curve": curve_ois},
            interpolation_type=InterpolationType.LINEAR_LOG,
            extrapolation_type=ExtrapolationType.LINEAR_LOG,
        )

        for i in range(len(ins_spec_3M)):
            model_quote = get_quote(refDate, ins_spec_3M[i], {"discount_curve": curve_ois, "fixing_curve": euribor3MCurve})
            self.assertAlmostEqual(model_quote, ins_quotes_3M[i], delta=1e-6)
            # per_diff = (model_quote - deposit_quotes[i]) / deposit_quotes[i] * 100
            # print(f"model: {model_quote} market: {deposit_quotes[i]} perdiff: {per_diff}")


class TestBSBootstrap(unittest.TestCase):
    """ """

    def setUp(self):
        """_summary_"""
        # set directory and file name for Input Quotes
        dirName = "./notebooks/marketdata"  # "./"
        fileName = "/multi_dates_tbs.csv"  # "/inputQuotes.csv"

        df = pd.read_csv(dirName + fileName, sep=";", decimal=",")
        column_names = list(df.columns)

        self.quotes_df = df
        self.column_names = column_names

    def test_tbs_3m_6m(self):
        """Using 2025 09 24 as a control date, to ensure the proper bootstrapping from Frontmark data example.
        Goes through the complete process of bootstrapping 3 times
        1. produce OIS curve for discounting
        2. produce fwd curve e.g. 3M euribor from IRS instruments
        3. produce 6M euribor from TBS instruments and 3m Euribor"""

        logger.debug(f"--------------------------------------------------------")
        logger.debug("start")
        # these inputs must be given by user
        mon = "09"
        day = "24"
        year = "2025"
        date_str = f"{day}.{mon}.{year}"
        refDate = datetime(int(year), int(mon), int(day))
        holidays = _ECB()

        df = self.quotes_df.copy()

        A = df[df["Date"] == date_str]

        logger.debug(f"Creating OIS discount Curve")
        # df_ins = df[df["Instrument"] == "OIS"]
        # dc_df_temp = df[(df["Date"] == selected_date) & (df["Currency"] == selected_currency)  & (df["UnderlyingIndex"] == "ESTR") & (df["Instrument"] == "OIS") ]
        df_ins = df[(df["Date"] == date_str) & (df["Currency"] == "EUR") & (df["UnderlyingIndex"] == "EONIA") & (df["Instrument"] == "OIS")]

        logger.debug("CSV loaded")

        ins_spec = sfc.load_specifications_from_pd(df_ins, refDate, holidays)
        ins_quotes = df_ins["Quote"].tolist()
        logger.debug("instrument specifications created")

        print("--------------Starting bootstrapper")
        logger.debug(f" OIS starting bootstrapper of {len(ins_quotes)} instruments")
        curve = bootstrap_curve(
            ref_date=refDate,
            curve_id="OIS_estr",
            day_count_convention=df_ins["DayCountFixed"].tolist()[0],  # taken the first entry and assume is valid for all other deposits
            instruments=ins_spec,
            quotes=ins_quotes,
            interpolation_type=InterpolationType.LINEAR_LOG,
            extrapolation_type=ExtrapolationType.LINEAR_LOG,
            # interpolation_type=InterpolationType.HAGAN_DF,
            # extrapolation_type=ExtrapolationType.CONSTANT_DF,
        )

        # OUtput discoutn curve dates and values for test:
        print(" OIS: curve valueus (date, DF)")
        dates_ois = curve.get_dates()
        df_ois = curve.get_df()
        for i in range(len(dates_ois)):
            print(f"{dates_ois[i]} , {df_ois[i]}")

        # --------------------------------- IR for 3M
        logger.debug(f"Loading  instruments for reference date and, currency, and EURIBOR for SHORT LEG, e.g. 3M in this case")
        df_ins_3m = df[
            (df["Date"] == date_str)
            & (df["Currency"] == "EUR")
            & (df["UnderlyingIndex"] == "EURIBOR")
            & (df["Instrument"] == "IRS")
            & (df["UnderlyingTenor"] == "3M")
        ]

        ins_spec_3m = sfc.load_specifications_from_pd(df_ins_3m, refDate, holidays)
        ins_quotes_3m = df_ins_3m["Quote"].tolist()
        logger.debug("instrument specifications created")

        curves = {"discount_curve": curve}
        print("--------------Starting bootstrapper")
        logger.debug(f" IRS starting bootstrapper of {len(ins_quotes_3m)} instruments")
        curve_3m = bootstrap_curve(
            ref_date=refDate,
            curve_id="euribor3m",
            day_count_convention=df_ins_3m["DayCountFixed"].tolist()[0],  # taken the first entry and assume is valid for all other deposits
            instruments=ins_spec_3m,
            quotes=ins_quotes_3m,
            curves=curves,
            interpolation_type=InterpolationType.LINEAR_LOG,
            extrapolation_type=ExtrapolationType.LINEAR_LOG,
            # interpolation_type=InterpolationType.HAGAN_DF,
            # extrapolation_type=ExtrapolationType.CONSTANT_DF,
        )

        # OUtput discoutn curve dates and values for test:
        print(" 3m euribor: curve valueus (date, DF)")
        dates_3m = curve_3m.get_dates()
        df_3m = curve_3m.get_df()
        for i in range(len(dates_3m)):
            print(f"{dates_3m[i]} , {df_3m[i]}")

        # --------------------------------- TBS
        logger.debug(f"Loading TBS instruments for reference date and, currency, and EURIBOR, 6M long tenor")
        df_ins_tbs = df[
            (df["Date"] == date_str)
            & (df["Currency"] == "EUR")
            & (df["UnderlyingIndex"] == "EURIBOR")
            & (df["Instrument"] == "TBS")
            & (df["UnderlyingTenor"] == "6M")
        ]

        if df_ins_tbs.empty:
            raise ValueError(f"No TBS instruments found for date {date_str} with EUR/EURIBOR.")

        print("--------- TBS instruments used ...")
        for _, item in df_ins_tbs.iterrows():
            print(item["Date"], item["Instrument"], item["Maturity"], item["Quote"], item["UnderlyingTenorShort"], item["UnderlyingTenor"])

        ins_spec_tbs = sfc.load_specifications_from_pd(df_ins_tbs, refDate, holidays)
        # ins_quotes_tbs = df_ins_tbs["Quote"].tolist()
        ins_quotes_tbs = (df_ins_tbs["Quote"] / 10000.0).tolist()
        logger.debug("instrument specifications created")

        curves["basis_curve"] = curve_3m

        print("--------------Starting bootstrapper")
        logger.debug(f" TBS starting bootstrapper of {len(df_ins_tbs)} instruments")
        curve_6m = bootstrap_curve(
            ref_date=refDate,
            curve_id="OIS_estr",
            day_count_convention=df_ins_tbs["DayCountFixed"].tolist()[0],  # taken the first entry and assume is valid for all other deposits
            instruments=ins_spec_tbs,
            quotes=ins_quotes_tbs,
            curves=curves,
            interpolation_type=InterpolationType.LINEAR_LOG,
            extrapolation_type=ExtrapolationType.LINEAR_LOG,
            # interpolation_type=InterpolationType.HAGAN_DF,
            # extrapolation_type=ExtrapolationType.CONSTANT_DF,
        )

        # print(curve.get_dates())
        self.assertIsInstance(curve_6m, DiscountCurve)
        self.assertEqual(curve_6m.get_dates()[0], refDate)
        logger.debug("bootstrapped curve dates matched")
        # the discount curve needs to be able to get the same market quote for the instrument
        for i in range(len(ins_spec_tbs)):
            model_quote = get_quote(refDate, ins_spec_tbs[i], {"discount_curve": curve, "fixing_curve": curve_6m, "basis_curve": curve_3m})
            # compare to given basis points
            self.assertAlmostEqual(model_quote, ins_quotes_tbs[i], delta=1e-5)  # since the quotes are only to 5 decimals
            # per_diff = (model_quote - deposit_quotes[i]) / deposit_quotes[i] * 100
            # print(f"model: {model_quote} market: {deposit_quotes[i]} perdiff: {per_diff}")
            # print(i, model_quote, curve.get_df()[i])

        logger.debug(f"asserted market quote matched -done")
        logger.debug(f"--------------------------------------------------------")
        # self.assertEqual(1, 1)

    def test_tbs_premade(self):
        """To speed up the test, and test only the production of the TBS curve given a discount curve and basis curve (e.g. 3M euribor)"""

        logger.debug(f"--------------------------------------------------------")
        logger.debug("start")
        # these inputs must be given by user
        mon = "09"
        day = "24"
        year = "2025"
        date_str = f"{day}.{mon}.{year}"
        refDate = datetime(int(year), int(mon), int(day))
        holidays = _ECB()

        df = self.quotes_df.copy()

        A = df[df["Date"] == date_str]

        logger.debug(f"Creating OIS discount Curve")
        # df_ins = df[df["Instrument"] == "OIS"]
        # dc_df_temp = df[(df["Date"] == selected_date) & (df["Currency"] == selected_currency)  & (df["UnderlyingIndex"] == "ESTR") & (df["Instrument"] == "OIS") ]
        df_ins = df[(df["Date"] == date_str) & (df["Currency"] == "EUR") & (df["UnderlyingIndex"] == "EONIA") & (df["Instrument"] == "OIS")]

        logger.debug("CSV loaded")

        ins_spec = sfc.load_specifications_from_pd(df_ins, refDate, holidays)
        ins_quotes = df_ins["Quote"].tolist()
        logger.debug("instrument specifications created")

        print("--------------Starting bootstrapper")
        logger.debug(f" OIS starting bootstrapper of {len(ins_quotes)} instruments")
        # curve = bootstrap_curve(
        #     ref_date=refDate,
        #     curve_id="OIS_estr",
        #     day_count_convention=df_ins["DayCountFixed"].tolist()[0],  # taken the first entry and assume is valid for all other deposits
        #     instruments=ins_spec,
        #     quotes=ins_quotes,
        #     interpolation_type=InterpolationType.LINEAR_LOG,
        #     extrapolation_type=ExtrapolationType.LINEAR_LOG,
        #     # interpolation_type=InterpolationType.HAGAN_DF,
        #     # extrapolation_type=ExtrapolationType.CONSTANT_DF,
        # )
        ois_dates = [
            datetime(2025, 9, 24, 0, 0),
            datetime(2025, 10, 3, 0, 0),
            datetime(2025, 10, 10, 0, 0),
            datetime(2025, 10, 27, 0, 0),
            datetime(2025, 11, 26, 0, 0),
            datetime(2025, 12, 29, 0, 0),
            datetime(2026, 1, 26, 0, 0),
            datetime(2026, 2, 26, 0, 0),
            datetime(2026, 3, 26, 0, 0),
            datetime(2026, 4, 27, 0, 0),
            datetime(2026, 5, 26, 0, 0),
            datetime(2026, 6, 26, 0, 0),
            datetime(2026, 7, 27, 0, 0),
            datetime(2026, 8, 26, 0, 0),
            datetime(2026, 9, 28, 0, 0),
            datetime(2027, 3, 30, 0, 0),
            datetime(2027, 9, 27, 0, 0),
            datetime(2028, 9, 26, 0, 0),
            datetime(2029, 9, 26, 0, 0),
            datetime(2030, 9, 26, 0, 0),
            datetime(2031, 9, 26, 0, 0),
            datetime(2032, 9, 27, 0, 0),
            datetime(2033, 9, 26, 0, 0),
            datetime(2034, 9, 26, 0, 0),
            datetime(2035, 9, 26, 0, 0),
            datetime(2040, 9, 26, 0, 0),
            datetime(2045, 9, 26, 0, 0),
            datetime(2055, 9, 27, 0, 0),
        ]

        ois_dfs = [
            1.0,
            0.9995183313174537,
            0.9991111934028121,
            0.9981592783704503,
            0.9964906189442938,
            0.9946699855622928,
            0.9931280849220783,
            0.9914487731890309,
            0.9899089952356281,
            0.9882165869484604,
            0.9866937212054259,
            0.9850722027277637,
            0.983433328030071,
            0.9818472981388028,
            0.9801584057930596,
            0.9706595090105641,
            0.9604274957650127,
            0.9388051694832871,
            0.9160391166426118,
            0.8922971398755308,
            0.8681004825478764,
            0.8432558412799663,
            0.8181088759777948,
            0.7927910510425162,
            0.7677179931517396,
            0.6488447684675088,
            0.5517820851696722,
            0.41133831341116706,
        ]
        # ACT365FIXED, LINEAR, NONE -  EXPECT ERROR TO BE THROWN for EXTRAPOLATIOn
        curve = DiscountCurve(
            "OIS_estr",
            refDate,
            ois_dates,
            ois_dfs,
            InterpolationType.LINEAR_LOG,
            ExtrapolationType.LINEAR_LOG,
            DayCounterType.ACT360,
        )

        # OUtput discoutn curve dates and values for test:
        print(" OIS: curve valueus (date, DF)")
        dates_ois = curve.get_dates()
        df_ois = curve.get_df()
        for i in range(len(dates_ois)):
            print(f"{dates_ois[i]} , {df_ois[i]}")

        # --------------------------------- IR for 3M
        logger.debug(f"Loading  instruments for reference date and, currency, and EURIBOR for SHORT LEG, e.g. 3M in this case")
        df_ins_3m = df[
            (df["Date"] == date_str)
            & (df["Currency"] == "EUR")
            & (df["UnderlyingIndex"] == "EURIBOR")
            & (df["Instrument"] == "IRS")
            & (df["UnderlyingTenor"] == "3M")
        ]

        ins_spec_3m = sfc.load_specifications_from_pd(df_ins_3m, refDate, holidays)
        ins_quotes_3m = df_ins_3m["Quote"].tolist()
        logger.debug("instrument specifications created")

        curves = {"discount_curve": curve}
        print("--------------Starting bootstrapper")
        logger.debug(f" IRS starting bootstrapper of {len(ins_quotes_3m)} instruments")
        # curve_3m = bootstrap_curve(
        #     ref_date=refDate,
        #     curve_id="euribor3m",
        #     day_count_convention=df_ins_3m["DayCountFixed"].tolist()[0],  # taken the first entry and assume is valid for all other deposits
        #     instruments=ins_spec_3m,
        #     quotes=ins_quotes_3m,
        #     curves=curves,
        #     interpolation_type=InterpolationType.LINEAR_LOG,
        #     extrapolation_type=ExtrapolationType.LINEAR_LOG,
        #     # interpolation_type=InterpolationType.HAGAN_DF,
        #     # extrapolation_type=ExtrapolationType.CONSTANT_DF,
        # )

        eur3m_dates = [
            datetime(2025, 9, 24, 0, 0),
            datetime(2026, 9, 28, 0, 0),
            datetime(2027, 9, 27, 0, 0),
            datetime(2028, 9, 26, 0, 0),
            datetime(2029, 9, 26, 0, 0),
            datetime(2030, 9, 26, 0, 0),
            datetime(2031, 9, 26, 0, 0),
            datetime(2032, 9, 27, 0, 0),
            datetime(2033, 9, 26, 0, 0),
            datetime(2034, 9, 26, 0, 0),
            datetime(2035, 9, 26, 0, 0),
            datetime(2036, 9, 26, 0, 0),
            datetime(2037, 9, 28, 0, 0),
            datetime(2040, 9, 26, 0, 0),
            datetime(2045, 9, 26, 0, 0),
            datetime(2050, 9, 26, 0, 0),
            datetime(2055, 9, 27, 0, 0),
            datetime(2065, 9, 28, 0, 0),
            datetime(2075, 9, 26, 0, 0),
            datetime(2085, 9, 26, 0, 0),
        ]

        eur3m_dfs = [
            1.0,
            0.9799030353171693,
            0.9597990484748371,
            0.9380048482094476,
            0.9149505395343006,
            0.8912209769075741,
            0.8670805932120672,
            0.8423413056820628,
            0.8173570936292627,
            0.7921402251628151,
            0.7670503220512309,
            0.7422680023945799,
            0.7177013315988795,
            0.6486808461523088,
            0.5525449999385698,
            0.4763668626646936,
            0.4129322567797108,
            0.31451512666881004,
            0.24795753632011802,
            0.20081662961055713,
        ]
        # ACT365FIXED, LINEAR, NONE -  EXPECT ERROR TO BE THROWN for EXTRAPOLATIOn
        curve_3m = DiscountCurve(
            "euribor3m",
            refDate,
            eur3m_dates,
            eur3m_dfs,
            InterpolationType.LINEAR_LOG,
            ExtrapolationType.LINEAR_LOG,
            DayCounterType.ACT360,
        )

        # OUtput discoutn curve dates and values for test:
        print(" 3m euribor: curve valueus (date, DF)")
        dates_3m = curve_3m.get_dates()
        df_3m = curve_3m.get_df()
        for i in range(len(dates_3m)):
            print(f"{dates_3m[i]} , {df_3m[i]}")

        # --------------------------------- TBS
        logger.debug(f"Loading TBS instruments for reference date and, currency, and EURIBOR, 6M long tenor")
        df_ins_tbs = df[
            (df["Date"] == date_str)
            & (df["Currency"] == "EUR")
            & (df["UnderlyingIndex"] == "EURIBOR")
            & (df["Instrument"] == "TBS")
            & (df["UnderlyingTenor"] == "6M")
        ]

        if df_ins_tbs.empty:
            raise ValueError(f"No TBS instruments found for date {date_str} with EUR/EURIBOR.")

        print("--------- TBS instruments used ...")
        for _, item in df_ins_tbs.iterrows():
            print(item["Date"], item["Instrument"], item["Maturity"], item["Quote"], item["UnderlyingTenorShort"], item["UnderlyingTenor"])
            logger.info(
                f"{item["Date"]}, {item["Instrument"]}, {item["Maturity"]}, {item["Quote"]}, {item["UnderlyingTenorShort"]}, {item["UnderlyingTenor"]}"
            )

        ins_spec_tbs = sfc.load_specifications_from_pd(df_ins_tbs, refDate, holidays)
        # ins_quotes_tbs = df_ins_tbs["Quote"].tolist()
        ins_quotes_tbs = (df_ins_tbs["Quote"] / 10000.0).tolist()
        logger.debug("instrument specifications created")

        curves["basis_curve"] = curve_3m

        print("--------------Starting bootstrapper")
        logger.debug(f" TBS starting bootstrapper of {len(df_ins_tbs)} instruments")
        curve_6m = bootstrap_curve(
            ref_date=refDate,
            curve_id="OIS_estr",
            day_count_convention=df_ins_tbs["DayCountFixed"].tolist()[0],  # taken the first entry and assume is valid for all other deposits
            instruments=ins_spec_tbs,
            quotes=ins_quotes_tbs,
            curves=curves,
            interpolation_type=InterpolationType.LINEAR_LOG,
            extrapolation_type=ExtrapolationType.LINEAR_LOG,
            # interpolation_type=InterpolationType.HAGAN_DF,
            # extrapolation_type=ExtrapolationType.CONSTANT_DF,
        )

        # print(curve.get_dates())
        self.assertIsInstance(curve_6m, DiscountCurve)
        self.assertEqual(curve_6m.get_dates()[0], refDate)
        logger.debug("bootstrapped curve dates matched")
        # the discount curve needs to be able to get the same market quote for the instrument
        for i in range(len(ins_spec_tbs)):
            model_quote = get_quote(refDate, ins_spec_tbs[i], {"discount_curve": curve, "fixing_curve": curve_6m, "basis_curve": curve_3m})
            # compare to given basis points
            self.assertAlmostEqual(model_quote, ins_quotes_tbs[i], delta=1e-5)  # since the quotes are only to 5 decimals
            # per_diff = (model_quote - deposit_quotes[i]) / deposit_quotes[i] * 100
            # print(f"model: {model_quote} market: {deposit_quotes[i]} perdiff: {per_diff}")
            # print(i, model_quote, curve.get_df()[i])

        logger.debug(f"asserted market quote matched -done")
        logger.debug(f"--------------------------------------------------------")
        # self.assertEqual(1, 1)


class TestReferenceDateDependance(unittest.TestCase):
    """Noticed that depending on the stated reference date, which is needed to calculate
    the start dates of the instruments, the bootstrapped curve can differ slightly.
    This test checks that bootstrapping with different reference dates, but otherwise
    identical input data, gives similar curves.

    The finer points is because the start dates, and the adjustments to the date can affect
    the actual day count fractions, and thus the cash flows, and thus the curve. Especially after applying
    conventions like modified following.

    Error was determined in the case of OIS, when feeding dates to the scheduler,
    if the expiry date is adjusted, when rolling back to get the start date, the start date can
    be earlier than the actual start date, causing inconsistencies and uexpected behaviour in the Scheduler.
    This is affected by the roll convention used.

    The solution implemented was for OIS swaps to calculate the actual expiry including the roll convention
    as well as an unadjusted expiry from which the rest of the dates can be calculated from.
    """

    def setUp(self):
        """_summary_"""
        # set directory and file name for Input Quotes
        dirName = "./notebooks/marketdata"  # "./"
        fileName = "/multi_dates.csv"  # "/inputQuotes.csv"

        df = pd.read_csv(dirName + fileName, sep=";", decimal=",")
        column_names = list(df.columns)

        self.quotes_df = df
        self.column_names = column_names

    def test_date1(self):
        """Using 2025 09 24 as a control date, to ensure the proper bootstrapping from Frontmark data example."""

        logger.debug(f"--------------------------------------------------------")
        logger.debug("test_date dependency 1 start")
        # these inputs must be given by user
        mon = "09"
        day = "24"
        year = "2025"
        date_str = f"{day}.{mon}.{year}"
        refDate = datetime(int(year), int(mon), int(day))
        holidays = _ECB()

        df = self.quotes_df.copy()

        A = df[df["Date"] == date_str]

        # df_ins = df[df["Instrument"] == "OIS"]
        # dc_df_temp = df[(df["Date"] == selected_date) & (df["Currency"] == selected_currency)  & (df["UnderlyingIndex"] == "ESTR") & (df["Instrument"] == "OIS") ]
        df_ins = df[(df["Date"] == date_str) & (df["Currency"] == "EUR") & (df["UnderlyingIndex"] == "EONIA") & (df["Instrument"] == "OIS")]

        logger.debug("CSV loaded")

        min_i = 0
        max_i = 18  # internal selection to determine problematic dates
        min_i2 = 26
        max_i2 = len(df_ins)
        # ins_spec = sfc.load_specifications_from_pd(df_ins.iloc[np.r_[min_i:max_i, min_i2:max_i2]], refDate, holidays)
        # ins_quotes = df_ins["Quote"].tolist()[min_i:max_i]
        # ins_quotes = df_ins["Quote"].tolist()[min_i:max_i] + df_ins["Quote"].tolist()[min_i2:max_i2]

        ins_spec = sfc.load_specifications_from_pd(df_ins, refDate, holidays)
        ins_quotes = df_ins["Quote"].tolist()
        logger.debug("instrument specifications created")

        # print("--------------DEBUG PARSING")
        # print(ins_quotes[0])
        # print(df_ins["DayCountFixed"].tolist()[0])
        # print(df_ins.iloc[min_i:max_i].copy())
        # print(len(ins_spec), len(ins_quotes))
        # print(len(df_ins["Quote"].tolist()))
        # print(len(df_ins.iloc[np.r_[min_i:max_i, min_i2:max_i2]]))
        # for i in range(len(ins_quotes)):
        #     print(i, ins_quotes[i])

        print("--------------Starting bootstrapper")
        logger.debug(f"starting bootstrapper of {len(ins_quotes)} instruments")
        curve = bootstrap_curve(
            ref_date=refDate,
            curve_id="OIS_estr",
            day_count_convention=df_ins["DayCountFixed"].tolist()[0],  # taken the first entry and assume is valid for all other deposits
            instruments=ins_spec,
            quotes=ins_quotes,
            interpolation_type=InterpolationType.LINEAR_LOG,
            extrapolation_type=ExtrapolationType.LINEAR_LOG,
            # interpolation_type=InterpolationType.HAGAN_DF,
            # extrapolation_type=ExtrapolationType.CONSTANT_DF,
        )
        # print(curve.get_dates())
        self.assertIsInstance(curve, DiscountCurve)
        self.assertEqual(curve.get_dates()[0], refDate)
        logger.debug("bootstrapped curve dates matched")
        # the discount curve needs to be able to get the same market quote for the instrument
        for i in range(len(ins_spec)):
            model_quote = get_quote(refDate, ins_spec[i], {"discount_curve": curve, "fixing_curve": curve})
            self.assertAlmostEqual(model_quote, ins_quotes[i], delta=1e-5)  # since the quotes are only to 5 decimals
            # per_diff = (model_quote - deposit_quotes[i]) / deposit_quotes[i] * 100
            # print(f"model: {model_quote} market: {deposit_quotes[i]} perdiff: {per_diff}")
            # print(i, model_quote, curve.get_df()[i])

        logger.debug(f"asserted market quote matched -done")
        logger.debug(f"--------------------------------------------------------")
        # self.assertEqual(1, 1)

    def test_date_many(self):
        """Test over all OIS, and EUR instruments for every available date in the input data set"""

        logger.debug(f"--------------------------------------------------------")
        logger.debug("Starting loop, performming bootstrap over all OIS instruments, EUR, for each unique date.")
        # these inputs must be given by user
        # mon = "09"
        # day = "24"
        # year = "2025"
        # date_str = f"{day}.{mon}.{year}"
        # refDate = datetime(int(year), int(mon), int(day))
        holidays = _ECB()

        df = self.quotes_df.copy()

        eur_ois = df[(df["Currency"] == "EUR") & (df["Instrument"] == "OIS")]

        for date, subset in eur_ois.groupby("Date"):
            logger.debug(f"--------------------------------------------------------")
            logger.debug(f"Processing {date}...")
            print(subset)
            day = date.split(".")[0]
            mon = date.split(".")[1]
            year = date.split(".")[2]
            refDate = datetime(int(year), int(mon), int(day))
            logger.debug(f"--------------------------------------------------------")

            df_ins = subset
            logger.debug("CSV loaded")

            ins_spec = sfc.load_specifications_from_pd(df_ins, refDate, holidays)
            ins_quotes = df_ins["Quote"].tolist()
            logger.debug("instrument specifications created")

            # print("--------------DEBUG PARSING")
            # print(ins_quotes[0])
            # print(df_ins["DayCountFixed"].tolist()[0])
            # print(df_ins.iloc[min_i:max_i].copy())
            # print(len(ins_spec), len(ins_quotes))
            # print(len(df_ins["Quote"].tolist()))
            # print(len(df_ins.iloc[np.r_[min_i:max_i, min_i2:max_i2]]))
            # for i in range(len(ins_quotes)):
            #     print(i, ins_quotes[i])

            print("--------------Starting bootstrapper")
            logger.debug(f"starting bootstrapper of {len(ins_quotes)} instruments")
            curve = bootstrap_curve(
                ref_date=refDate,
                curve_id="OIS_estr",
                day_count_convention=df_ins["DayCountFixed"].tolist()[0],  # taken the first entry and assume is valid for all other deposits
                instruments=ins_spec,
                quotes=ins_quotes,
                # interpolation_type=InterpolationType.HAGAN_DF,
                # extrapolation_type=ExtrapolationType.CONSTANT_DF,
                interpolation_type=InterpolationType.LINEAR_LOG,
                extrapolation_type=ExtrapolationType.LINEAR_LOG,
            )
            # print(curve.get_dates())
            self.assertIsInstance(curve, DiscountCurve)
            self.assertEqual(curve.get_dates()[0], refDate)
            logger.debug("bootstrapped curve dates matched")
            # the discount curve needs to be able to get the same market quote for the instrument
            for i in range(len(ins_spec)):
                model_quote = get_quote(refDate, ins_spec[i], {"discount_curve": curve, "fixing_curve": curve})
                self.assertAlmostEqual(
                    model_quote, ins_quotes[i], delta=tolerance_from_quote(ins_quotes[i])
                )  # we adjust to check up to decimal fo the given market quote
                # per_diff = (model_quote - deposit_quotes[i]) / deposit_quotes[i] * 100
                # print(f"model: {model_quote} market: {deposit_quotes[i]} perdiff: {per_diff}")

            logger.debug(f"asserted market quote matched -done")
            logger.debug(f"--------------------------------------------------------")
            # self.assertEqual(1, 1)

        logger.debug(f"All unique datets -done")
        logger.debug(f"--------------------------------------------------------")


if __name__ == "__main__":
    # Open a file for capturing output
    with open("test_output.txt", "w") as f:
        # Save original stdout
        original_stdout = sys.stdout
        sys.stdout = f
        # Run your tests
        unittest.main(argv=["first-arg-is-ignored"], exit=False)

        # Restore stdout
        sys.stdout = original_stdout
    # unittest.main()
