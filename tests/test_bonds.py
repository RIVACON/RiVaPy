from unittest import main, TestCase
from rivapy.tools.datetools import DayCounter, Schedule, _term_to_period
from rivapy.marketdata.fixing_table import FixingTable
from rivapy.instruments.bond_specifications import (
    ZeroBondSpecification,
    FixedRateBondSpecification,
    FloatingRateBondSpecification,
)
from rivapy.tools.enums import RollConvention, SecuritizationLevel, DayCounterType, Currency
from rivapy.pricing import DeterministicCashflowPricer, price
from rivapy.marketdata import DiscountCurveParametrized, ConstantRate
from datetime import date, datetime


class BondSpecificationTests(TestCase):

    def test_bond_specification(self):
        # zero coupon bond
        zero_coupon_bond = ZeroBondSpecification(
            obj_id="US500769CH58",
            issue_price=85.0,
            issue_date=datetime(2007, 6, 29),
            maturity_date=datetime(2037, 6, 29),
            currency="USD",
            notional=1000,
            issuer="KfW",
            securitization_level=SecuritizationLevel.SENIOR_UNSECURED,
        )

        self.assertEqual(zero_coupon_bond.obj_id, "US500769CH58")
        self.assertEqual(zero_coupon_bond.issue_date, datetime(2007, 6, 29))
        self.assertEqual(zero_coupon_bond.maturity_date, datetime(2037, 6, 29))
        self.assertEqual(zero_coupon_bond.currency, "USD")
        self.assertEqual(zero_coupon_bond.notional_amount(), 1000)
        self.assertEqual(zero_coupon_bond.issuer, "KfW")
        self.assertEqual(zero_coupon_bond.securitization_level, "SENIOR_UNSECURED")
        self.assertEqual(zero_coupon_bond.issue_price, 85.0)

        # fixed rate bond
        fixed_rate_bond = FixedRateBondSpecification(
            obj_id="DE000CZ40NT7",
            issue_date=datetime(2019, 3, 11),
            maturity_date=datetime(2024, 9, 11),
            coupon=0.0125,
            frequency="1Y",
            business_day_convention=RollConvention.FOLLOWING,
            currency="EUR",
            notional=100000,
            issuer="Commerzbank",
            securitization_level=SecuritizationLevel.NON_PREFERRED_SENIOR,
            stub_type_is_Long=False,
        )
        self.assertEqual(fixed_rate_bond.obj_id, "DE000CZ40NT7")
        self.assertEqual(fixed_rate_bond.issue_date, datetime(2019, 3, 11))
        self.assertEqual(fixed_rate_bond.maturity_date, datetime(2024, 9, 11))
        self.assertEqual(
            fixed_rate_bond.get_schedule().generate_dates(True),
            [
                datetime(2019, 9, 11),
                datetime(2020, 9, 11),
                datetime(2021, 9, 13),
                datetime(2022, 9, 12),
                datetime(2023, 9, 11),
                datetime(2024, 9, 11),
            ],
        )
        # self.assertEqual(fixed_rate_bond.coupons, [0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125])
        self.assertEqual(fixed_rate_bond.currency, "EUR")
        self.assertEqual(fixed_rate_bond.notional_amount(), 100000)
        self.assertEqual(fixed_rate_bond.issuer, "Commerzbank")
        self.assertEqual(fixed_rate_bond.securitization_level, "NON_PREFERRED_SENIOR")
        fixed_rate_bond = FixedRateBondSpecification(
            obj_id="DE000CZ40NT7",
            issue_date=datetime(2019, 3, 11),
            maturity_date=datetime(2024, 9, 11),
            frequency="1Y",
            coupon=0.0125,
            currency="EUR",
            notional=100000,
            issuer="Commerzbank",
            securitization_level=SecuritizationLevel.NON_PREFERRED_SENIOR,
            stub_type_is_Long=False,
        )
        self.assertEqual(fixed_rate_bond.obj_id, "DE000CZ40NT7")
        self.assertEqual(fixed_rate_bond.issue_date, datetime(2019, 3, 11))
        self.assertEqual(fixed_rate_bond.maturity_date, datetime(2024, 9, 11))
        self.assertEqual(
            fixed_rate_bond.get_schedule().generate_dates(True),
            [
                datetime(2019, 9, 11),
                datetime(2020, 9, 11),
                datetime(2021, 9, 13),
                datetime(2022, 9, 12),
                datetime(2023, 9, 11),
                datetime(2024, 9, 11),
            ],
        )
        self.assertEqual(fixed_rate_bond.coupon, 0.0125)
        self.assertEqual(fixed_rate_bond.currency, "EUR")
        self.assertEqual(fixed_rate_bond.notional_amount(), 100000)
        self.assertEqual(fixed_rate_bond.issuer, "Commerzbank")
        self.assertEqual(fixed_rate_bond.securitization_level, "NON_PREFERRED_SENIOR")

        fixed_rate_bond_amort = FixedRateBondSpecification(
            obj_id="DE000CZ40NT7",
            issue_date=datetime(2019, 3, 11),
            maturity_date=datetime(2024, 9, 11),
            frequency="1Y",
            coupon=0.0125,
            currency="EUR",
            notional=100000,
            amortization_scheme="linear",
            issuer="Commerzbank",
            securitization_level=SecuritizationLevel.NON_PREFERRED_SENIOR,
            stub_type_is_Long=False,
        )
        self.assertEqual(fixed_rate_bond_amort.notional_amount(), 100000)
        self.assertEqual(fixed_rate_bond_amort._notional.get_amount(), 100000)
        self.assertEqual(fixed_rate_bond_amort._amortization_scheme.get_total_amortization(), 100.0)
        self.assertEqual(fixed_rate_bond_amort._amortization_scheme.n_steps, 1)
        self.assertEqual(fixed_rate_bond_amort._notional.get_size(), 1)
        self.assertEqual(fixed_rate_bond_amort._notional.get_amortizations_by_index(), [(1, 100000.0)])

        fixed_rate_bond_amort2 = FixedRateBondSpecification(
            obj_id="DE000CZ40NT7",
            issue_date=datetime(2019, 3, 11),
            maturity_date=datetime(2024, 9, 11),
            frequency="1Y",
            coupon=0.0125,
            currency="EUR",
            notional=100000,
            amortization_scheme="linear",
            issuer="Commerzbank",
            securitization_level=SecuritizationLevel.NON_PREFERRED_SENIOR,
            stub_type_is_Long=False,
        )
        fixed_rate_bond_amort2._notional.n_steps = 5
        self.assertEqual(fixed_rate_bond_amort2.notional_amount(), 100000)
        self.assertEqual(fixed_rate_bond_amort2._notional.get_amount(), 100000)
        self.assertEqual(fixed_rate_bond_amort2._amortization_scheme.get_total_amortization(), 100.0)
        self.assertEqual(fixed_rate_bond_amort2._amortization_scheme.n_steps, 1)
        self.assertEqual(fixed_rate_bond_amort2._notional.get_size(), 5)
        self.assertEqual(fixed_rate_bond_amort2._notional.get_amortizations_by_index(), [(1, 25000.0), (2, 25000.0), (3, 25000.0), (4, 25000.0)])

        # floating rate bond
        floating_rate_note = FloatingRateBondSpecification(
            obj_id="DE000HLB3DU1",
            issue_date=datetime(2016, 6, 23),
            maturity_date=datetime(2024, 6, 27),
            frequency="3M",
            adjust_start_date=False,
            adjust_end_date=False,
            business_day_convention=RollConvention.FOLLOWING,
            day_count_convention=DayCounterType.ThirtyU360,
            margin=0.0,
            index="EURIBOR_3M",
            currency="EUR",
            notional=1000,
            issuer="Helaba",
            securitization_level=SecuritizationLevel.NON_PREFERRED_SENIOR,
        )
        self.assertEqual(floating_rate_note.obj_id, "DE000HLB3DU1")
        self.assertEqual(floating_rate_note.issue_date, datetime(2016, 6, 23))
        self.assertEqual(floating_rate_note.maturity_date, datetime(2024, 6, 27))
        # Coupon period dates including start of first accrual period
        self.assertEqual(
            Schedule._roll_out(
                floating_rate_note.maturity_date,
                floating_rate_note.start_date,
                floating_rate_note._frequency,
                backwards=True,
            ),  # reverse to get in correct order
            [
                datetime(2016, 6, 23),
                datetime(2016, 9, 27),
                datetime(2016, 12, 27),
                datetime(2017, 3, 27),
                datetime(2017, 6, 27),
                datetime(2017, 9, 27),
                datetime(2017, 12, 27),
                datetime(2018, 3, 27),
                datetime(2018, 6, 27),
                datetime(2018, 9, 27),
                datetime(2018, 12, 27),
                datetime(2019, 3, 27),
                datetime(2019, 6, 27),
                datetime(2019, 9, 27),
                datetime(2019, 12, 27),
                datetime(2020, 3, 27),
                datetime(2020, 6, 27),
                datetime(2020, 9, 27),
                datetime(2020, 12, 27),
                datetime(2021, 3, 27),
                datetime(2021, 6, 27),
                datetime(2021, 9, 27),
                datetime(2021, 12, 27),
                datetime(2022, 3, 27),
                datetime(2022, 6, 27),
                datetime(2022, 9, 27),
                datetime(2022, 12, 27),
                datetime(2023, 3, 27),
                datetime(2023, 6, 27),
                datetime(2023, 9, 27),
                datetime(2023, 12, 27),
                datetime(2024, 3, 27),
                datetime(2024, 6, 27),
            ],
        )
        # Coupon payment dates
        self.assertEqual(
            floating_rate_note.get_schedule().generate_dates(True),
            [
                datetime(2016, 9, 27),
                datetime(2016, 12, 27),
                datetime(2017, 3, 27),
                datetime(2017, 6, 27),
                datetime(2017, 9, 27),
                datetime(2017, 12, 27),
                datetime(2018, 3, 27),
                datetime(2018, 6, 27),
                datetime(2018, 9, 27),
                datetime(2018, 12, 27),
                datetime(2019, 3, 27),
                datetime(2019, 6, 27),
                datetime(2019, 9, 27),
                datetime(2019, 12, 27),
                datetime(2020, 3, 27),
                datetime(2020, 6, 29),
                datetime(2020, 9, 28),
                datetime(2020, 12, 28),
                datetime(2021, 3, 29),
                datetime(2021, 6, 28),
                datetime(2021, 9, 27),
                datetime(2021, 12, 27),
                datetime(2022, 3, 28),
                datetime(2022, 6, 27),
                datetime(2022, 9, 27),
                datetime(2022, 12, 27),
                datetime(2023, 3, 27),
                datetime(2023, 6, 27),
                datetime(2023, 9, 27),
                datetime(2023, 12, 27),
                datetime(2024, 3, 27),
                datetime(2024, 6, 27),
            ],
        )

        self.assertEqual(floating_rate_note._day_count_convention.value, "30U360")
        self.assertEqual(floating_rate_note._margin, 0.0)
        self.assertEqual(floating_rate_note._index, "EURIBOR_3M")
        if floating_rate_note._ir_index is not None:
            self.assertEqual(floating_rate_note._ir_index.value.name, "EURIBOR 3M")
        else:
            print("IR index is None.")
        self.assertEqual(floating_rate_note.currency, "EUR")
        self.assertEqual(floating_rate_note.notional_amount(), 1000)
        self.assertEqual(floating_rate_note.issuer, "Helaba")
        self.assertEqual(floating_rate_note.securitization_level, "NON_PREFERRED_SENIOR")
        floating_rate_note = FloatingRateBondSpecification(
            obj_id="DE000HLB3DU1",
            issue_date=datetime(2016, 6, 23),
            maturity_date=datetime(2024, 6, 27),
            # coupon_period_dates=floating_rate_note.coupon_period_dates,
            day_count_convention=DayCounterType.ThirtyU360,
            margin=floating_rate_note._margin,
            index="EURIBOR_3M",
            currency="EUR",
            notional=1000,
            issuer="Helaba",
            securitization_level=SecuritizationLevel.NON_PREFERRED_SENIOR,
        )
        self.assertEqual(floating_rate_note.obj_id, "DE000HLB3DU1")
        self.assertEqual(floating_rate_note.issue_date, datetime(2016, 6, 23))
        self.assertEqual(floating_rate_note.maturity_date, datetime(2024, 6, 27))
        self.assertEqual(
            floating_rate_note.get_schedule().generate_dates(False),
            [
                datetime(2016, 6, 23),
                datetime(2016, 9, 27),
                datetime(2016, 12, 27),
                datetime(2017, 3, 27),
                datetime(2017, 6, 27),
                datetime(2017, 9, 27),
                datetime(2017, 12, 27),
                datetime(2018, 3, 27),
                datetime(2018, 6, 27),
                datetime(2018, 9, 27),
                datetime(2018, 12, 27),
                datetime(2019, 3, 27),
                datetime(2019, 6, 27),
                datetime(2019, 9, 27),
                datetime(2019, 12, 27),
                datetime(2020, 3, 27),
                datetime(2020, 6, 29),
                datetime(2020, 9, 28),
                datetime(2020, 12, 28),
                datetime(2021, 3, 29),
                datetime(2021, 6, 28),
                datetime(2021, 9, 27),
                datetime(2021, 12, 27),
                datetime(2022, 3, 28),
                datetime(2022, 6, 27),
                datetime(2022, 9, 27),
                datetime(2022, 12, 27),
                datetime(2023, 3, 27),
                datetime(2023, 6, 27),
                datetime(2023, 9, 27),
                datetime(2023, 12, 27),
                datetime(2024, 3, 27),
                datetime(2024, 6, 27),
            ],
        )
        self.assertEqual(floating_rate_note._day_count_convention.value, "30U360")
        self.assertEqual(floating_rate_note._ir_index.value.name, "EURIBOR 3M")
        self.assertEqual(floating_rate_note.currency, "EUR")
        self.assertEqual(floating_rate_note.notional_amount(), 1000)
        self.assertEqual(floating_rate_note.issuer, "Helaba")
        self.assertEqual(floating_rate_note.securitization_level, "NON_PREFERRED_SENIOR")

        # fixed-to-floating rate note
        # if False:
        #     # not correctly working
        #     fixed_to_floating_rate_note = FixedToFloatingRateNote.from_master_data('XS1887493309', datetime(2018, 10, 4),
        #                                                                         datetime(2022, 1, 20), datetime(2023, 1, 20),
        #                                                                         0.04247, '6M', '3M', True, True, True,
        #                                                                         False, RollConvention.MODIFIED_FOLLOWING,
        #                                                                         RollConvention.MODIFIED_FOLLOWING, 'DE',
        #                                                                         'DE', DayCounterType.ThirtyU360, 0.0115,
        #                                                                         'US_LIBOR_3M', 'USD', 1000000,
        #                                                                         'Standard Chartered PLC',
        #                                                                         SecuritizationLevel.SENIOR_SECURED)
        #     self.assertEqual(fixed_to_floating_rate_note.obj_id, 'XS1887493309')
        #     # self.assertEqual(fixed_to_floating_rate_note.issue_date, datetime(2018, 10, 4))
        #     self.assertEqual(fixed_to_floating_rate_note.maturity_date, datetime(2023, 1, 20))
        #     self.assertEqual(fixed_to_floating_rate_note.coupon_payment_dates, [datetime(2019, 1, 21), datetime(2019, 7, 22),
        #                                                                         datetime(2020, 1, 20), datetime(2020, 7, 20),
        #                                                                         datetime(2021, 1, 20), datetime(2021, 7, 20),
        #                                                                         datetime(2022, 1, 20)])
        #     self.assertEqual(fixed_to_floating_rate_note.coupons, [0.04247, 0.04247, 0.04247, 0.04247, 0.04247, 0.04247,
        #                                                         0.04247])
        #     self.assertEqual(fixed_to_floating_rate_note.coupon_period_dates, [datetime(2022, 1, 20), datetime(2022, 4, 20),
        #                                                                     datetime(2022, 7, 20), datetime(2022, 10, 20),
        #                                                                     datetime(2023, 1, 20)])
        #     self.assertEqual(fixed_to_floating_rate_note.day_count_convention, '30U360')
        #     self.assertEqual(fixed_to_floating_rate_note.spreads, [0.0115, 0.0115, 0.0115, 0.0115])
        #     self.assertEqual(fixed_to_floating_rate_note.currency, 'USD')
        #     self.assertEqual(fixed_to_floating_rate_note.notional, 1000000)
        #     self.assertEqual(fixed_to_floating_rate_note.issuer, 'Standard Chartered PLC')
        #     self.assertEqual(fixed_to_floating_rate_note.securitization_level, 'SENIOR_SECURED')


class BondPricingTests(TestCase):

    def test_yield_bond(self):
        """Simple test for yield computation: Fixed rate used to compute pv which is then used to compute yield."""
        ref_date = datetime(2023, 5, 1)
        bond_spec = FixedRateBondSpecification(
            "PV_BOND",
            issue_date=datetime(2023, 1, 1),
            maturity_date=datetime(2025, 1, 2),
            currency=Currency.EUR,
            notional=100.0,
            issuer="None",
            securitization_level=SecuritizationLevel.SUBORDINATED,
            coupon=0.05,
            frequency="1Y",
        )
        dc = DiscountCurveParametrized("", ref_date, ConstantRate(0.05))
        nr_annual_payments = bond_spec.get_nr_annual_payments()
        self.assertEqual(nr_annual_payments, 1)
        # schedule for accrual periods rolled out
        dates = bond_spec.get_schedule()._roll_out(
            from_=bond_spec._start_date if not bond_spec._backwards else bond_spec._end_date,
            to_=bond_spec._end_date if not bond_spec._backwards else bond_spec._start_date,
            term=_term_to_period(bond_spec._frequency),
            long_stub=bond_spec._stub_type_is_Long,
            backwards=bond_spec._backwards,
        )
        ## discount factors

        df1 = dc.value(ref_date, datetime(2024, 1, 2))
        df2 = dc.value(ref_date, datetime(2025, 1, 2))
        dcc = DayCounter(bond_spec.day_count_convention)
        self.assertAlmostEqual(df1, 0.9668628440577077, delta=1e-6)
        self.assertAlmostEqual(df2, 0.9195824079027841, delta=1e-6)
        price = df1 * 5.0 + df2 * 105.0
        self.assertAlmostEqual(DeterministicCashflowPricer.get_pv_cashflows(ref_date, bond_spec, dc), price, delta=1e-6)

        self.assertEqual(dates, [datetime(2023, 1, 2), datetime(2024, 1, 2), datetime(2025, 1, 2)])
        self.assertEqual(
            DeterministicCashflowPricer.get_expected_cashflows(bond_spec),
            [(datetime(2023, 1, 2), -100.0), (datetime(2024, 1, 2), 5.0), (datetime(2025, 1, 2), 5.0), (datetime(2025, 1, 2), 100.0)],
        )
        bond_price = DeterministicCashflowPricer.get_pv_cashflows(ref_date, bond_spec, dc)
        bond_yield = DeterministicCashflowPricer.get_compute_yield(target_dirty_price=bond_price, val_date=ref_date, specification=bond_spec)
        self.assertAlmostEqual(bond_yield, 0.05, delta=1e-3)
        bpricer = DeterministicCashflowPricer(ref_date, bond_spec, dc)
        self.assertAlmostEqual(bpricer.compute_yield(bond_price), 0.05, delta=1e-3)
        mc_duration = (
            dcc.yf(ref_date, datetime(2024, 1, 2), bond_spec.dates, bond_spec.nr_annual_payments) * df1 * 5.0 / bond_price
            + dcc.yf(ref_date, datetime(2025, 1, 2), bond_spec.dates, bond_spec.nr_annual_payments) * df2 * 105.0 / bond_price
        )
        self.assertAlmostEqual(bpricer.macaulay_duration(), mc_duration, delta=1e-3)
        self.assertAlmostEqual(bpricer.z_spread(bond_price), 0.0, delta=1e-6)
        self.assertAlmostEqual(bpricer.macaulay_duration(), mc_duration, delta=1e-3)
        self.assertAlmostEqual(bpricer.modified_duration(bond_price), mc_duration / 1.05, delta=1e-3)

    def test_float_bond(self):
        ref_date = datetime(2023, 5, 1)
        fix_table = FixingTable("PV_BOND")
        fix_table.add("1Y", datetime(2022, 12, 30), 0.05)
        float_bond_spec = FloatingRateBondSpecification(
            "PV_BOND",
            issue_date=datetime(2023, 1, 1),
            maturity_date=datetime(2025, 1, 2),
            currency=Currency.EUR,
            notional=100.0,
            issuer="None",
            securitization_level=SecuritizationLevel.SUBORDINATED,
            day_count_convention="ACTACTICMA",
            business_day_convention="ModifiedFollowing",
            margin=0.00,
            frequency="1Y",
            fixings=fix_table,
        )
        dc = DiscountCurveParametrized("", ref_date, ConstantRate(0.05))
        nr_annual_payments = float_bond_spec.get_nr_annual_payments()
        self.assertEqual(nr_annual_payments, 1)
        dates = float_bond_spec.get_schedule()._roll_out(
            from_=float_bond_spec._start_date if not float_bond_spec._backwards else float_bond_spec._end_date,
            to_=float_bond_spec._end_date if not float_bond_spec._backwards else float_bond_spec._start_date,
            term=_term_to_period(float_bond_spec._frequency),
            long_stub=float_bond_spec._stub_type_is_Long,
            backwards=float_bond_spec._backwards,
        )
        ## discount factors

        df1 = dc.value(ref_date, datetime(2024, 1, 2))
        df2 = dc.value(ref_date, datetime(2025, 1, 2))
        dcc = DayCounter(float_bond_spec.day_count_convention)
        self.assertAlmostEqual(df1, 0.9668628440577077, delta=1e-6)
        self.assertAlmostEqual(df2, 0.9195824079027841, delta=1e-6)
        price = df1 * 5.0 + df2 * 105.0
        print(DeterministicCashflowPricer.get_expected_cashflows(float_bond_spec, ref_date, dc))
        self.assertEqual(dates, [datetime(2023, 1, 2), datetime(2024, 1, 2), datetime(2025, 1, 2)])
        expected = [
            (datetime(2023, 1, 2), -100.0),
            (datetime(2024, 1, 2), 5.0),
            (datetime(2025, 1, 2), 5.0),
            (datetime(2025, 1, 2), 100.0),
        ]
        actual = DeterministicCashflowPricer.get_expected_cashflows(float_bond_spec, ref_date, dc)

        self.assertEqual(len(actual), len(expected))
        for i, (exp, act) in enumerate(zip(expected, actual)):
            with self.subTest(i=i):
                # exact equality for the date
                self.assertEqual(exp[0], act[0])
                # almost-equal for the amount with a tolerance (choose places or delta)
                self.assertAlmostEqual(exp[1], act[1], places=6)
        self.assertAlmostEqual(DeterministicCashflowPricer.get_pv_cashflows(ref_date, float_bond_spec, dc, dc), price, delta=1e-6)
        bond_yield = DeterministicCashflowPricer.get_compute_yield(
            target_dirty_price=price, val_date=ref_date, specification=float_bond_spec, fwd_curve=dc
        )
        self.assertAlmostEqual(bond_yield, 0.05, delta=1e-3)
        bpricer = DeterministicCashflowPricer(ref_date, float_bond_spec, dc, dc)
        self.assertAlmostEqual(bpricer.compute_yield(price), 0.05, delta=1e-3)
        mc_duration = (
            dcc.yf(ref_date, datetime(2024, 1, 2), float_bond_spec.dates, float_bond_spec.nr_annual_payments) * df1 * 5.0 / price
            + dcc.yf(ref_date, datetime(2025, 1, 2), float_bond_spec.dates, float_bond_spec.nr_annual_payments) * df2 * 105.0 / price
        )
        self.assertAlmostEqual(bpricer.macaulay_duration(), mc_duration, delta=1e-3)
        self.assertAlmostEqual(bpricer.z_spread(price), 0.0, delta=1e-6)
        self.assertAlmostEqual(bpricer.macaulay_duration(), mc_duration, delta=1e-3)
        self.assertAlmostEqual(bpricer.modified_duration(price), mc_duration / 1.05, delta=1e-3)
        print(bpricer.get_accrued_interest(float_bond_spec, ref_date, dc))

    def test_index_float_bond(self):
        ref_date = datetime(2023, 5, 1)
        fix_table = FixingTable("PV_BOND")
        fix_table.add("EURIBOR 12M", datetime(2022, 12, 30), 0.05)
        float_index_bond_spec = FloatingRateBondSpecification(
            "PV_BOND",
            issue_date=datetime(2023, 1, 1),
            maturity_date=datetime(2025, 1, 2),
            currency=Currency.EUR,
            notional=100.0,
            issuer="None",
            securitization_level=SecuritizationLevel.SUBORDINATED,
            day_count_convention="ACTACTICMA",
            business_day_convention="ModifiedFollowing",
            margin=0.00,
            index="EURIBOR_1Y",
            frequency="1Y",
            fixings=fix_table,
            adjust_accruals=False,
            adjust_schedule=False,
        )
        self.assertEqual(float_index_bond_spec._index, "EURIBOR_1Y")
        self.assertEqual(float_index_bond_spec._ir_index.value.name, "EURIBOR 12M")
        dc = DiscountCurveParametrized("", ref_date, ConstantRate(0.05))
        nr_annual_payments = float_index_bond_spec.get_nr_annual_payments()
        self.assertEqual(nr_annual_payments, 1)
        dates = float_index_bond_spec.get_schedule()._roll_out(
            from_=float_index_bond_spec._start_date if not float_index_bond_spec._backwards else float_index_bond_spec._end_date,
            to_=float_index_bond_spec._end_date if not float_index_bond_spec._backwards else float_index_bond_spec._start_date,
            term=_term_to_period(float_index_bond_spec._frequency),
            long_stub=float_index_bond_spec._stub_type_is_Long,
            backwards=float_index_bond_spec._backwards,
        )
        ## discount factors

        df1 = dc.value(ref_date, datetime(2024, 1, 2))
        df2 = dc.value(ref_date, datetime(2025, 1, 2))
        dcc = DayCounter(float_index_bond_spec.day_count_convention)
        self.assertAlmostEqual(df1, 0.9668628440577077, delta=1e-6)
        self.assertAlmostEqual(df2, 0.9195824079027841, delta=1e-6)
        price = df1 * 5.0 + df2 * 105.0
        self.assertEqual(dates, [datetime(2023, 1, 2), datetime(2024, 1, 2), datetime(2025, 1, 2)])
        expected = [
            (datetime(2023, 1, 2), -100.0),
            (datetime(2024, 1, 2), 5.0),
            (datetime(2025, 1, 2), 5.0),
            (datetime(2025, 1, 2), 100.0),
        ]
        actual = DeterministicCashflowPricer.get_expected_cashflows(float_index_bond_spec, ref_date, dc)

        self.assertEqual(len(actual), len(expected))
        for i, (exp, act) in enumerate(zip(expected, actual)):
            with self.subTest(i=i):
                # exact equality for the date
                self.assertEqual(exp[0], act[0])
                # almost-equal for the amount with a tolerance (choose places or delta)
                self.assertAlmostEqual(exp[1], act[1], places=6)
        self.assertAlmostEqual(DeterministicCashflowPricer.get_pv_cashflows(ref_date, float_index_bond_spec, dc, dc), price, delta=1e-6)
        bond_yield = DeterministicCashflowPricer.get_compute_yield(
            target_dirty_price=price, val_date=ref_date, specification=float_index_bond_spec, fwd_curve=dc
        )
        self.assertAlmostEqual(bond_yield, 0.05, delta=1e-3)
        bpricer = DeterministicCashflowPricer(ref_date, float_index_bond_spec, dc, dc)
        self.assertAlmostEqual(bpricer.compute_yield(price), 0.05, delta=1e-3)
        mc_duration = (
            dcc.yf(ref_date, datetime(2024, 1, 2), float_index_bond_spec.dates, float_index_bond_spec.nr_annual_payments) * df1 * 5.0 / price
            + dcc.yf(ref_date, datetime(2025, 1, 2), float_index_bond_spec.dates, float_index_bond_spec.nr_annual_payments) * df2 * 105.0 / price
        )
        self.assertAlmostEqual(bpricer.macaulay_duration(), mc_duration, delta=1e-3)
        self.assertAlmostEqual(bpricer.z_spread(price), 0.0, delta=1e-6)
        self.assertAlmostEqual(bpricer.macaulay_duration(), mc_duration, delta=1e-3)
        self.assertAlmostEqual(bpricer.modified_duration(price), mc_duration / 1.05, delta=1e-3)

    def test_amort_bond(self):
        """Simple test for yield computation: Fixed rate used to compute pv which is then used to compute yield."""
        ref_date = datetime(2023, 5, 1)
        bond_amort_spec = FixedRateBondSpecification(
            "PV_BOND",
            issue_date=datetime(2023, 1, 1),
            maturity_date=datetime(2025, 1, 2),
            currency=Currency.EUR,
            notional=100.0,
            amortization_scheme="linear",
            issuer="None",
            securitization_level=SecuritizationLevel.SUBORDINATED,
            coupon=0.05,
            frequency="1Y",
        )
        dc = DiscountCurveParametrized("", ref_date, ConstantRate(0.05))
        nr_annual_payments = bond_amort_spec.get_nr_annual_payments()
        self.assertEqual(nr_annual_payments, 1)
        # schedule for accrual periods rolled out
        dates = bond_amort_spec.get_schedule()._roll_out(
            from_=bond_amort_spec._start_date if not bond_amort_spec._backwards else bond_amort_spec._end_date,
            to_=bond_amort_spec._end_date if not bond_amort_spec._backwards else bond_amort_spec._start_date,
            term=_term_to_period(bond_amort_spec._frequency),
            long_stub=bond_amort_spec._stub_type_is_Long,
            backwards=bond_amort_spec._backwards,
        )
        # print(bond_amort_spec._notional.notional)

        ## discount factors

        df1 = dc.value(ref_date, datetime(2024, 1, 2))
        df2 = dc.value(ref_date, datetime(2025, 1, 2))
        dcc = DayCounter(bond_amort_spec.day_count_convention)
        self.assertAlmostEqual(df1, 0.9668628440577077, delta=1e-6)
        self.assertAlmostEqual(df2, 0.9195824079027841, delta=1e-6)
        price = df1 * 55.0 + df2 * 52.5
        self.assertAlmostEqual(DeterministicCashflowPricer.get_pv_cashflows(ref_date, bond_amort_spec, dc), price, delta=1e-6)

        self.assertEqual(dates, [datetime(2023, 1, 2), datetime(2024, 1, 2), datetime(2025, 1, 2)])
        # print(DeterministicCashflowPricer.get_expected_cashflows(bond_amort_spec))
        self.assertEqual(
            DeterministicCashflowPricer.get_expected_cashflows(bond_amort_spec),
            [
                (datetime(2023, 1, 2), -100.0),
                (datetime(2024, 1, 2), 5.0),
                (datetime(2024, 1, 2), 50.0),
                (datetime(2025, 1, 2), 2.5),
                (datetime(2025, 1, 2), 50.0),
            ],
        )
        bond_price = DeterministicCashflowPricer.get_pv_cashflows(ref_date, bond_amort_spec, dc)
        bond_yield = DeterministicCashflowPricer.get_compute_yield(target_dirty_price=bond_price, val_date=ref_date, specification=bond_amort_spec)
        self.assertAlmostEqual(bond_yield, 0.05, delta=1e-3)
        bpricer = DeterministicCashflowPricer(ref_date, bond_amort_spec, dc)
        self.assertAlmostEqual(bpricer.compute_yield(bond_price), 0.05, delta=1e-3)
        mc_duration = (
            dcc.yf(ref_date, datetime(2024, 1, 2), bond_amort_spec.dates, bond_amort_spec.nr_annual_payments) * df1 * 55.0 / bond_price
            + dcc.yf(ref_date, datetime(2025, 1, 2), bond_amort_spec.dates, bond_amort_spec.nr_annual_payments) * df2 * 52.5 / bond_price
        )
        self.assertAlmostEqual(bpricer.macaulay_duration(), mc_duration, delta=1e-3)
        self.assertAlmostEqual(bpricer.z_spread(bond_price), 0.0, delta=1e-6)
        self.assertAlmostEqual(bpricer.macaulay_duration(), mc_duration, delta=1e-3)
        self.assertAlmostEqual(bpricer.modified_duration(bond_price), mc_duration / 1.05, delta=1e-3)


if __name__ == "__main__":
    main()
