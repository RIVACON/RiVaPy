from unittest import main, TestCase

from matplotlib.dates import relativedelta
from rivapy.tools.datetools import DayCounter, Schedule, _term_to_period, calc_end_day
from rivapy.tools.holidays_compat import EuropeanCentralBank as _ECB
from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.tools.enums import RollConvention, SecuritizationLevel, DayCounterType, Currency
from rivapy.pricing import DeterministicCashflowPricer, price
from rivapy.marketdata import DiscountCurveParametrized, ConstantRate
from datetime import date, datetime as dt


class DepositSpecificationTests(TestCase):

    def test_deposit_specification(self):
        ccy = "EUR"
        ref_date = dt(2023, 1, 2)
        issue_date = ref_date
        maturity_date = dt(2024, 1, 26)
        dcc = "Act360"
        rate = 0.03
        notional = 1000.0

        deposit_spec = DepositSpecification(
            obj_id="dummy_id",
            issuer="",
            currency=ccy,
            issue_date=issue_date,
            maturity_date=maturity_date,
            notional=notional,
            rate=rate,
            day_count_convention=dcc,
            roll_convention="NONE",
            payment_days=5,
            spot_days=5,
        )

        self.assertEqual(deposit_spec.obj_id, "dummy_id")
        self.assertEqual(deposit_spec.issuer, "")
        self.assertEqual(deposit_spec.notional.get_amount(), notional)
        self.assertEqual(deposit_spec.currency, ccy)
        self.assertEqual(deposit_spec.maturity_date, maturity_date)
        self.assertEqual(deposit_spec.issue_date, issue_date)
        self.assertEqual(deposit_spec.rate, rate)
        self.assertEqual(deposit_spec.day_count_convention, dcc)
        self.assertEqual(deposit_spec.roll_convention, "NONE")
        self.assertEqual(deposit_spec.payment_days, 5)
        self.assertEqual(deposit_spec.spot_days, 5)
        self.assertEqual(deposit_spec.securitization_level, "NONE")
        self.assertEqual(deposit_spec.ins_type().value, "DEPOSIT")
        self.assertEqual(deposit_spec.get_end_date(), maturity_date)
        # holiday calendars may be represented differently across `holidays` versions
        # compare a known holiday entry instead of relying on object equality
        self.assertIn(date(2023, 1, 1), deposit_spec.calendar)
        self.assertEqual(deposit_spec.calendar[date(2023, 1, 1)], _ECB()[date(2023, 1, 1)])
        self.assertEqual(deposit_spec.adjust_start_date, True)
        self.assertEqual(deposit_spec.adjust_end_date, False)
        self.assertEqual(deposit_spec.frequency, "389D")
        self.assertEqual(deposit_spec.dates, [issue_date, maturity_date])

        deposit_spec_2 = DepositSpecification(
            obj_id="dummy_id",
            issuer="",
            term="1Y",
            currency=ccy,
            issue_date=issue_date,
            notional=notional,
            rate=rate,
            day_count_convention=dcc,
            roll_convention="NONE",
        )
        maturity_date = calc_end_day(issue_date, "1Y", "MODIFIED_FOLLOWING", _ECB())
        self.assertEqual(deposit_spec_2.maturity_date, maturity_date)
        self.assertEqual(deposit_spec_2.start_date, dt(2023, 1, 2))
        self.assertEqual(deposit_spec_2.spot_days, 2)

        issue_date = dt(2023, 1, 1)
        deposit_spec_3 = DepositSpecification(
            obj_id="dummy_id",
            issuer="",
            term="1Y",
            currency=ccy,
            issue_date=issue_date,
            notional=notional,
            rate=rate,
            day_count_convention=dcc,
            roll_convention="NONE",
            adjust_start_date=False,
        )
        self.assertEqual(deposit_spec_3.issue_date, dt(2023, 1, 1))
        self.assertEqual(deposit_spec_3.start_date, dt(2023, 1, 1))
        self.assertEqual(deposit_spec_3.end_date, dt(2024, 1, 1))
        self.assertEqual(deposit_spec_3.maturity_date, dt(2024, 1, 2))

        deposit_spec_4 = DepositSpecification(
            obj_id="dummy_id",
            issuer="",
            term="1Y",
            currency=ccy,
            issue_date=issue_date,
            notional=notional,
            rate=rate,
            day_count_convention=dcc,
            roll_convention="NONE",
            adjust_start_date=True,
        )
        self.assertEqual(deposit_spec_4.issue_date, dt(2023, 1, 1))
        self.assertEqual(deposit_spec_4.start_date, dt(2023, 1, 2))
        self.assertEqual(deposit_spec_4.end_date, dt(2024, 1, 2))
        self.assertEqual(deposit_spec_4.maturity_date, dt(2024, 1, 2))

        deposit_spec_5 = DepositSpecification(
            obj_id="dummy_id",
            issuer="",
            term="O/N",
            currency=ccy,
            issue_date=issue_date,
            notional=notional,
            rate=rate,
            day_count_convention=dcc,
            roll_convention="NONE",
        )
        self.assertEqual(deposit_spec_5.maturity_date, dt(2023, 1, 3))
        self.assertEqual(deposit_spec_5.start_date, dt(2023, 1, 2))
        self.assertEqual(deposit_spec_5.end_date, dt(2023, 1, 3))
        self.assertEqual(deposit_spec_5.spot_days, 0)


# class DepositPricingTests(TestCase):

#     def test_yield_deposit(self):
#         """Simple test for yield computation: Fixed rate used to compute pv which is then used to compute yield."""
#         ref_date = datetime(2023, 5, 1)
#         bond_spec = FixedRateBondSpecification(
#             "PV_BOND",
#             issue_date=datetime(2023, 1, 1),
#             maturity_date=datetime(2025, 1, 2),
#             currency=Currency.EUR,
#             notional=100.0,
#             issuer="None",
#             securitization_level=SecuritizationLevel.SUBORDINATED,
#             coupon=0.05,
#             frequency="1Y",
#         )
#         dc = DiscountCurveParametrized("", ref_date, ConstantRate(0.05))
#         nr_annual_payments = bond_spec.get_nr_annual_payments()
#         self.assertEqual(nr_annual_payments, 1)
#         # schedule for accrual periods rolled out
#         dates = bond_spec.get_schedule()._roll_out(
#             from_=bond_spec._start_date if not bond_spec._backwards else bond_spec._end_date,
#             to_=bond_spec._end_date if not bond_spec._backwards else bond_spec._start_date,
#             term=_term_to_period(bond_spec._frequency),
#             long_stub=bond_spec._stub_type_is_Long,
#             backwards=bond_spec._backwards,
#         )
#         ## discount factors

#         df1 = dc.value(ref_date, datetime(2024, 1, 2))
#         df2 = dc.value(ref_date, datetime(2025, 1, 2))
#         dcc = DayCounter(bond_spec.day_count_convention)
#         self.assertAlmostEqual(df1, 0.9668628440577077, delta=1e-6)
#         self.assertAlmostEqual(df2, 0.9195824079027841, delta=1e-6)
#         price = df1 * 5.0 + df2 * 105.0
#         self.assertAlmostEqual(DeterministicCashflowPricer.get_pv_cashflows(ref_date, bond_spec, dc), price, delta=1e-6)

#         self.assertEqual(dates, [datetime(2023, 1, 2), datetime(2024, 1, 2), datetime(2025, 1, 2)])
#         self.assertEqual(
#             DeterministicCashflowPricer.get_expected_cashflows(bond_spec),
#             [(datetime(2023, 1, 2), -100.0), (datetime(2024, 1, 2), 5.0), (datetime(2025, 1, 2), 5.0), (datetime(2025, 1, 2), 100.0)],
#         )
#         bond_price = DeterministicCashflowPricer.get_pv_cashflows(ref_date, bond_spec, dc)
#         bond_yield = DeterministicCashflowPricer.get_compute_yield(target_dirty_price=bond_price, val_date=ref_date, specification=bond_spec)
#         self.assertAlmostEqual(bond_yield, 0.05, delta=1e-3)
#         bpricer = DeterministicCashflowPricer(ref_date, bond_spec, dc)
#         self.assertAlmostEqual(bpricer.compute_yield(bond_price), 0.05, delta=1e-3)
#         mc_duration = (
#             dcc.yf(ref_date, datetime(2024, 1, 2), bond_spec.dates, bond_spec.nr_annual_payments) * df1 * 5.0 / bond_price
#             + dcc.yf(ref_date, datetime(2025, 1, 2), bond_spec.dates, bond_spec.nr_annual_payments) * df2 * 105.0 / bond_price
#         )
#         self.assertAlmostEqual(bpricer.macaulay_duration(), mc_duration, delta=1e-3)
#         self.assertAlmostEqual(bpricer.z_spread(bond_price), 0.0, delta=1e-6)
#         self.assertAlmostEqual(bpricer.macaulay_duration(), mc_duration, delta=1e-3)
#         self.assertAlmostEqual(bpricer.modified_duration(bond_price), mc_duration / 1.05, delta=1e-3)

#     def test_float_bond(self):
#         ref_date = datetime(2023, 5, 1)
#         fix_table = FixingTable("PV_BOND")
#         fix_table.add("1Y", datetime(2022, 12, 30), 0.05)
#         float_bond_spec = FloatingRateBondSpecification(
#             "PV_BOND",
#             issue_date=datetime(2023, 1, 1),
#             maturity_date=datetime(2025, 1, 2),
#             currency=Currency.EUR,
#             notional=100.0,
#             issuer="None",
#             securitization_level=SecuritizationLevel.SUBORDINATED,
#             day_count_convention="ACTACTICMA",
#             business_day_convention="ModifiedFollowing",
#             margin=0.00,
#             frequency="1Y",
#             fixings=fix_table,
#         )
#         dc = DiscountCurveParametrized("", ref_date, ConstantRate(0.05))
#         nr_annual_payments = float_bond_spec.get_nr_annual_payments()
#         self.assertEqual(nr_annual_payments, 1)
#         dates = float_bond_spec.get_schedule()._roll_out(
#             from_=float_bond_spec._start_date if not float_bond_spec._backwards else float_bond_spec._end_date,
#             to_=float_bond_spec._end_date if not float_bond_spec._backwards else float_bond_spec._start_date,
#             term=_term_to_period(float_bond_spec._frequency),
#             long_stub=float_bond_spec._stub_type_is_Long,
#             backwards=float_bond_spec._backwards,
#         )
#         ## discount factors

#         df1 = dc.value(ref_date, datetime(2024, 1, 2))
#         df2 = dc.value(ref_date, datetime(2025, 1, 2))
#         dcc = DayCounter(float_bond_spec.day_count_convention)
#         self.assertAlmostEqual(df1, 0.9668628440577077, delta=1e-6)
#         self.assertAlmostEqual(df2, 0.9195824079027841, delta=1e-6)
#         price = df1 * 5.0 + df2 * 105.0
#         print(DeterministicCashflowPricer.get_expected_cashflows(float_bond_spec, ref_date, dc))
#         self.assertEqual(dates, [datetime(2023, 1, 2), datetime(2024, 1, 2), datetime(2025, 1, 2)])
#         expected = [
#             (datetime(2023, 1, 2), -100.0),
#             (datetime(2024, 1, 2), 5.0),
#             (datetime(2025, 1, 2), 5.0),
#             (datetime(2025, 1, 2), 100.0),
#         ]
#         actual = DeterministicCashflowPricer.get_expected_cashflows(float_bond_spec, ref_date, dc)

#         self.assertEqual(len(actual), len(expected))
#         for i, (exp, act) in enumerate(zip(expected, actual)):
#             with self.subTest(i=i):
#                 # exact equality for the date
#                 self.assertEqual(exp[0], act[0])
#                 # almost-equal for the amount with a tolerance (choose places or delta)
#                 self.assertAlmostEqual(exp[1], act[1], places=6)
#         self.assertAlmostEqual(DeterministicCashflowPricer.get_pv_cashflows(ref_date, float_bond_spec, dc, dc), price, delta=1e-6)
#         bond_yield = DeterministicCashflowPricer.get_compute_yield(
#             target_dirty_price=price, val_date=ref_date, specification=float_bond_spec, fwd_curve=dc
#         )
#         self.assertAlmostEqual(bond_yield, 0.05, delta=1e-3)
#         bpricer = DeterministicCashflowPricer(ref_date, float_bond_spec, dc, dc)
#         self.assertAlmostEqual(bpricer.compute_yield(price), 0.05, delta=1e-3)
#         mc_duration = (
#             dcc.yf(ref_date, datetime(2024, 1, 2), float_bond_spec.dates, float_bond_spec.nr_annual_payments) * df1 * 5.0 / price
#             + dcc.yf(ref_date, datetime(2025, 1, 2), float_bond_spec.dates, float_bond_spec.nr_annual_payments) * df2 * 105.0 / price
#         )
#         self.assertAlmostEqual(bpricer.macaulay_duration(), mc_duration, delta=1e-3)
#         self.assertAlmostEqual(bpricer.z_spread(price), 0.0, delta=1e-6)
#         self.assertAlmostEqual(bpricer.macaulay_duration(), mc_duration, delta=1e-3)
#         self.assertAlmostEqual(bpricer.modified_duration(price), mc_duration / 1.05, delta=1e-3)
#         print(bpricer.get_accrued_interest(float_bond_spec, ref_date, dc))

#     def test_index_float_bond(self):
#         ref_date = datetime(2023, 5, 1)
#         fix_table = FixingTable("PV_BOND")
#         fix_table.add("EURIBOR 12M", datetime(2022, 12, 30), 0.05)
#         float_index_bond_spec = FloatingRateBondSpecification(
#             "PV_BOND",
#             issue_date=datetime(2023, 1, 1),
#             maturity_date=datetime(2025, 1, 2),
#             currency=Currency.EUR,
#             notional=100.0,
#             issuer="None",
#             securitization_level=SecuritizationLevel.SUBORDINATED,
#             day_count_convention="ACTACTICMA",
#             business_day_convention="ModifiedFollowing",
#             margin=0.00,
#             index="EURIBOR_1Y",
#             frequency="1Y",
#             fixings=fix_table,
#             adjust_accruals=False,
#             adjust_schedule=False,
#         )
#         self.assertEqual(float_index_bond_spec._index, "EURIBOR_1Y")
#         self.assertEqual(float_index_bond_spec._ir_index.value.name, "EURIBOR 12M")
#         dc = DiscountCurveParametrized("", ref_date, ConstantRate(0.05))
#         nr_annual_payments = float_index_bond_spec.get_nr_annual_payments()
#         self.assertEqual(nr_annual_payments, 1)
#         dates = float_index_bond_spec.get_schedule()._roll_out(
#             from_=float_index_bond_spec._start_date if not float_index_bond_spec._backwards else float_index_bond_spec._end_date,
#             to_=float_index_bond_spec._end_date if not float_index_bond_spec._backwards else float_index_bond_spec._start_date,
#             term=_term_to_period(float_index_bond_spec._frequency),
#             long_stub=float_index_bond_spec._stub_type_is_Long,
#             backwards=float_index_bond_spec._backwards,
#         )
#         ## discount factors

#         df1 = dc.value(ref_date, datetime(2024, 1, 2))
#         df2 = dc.value(ref_date, datetime(2025, 1, 2))
#         dcc = DayCounter(float_index_bond_spec.day_count_convention)
#         self.assertAlmostEqual(df1, 0.9668628440577077, delta=1e-6)
#         self.assertAlmostEqual(df2, 0.9195824079027841, delta=1e-6)
#         price = df1 * 5.0 + df2 * 105.0
#         self.assertEqual(dates, [datetime(2023, 1, 2), datetime(2024, 1, 2), datetime(2025, 1, 2)])
#         expected = [
#             (datetime(2023, 1, 2), -100.0),
#             (datetime(2024, 1, 2), 5.0),
#             (datetime(2025, 1, 2), 5.0),
#             (datetime(2025, 1, 2), 100.0),
#         ]
#         actual = DeterministicCashflowPricer.get_expected_cashflows(float_index_bond_spec, ref_date, dc)

#         self.assertEqual(len(actual), len(expected))
#         for i, (exp, act) in enumerate(zip(expected, actual)):
#             with self.subTest(i=i):
#                 # exact equality for the date
#                 self.assertEqual(exp[0], act[0])
#                 # almost-equal for the amount with a tolerance (choose places or delta)
#                 self.assertAlmostEqual(exp[1], act[1], places=6)
#         self.assertAlmostEqual(DeterministicCashflowPricer.get_pv_cashflows(ref_date, float_index_bond_spec, dc, dc), price, delta=1e-6)
#         bond_yield = DeterministicCashflowPricer.get_compute_yield(
#             target_dirty_price=price, val_date=ref_date, specification=float_index_bond_spec, fwd_curve=dc
#         )
#         self.assertAlmostEqual(bond_yield, 0.05, delta=1e-3)
#         bpricer = DeterministicCashflowPricer(ref_date, float_index_bond_spec, dc, dc)
#         self.assertAlmostEqual(bpricer.compute_yield(price), 0.05, delta=1e-3)
#         mc_duration = (
#             dcc.yf(ref_date, datetime(2024, 1, 2), float_index_bond_spec.dates, float_index_bond_spec.nr_annual_payments) * df1 * 5.0 / price
#             + dcc.yf(ref_date, datetime(2025, 1, 2), float_index_bond_spec.dates, float_index_bond_spec.nr_annual_payments) * df2 * 105.0 / price
#         )
#         self.assertAlmostEqual(bpricer.macaulay_duration(), mc_duration, delta=1e-3)
#         self.assertAlmostEqual(bpricer.z_spread(price), 0.0, delta=1e-6)
#         self.assertAlmostEqual(bpricer.macaulay_duration(), mc_duration, delta=1e-3)
#         self.assertAlmostEqual(bpricer.modified_duration(price), mc_duration / 1.05, delta=1e-3)

#     def test_amort_bond(self):
#         """Simple test for yield computation: Fixed rate used to compute pv which is then used to compute yield."""
#         ref_date = datetime(2023, 5, 1)
#         bond_amort_spec = FixedRateBondSpecification(
#             "PV_BOND",
#             issue_date=datetime(2023, 1, 1),
#             maturity_date=datetime(2025, 1, 2),
#             currency=Currency.EUR,
#             notional=100.0,
#             amortization_scheme="linear",
#             issuer="None",
#             securitization_level=SecuritizationLevel.SUBORDINATED,
#             coupon=0.05,
#             frequency="1Y",
#         )
#         dc = DiscountCurveParametrized("", ref_date, ConstantRate(0.05))
#         nr_annual_payments = bond_amort_spec.get_nr_annual_payments()
#         self.assertEqual(nr_annual_payments, 1)
#         # schedule for accrual periods rolled out
#         dates = bond_amort_spec.get_schedule()._roll_out(
#             from_=bond_amort_spec._start_date if not bond_amort_spec._backwards else bond_amort_spec._end_date,
#             to_=bond_amort_spec._end_date if not bond_amort_spec._backwards else bond_amort_spec._start_date,
#             term=_term_to_period(bond_amort_spec._frequency),
#             long_stub=bond_amort_spec._stub_type_is_Long,
#             backwards=bond_amort_spec._backwards,
#         )
#         # print(bond_amort_spec._notional.notional)

#         ## discount factors

#         df1 = dc.value(ref_date, datetime(2024, 1, 2))
#         df2 = dc.value(ref_date, datetime(2025, 1, 2))
#         dcc = DayCounter(bond_amort_spec.day_count_convention)
#         self.assertAlmostEqual(df1, 0.9668628440577077, delta=1e-6)
#         self.assertAlmostEqual(df2, 0.9195824079027841, delta=1e-6)
#         price = df1 * 55.0 + df2 * 52.5
#         self.assertAlmostEqual(DeterministicCashflowPricer.get_pv_cashflows(ref_date, bond_amort_spec, dc), price, delta=1e-6)

#         self.assertEqual(dates, [datetime(2023, 1, 2), datetime(2024, 1, 2), datetime(2025, 1, 2)])
#         # print(DeterministicCashflowPricer.get_expected_cashflows(bond_amort_spec))
#         self.assertEqual(
#             DeterministicCashflowPricer.get_expected_cashflows(bond_amort_spec),
#             [
#                 (datetime(2023, 1, 2), -100.0),
#                 (datetime(2024, 1, 2), 5.0),
#                 (datetime(2024, 1, 2), 50.0),
#                 (datetime(2025, 1, 2), 2.5),
#                 (datetime(2025, 1, 2), 50.0),
#             ],
#         )
#         bond_price = DeterministicCashflowPricer.get_pv_cashflows(ref_date, bond_amort_spec, dc)
#         bond_yield = DeterministicCashflowPricer.get_compute_yield(target_dirty_price=bond_price, val_date=ref_date, specification=bond_amort_spec)
#         self.assertAlmostEqual(bond_yield, 0.05, delta=1e-3)
#         bpricer = DeterministicCashflowPricer(ref_date, bond_amort_spec, dc)
#         self.assertAlmostEqual(bpricer.compute_yield(bond_price), 0.05, delta=1e-3)
#         mc_duration = (
#             dcc.yf(ref_date, datetime(2024, 1, 2), bond_amort_spec.dates, bond_amort_spec.nr_annual_payments) * df1 * 55.0 / bond_price
#             + dcc.yf(ref_date, datetime(2025, 1, 2), bond_amort_spec.dates, bond_amort_spec.nr_annual_payments) * df2 * 52.5 / bond_price
#         )
#         self.assertAlmostEqual(bpricer.macaulay_duration(), mc_duration, delta=1e-3)
#         self.assertAlmostEqual(bpricer.z_spread(bond_price), 0.0, delta=1e-6)
#         self.assertAlmostEqual(bpricer.macaulay_duration(), mc_duration, delta=1e-3)
#         self.assertAlmostEqual(bpricer.modified_duration(bond_price), mc_duration / 1.05, delta=1e-3)


if __name__ == "__main__":
    main()
