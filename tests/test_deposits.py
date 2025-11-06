import math
from unittest import main, TestCase

from matplotlib.dates import relativedelta
from rivapy.marketdata.curves import DiscountCurve
from rivapy.pricing.deposit_pricing import DepositPricer
from rivapy.tools.datetools import DayCounter, Schedule, _term_to_period, calc_end_day
from rivapy.tools.holidays_compat import EuropeanCentralBank as _ECB
from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.tools.enums import DayCounterType, InterpolationType, ExtrapolationType
from rivapy.tools.enums import InterpolationType, RollConvention, SecuritizationLevel, DayCounterType, Currency
from rivapy.pricing import DeterministicCashflowPricer, price
from rivapy.marketdata import DiscountCurveParametrized, ConstantRate
import datetime as dt


class DepositSpecificationTests(TestCase):

    def test_deposit_specification(self):
        ccy = "EUR"
        ref_date = dt.datetime(2023, 1, 2)
        issue_date = ref_date
        maturity_date = dt.datetime(2024, 1, 26)
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
        self.assertIn(dt.date(2023, 1, 1), deposit_spec.calendar)
        self.assertEqual(deposit_spec.calendar[dt.date(2023, 1, 1)], _ECB()[dt.date(2023, 1, 1)])
        self.assertEqual(deposit_spec.adjust_start_date, True)
        self.assertEqual(deposit_spec.adjust_end_date, False)
        self.assertEqual(deposit_spec.frequency, "389D")
        self.assertEqual(deposit_spec.dates, [issue_date, maturity_date])

        dep_dict = deposit_spec.to_dict()
        self.assertEqual(dep_dict["obj_id"], "dummy_id")
        self.assertEqual(dep_dict["issue_date"], issue_date.isoformat())
        self.assertEqual(dep_dict["maturity_date"], maturity_date.isoformat())
        self.assertEqual(dep_dict["currency"], ccy)
        self.assertEqual(dep_dict["notional"].get_amount(), notional)
        self.assertEqual(dep_dict["rate"], rate)
        self.assertEqual(dep_dict["day_count_convention"], dcc)
        self.assertEqual(dep_dict["roll_convention"], "NONE")
        self.assertEqual(dep_dict["spot_days"], 5)
        self.assertEqual(dep_dict["business_day_convention"], "ModifiedFollowing")
        self.assertEqual(dep_dict["issuer"], "")
        self.assertEqual(dep_dict["securitization_level"], "NONE")
        self.assertEqual(dep_dict["payment_days"], 5)

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
        self.assertEqual(deposit_spec_2.start_date, dt.datetime(2023, 1, 2))
        self.assertEqual(deposit_spec_2.spot_days, 2)

        issue_date = dt.datetime(2023, 1, 1)
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
        self.assertEqual(deposit_spec_3.issue_date, dt.datetime(2023, 1, 1))
        self.assertEqual(deposit_spec_3.start_date, dt.datetime(2023, 1, 1))
        self.assertEqual(deposit_spec_3.end_date, dt.datetime(2024, 1, 1))
        self.assertEqual(deposit_spec_3.maturity_date, dt.datetime(2024, 1, 2))

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
        self.assertEqual(deposit_spec_4.issue_date, dt.datetime(2023, 1, 1))
        self.assertEqual(deposit_spec_4.start_date, dt.datetime(2023, 1, 2))
        self.assertEqual(deposit_spec_4.end_date, dt.datetime(2024, 1, 2))
        self.assertEqual(deposit_spec_4.maturity_date, dt.datetime(2024, 1, 2))

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
        self.assertEqual(deposit_spec_5.maturity_date, dt.datetime(2023, 1, 3))
        self.assertEqual(deposit_spec_5.start_date, dt.datetime(2023, 1, 2))
        self.assertEqual(deposit_spec_5.end_date, dt.datetime(2023, 1, 3))
        self.assertEqual(deposit_spec_5.spot_days, 0)


class DepositPricingTests(TestCase):

    def test_deposit_pricing(self):
        """Simple test for yield computation: Fixed rate used to compute pv which is then used to compute yield."""

        ccy = "EUR"
        ref_date = dt.datetime(2023, 1, 2)
        issue_date = ref_date
        maturity_date = dt.datetime(2024, 1, 26)
        dcc = "Act360"
        rate = 0.03
        notional = 100.0
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

        object_id = "TEST_CURVE"
        flat_rate = 0.025

        days_to_maturity = [1, 180, 365, 720, 3 * 365, 4 * 365, 10 * 365]
        dates = [ref_date + dt.timedelta(days=d) for d in days_to_maturity]

        df = [math.exp(-d / 365.0 * flat_rate) for d in days_to_maturity]
        dc = DiscountCurve(
            id=object_id, refdate=ref_date, dates=dates, df=df, interpolation=InterpolationType.LINEAR, extrapolation=ExtrapolationType.LINEAR
        )

        self.assertEqual(
            DeterministicCashflowPricer.get_expected_cashflows(deposit_spec, ref_date),
            [(dt.datetime(2023, 1, 2, 0, 0), -100.0), (dt.datetime(2024, 1, 31, 0, 0), 3.2416666666666663), (dt.datetime(2024, 1, 31, 0, 0), 100.0)],
        )
        self.assertAlmostEqual(DepositPricer.get_implied_simply_compounded_rate(ref_date, deposit_spec, dc), 0.024971370182974097, delta=1e-6)
        self.assertAlmostEqual(DepositPricer.get_price(ref_date, deposit_spec, dc), 100.4950269333136, delta=1e-6)

        fair_deposit_spec = DepositSpecification(
            obj_id="dummy_id",
            issuer="dummy_issuer",
            securitization_level="NONE",
            currency=ccy,
            issue_date=ref_date,
            maturity_date=deposit_spec.end_date,
            notional=deposit_spec.notional,
            rate=0.024971370182974097,
            day_count_convention=deposit_spec.day_count_convention,
            payment_days=deposit_spec.payment_days,
            spot_days=deposit_spec.spot_days,
        )
        df_curve = dc.value_fwd(ref_date, fair_deposit_spec.start_date, fair_deposit_spec.end_date)
        dcc = DayCounter(fair_deposit_spec.day_count_convention)
        delta_t = dcc.yf(fair_deposit_spec.start_date, fair_deposit_spec.end_date)
        df_implied = 1.0 / (1.0 + delta_t * 0.024971370182974097)

        self.assertAlmostEqual(df_curve, df_implied, delta=1e-10)


if __name__ == "__main__":
    main()
