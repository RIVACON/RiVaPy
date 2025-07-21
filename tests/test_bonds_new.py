import unittest
import datetime as dt
from rivapy.marketdata.curves import DiscountCurve
from rivapy.tools.datetools import Schedule, Period, DayCounter
from rivapy.tools.enums import DayCounterType, InterpolationType, ExtrapolationType, SecuritizationLevel, RollConvention
from rivapy.instruments.bond_specifications import FixedRateBond, FloatingRateBond
from holidays.financial import ECB


class TestFixedRateBonds(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFixedRateBonds, self).__init__(*args, **kwargs)

    def test_annually_coupon_unadjusted(self):
        obj_id = "test_bond"
        issue_date = dt.datetime(2019, 1, 1)
        maturity_date = dt.datetime(2024, 1, 1)
        coupon_rate = 0.03
        coupon_payment = Period(years=1)
        notional = 100.0
        securitisation_level_val = SecuritizationLevel.NONE
        discount_rate = 0.02
        daycounter = DayCounterType.ACT_ACT

        scheduler = Schedule(
            start_day=issue_date,
            end_day=maturity_date,
            time_period=coupon_payment,
            backwards=True,
            stub=True,
            business_day_convention=RollConvention.UNADJUSTED,
            calendar=ECB(years=range(issue_date.year, maturity_date.year + 1)),
        )
        refdate = issue_date - dt.timedelta(days=1)
        dates = [issue_date, maturity_date]
        rates = [discount_rate] * len(dates)

        discount_curve = DiscountCurve(
            id="test_curve",
            refdate=refdate,
            dates=dates,
            df=rates,
            interpolation=InterpolationType.LINEAR,
            extrapolation=ExtrapolationType.LINEAR,
            daycounter=daycounter,
        )

        bond = FixedRateBond(
            obj_id=obj_id,
            schedule=scheduler,
            notional=notional,
            currency="EUR",
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon_rate=coupon_rate,
            issuer="Test",
            securitization_level=securitisation_level_val,
            accrual_day_counter_type=daycounter,
        )

        target_cashflows = [
            (dt.datetime(2020, 1, 1), 3.0),
            (dt.datetime(2021, 1, 1), 3.0),
            (dt.datetime(2022, 1, 1), 3.0),
            (dt.datetime(2023, 1, 1), 3.0),
            (dt.datetime(2024, 1, 1), 103.0),
        ]
        true_cashflows = bond.cashflows

        for i in range(len(target_cashflows)):
            target_cashflow_tpl = target_cashflows[i]
            self.assertTupleEqual(target_cashflow_tpl, true_cashflows[i])

        # valuation 1
        value_date = dt.datetime(2019, 1, 1)
        target_ytm = 0.02
        target_clean_price = 104.7134595085042
        target_dirty_price = 104.7134595085042
        target_accrued_interest = 0.0

        dirty_price = bond.compute_dirty_price(value_date=value_date, discount_curve=discount_curve)
        clean_price = bond.compute_clean_price(value_date=value_date, discount_curve=discount_curve)
        accrued_interest = bond.compute_accrued_interest(valuation_date=value_date)
        ytm = bond.compute_yield(dirty_price=dirty_price, val_date=value_date)

        self.assertAlmostEqual(target_clean_price - clean_price, 0.0, places=8)
        self.assertAlmostEqual(target_dirty_price - dirty_price, 0.0, places=8)
        self.assertAlmostEqual(target_accrued_interest - accrued_interest, 0.0, places=8)
        self.assertAlmostEqual(target_ytm - ytm, 0, places=8)

        # valuation 2
        value_date = dt.datetime(2023, 6, 1)
        target_accrued_interest = 1.2410958904109588
        target_dirty_price = 101.81105369821792
        target_clean_price = 100.58428398614735

        dirty_price = bond.compute_dirty_price(value_date=value_date, discount_curve=discount_curve)
        clean_price = bond.compute_clean_price(value_date=value_date, discount_curve=discount_curve)
        accrued_interest = bond.compute_accrued_interest(valuation_date=value_date)

        self.assertAlmostEqual(target_clean_price - clean_price, 0.0, places=8)
        self.assertAlmostEqual(target_dirty_price - dirty_price, 0.0, places=8)
        self.assertAlmostEqual(target_accrued_interest - accrued_interest, 0.0, places=8)

    def test_semi_annually_coupon_following(self):
        obj_id = "test_bond"
        issue_date = dt.datetime(2023, 1, 1)
        maturity_date = dt.datetime(2024, 1, 1)
        coupon_rate = 0.03
        coupon_payment = Period(months=6)
        notional = 100.0
        securitisation_level_val = SecuritizationLevel.NONE
        discount_rate = 0.02
        daycounter = DayCounterType.ActActICMA

        scheduler = Schedule(
            start_day=issue_date,
            end_day=maturity_date,
            time_period=coupon_payment,
            backwards=True,
            stub=True,
            business_day_convention=RollConvention.FOLLOWING,
            calendar=ECB(years=range(issue_date.year, maturity_date.year + 1)),
        )

        refdate = issue_date - dt.timedelta(days=1)
        dates = [issue_date, maturity_date]
        rates = [discount_rate] * len(dates)

        discount_curve = DiscountCurve(
            id="test_curve",
            refdate=refdate,
            dates=dates,
            df=rates,
            interpolation=InterpolationType.LINEAR,
            extrapolation=ExtrapolationType.LINEAR,
            daycounter=daycounter,
        )
        bond = FixedRateBond(
            obj_id=obj_id,
            schedule=scheduler,
            notional=notional,
            currency="EUR",
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon_rate=coupon_rate,
            issuer="Test",
            securitization_level=securitisation_level_val,
            accrual_day_counter_type=daycounter,
        )

        value_date = dt.datetime(2023, 6, 1)
        target_clean_price = 100.58844590976388
        target_dirty_price = 101.82255931875143
        target_accrued_interest = 1.2362637362637363

        target_cashflows = [
            (dt.datetime(2023, 7, 3), 1.5),
            (dt.datetime(2024, 1, 2), 101.5),
        ]
        true_cashflows = bond.cashflows

        for i in range(len(target_cashflows)):
            target_cashflow_tpl = target_cashflows[i]
            self.assertTupleEqual(target_cashflow_tpl, true_cashflows[i])

        dirty_price = bond.compute_dirty_price(value_date=value_date, discount_curve=discount_curve)
        clean_price = bond.compute_clean_price(value_date=value_date, discount_curve=discount_curve)
        accrued_interest = bond.compute_accrued_interest(valuation_date=value_date)

        self.assertAlmostEqual(target_clean_price - clean_price, 0.0, places=8)
        self.assertAlmostEqual(target_dirty_price - dirty_price, 0.0, places=8)
        self.assertAlmostEqual(target_accrued_interest - accrued_interest, 0.0, places=8)


class TestFloatingRateBonds(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFloatingRateBonds, self).__init__(*args, **kwargs)

    def test_semi_annually_coupon_unadjusted(self):
        obj_id = "test_bond"
        issue_date = dt.datetime(2023, 1, 1)
        maturity_date = dt.datetime(2024, 1, 1)
        coupon_rate = 0.03
        coupon_payment = Period(months=6)
        notional = 100.0
        securitisation_level_val = SecuritizationLevel.NONE

        daycounter = DayCounterType.ACT_ACT

        scheduler = Schedule(
            start_day=issue_date,
            end_day=maturity_date,
            time_period=coupon_payment,
            backwards=True,
            stub=True,
            business_day_convention=RollConvention.UNADJUSTED,
            calendar=ECB(years=range(issue_date.year, maturity_date.year + 1)),
        )

        refdate = issue_date - dt.timedelta(days=1)
        dates = [issue_date, maturity_date]
        rates = [0.02, 0.04]

        discount_curve = DiscountCurve(
            id="test_curve",
            refdate=refdate,
            dates=dates,
            df=rates,
            interpolation=InterpolationType.LINEAR,
            extrapolation=ExtrapolationType.LINEAR,
            daycounter=daycounter,
        )
        value_date = dt.datetime(2023, 6, 1)
        refdate = value_date - dt.timedelta(days=1)
        dates = [value_date, maturity_date]
        rates = [0.03, 0.06]

        ref_curve = DiscountCurve(
            id="test_curve",
            refdate=refdate,
            dates=dates,
            df=rates,
            interpolation=InterpolationType.LINEAR,
            extrapolation=ExtrapolationType.LINEAR,
            daycounter=daycounter,
        )
        margin = 0.01
        fixing_coupon_rate = 0.02
        bond = FloatingRateBond(
            obj_id=obj_id,
            schedule=scheduler,
            notional=notional,
            currency="EUR",
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon_ref_curve=ref_curve,
            margin=margin,
            fixing_date=value_date,
            fixing_coupon_rate=fixing_coupon_rate,
            issuer="Test",
            securitization_level=securitisation_level_val,
            accrual_day_counter_type=daycounter,
        )

        target_cashflows = [
            (dt.datetime(2023, 7, 1), DayCounter.yf_ActAct(d1=dt.datetime(2023, 1, 1), d2=dt.datetime(2023, 7, 1)) * 0.02 * notional),
            (
                dt.datetime(2024, 1, 1),
                DayCounter.yf_ActAct(d1=dt.datetime(2023, 7, 1), d2=dt.datetime(2024, 1, 1)) * (0.06 + margin) * notional + notional,
            ),
        ]

        true_cashflows = bond.cashflows

        for i in range(len(target_cashflows)):
            target_cashflow_tpl = target_cashflows[i]
            self.assertTupleEqual(target_cashflow_tpl, true_cashflows[i])

        target_accrued_interest = 0.8273972602739728
        target_dirty_price = 102.16465071930195
        target_clean_price = 101.33925575930313

        dirty_price = bond.compute_dirty_price(value_date=value_date, discount_curve=discount_curve)
        clean_price = bond.compute_clean_price(value_date=value_date, discount_curve=discount_curve)
        accrued_interest = bond.compute_accrued_interest(valuation_date=value_date)

        self.assertAlmostEqual(target_clean_price - clean_price, 0.0, places=8)
        self.assertAlmostEqual(target_dirty_price - dirty_price, 0.0, places=8)
        self.assertAlmostEqual(target_accrued_interest - accrued_interest, 0.0, places=8)

    def test_assertion(self):
        obj_id = "test_bond"
        issue_date = dt.datetime(2023, 1, 1)
        maturity_date = dt.datetime(2025, 1, 1)
        coupon_payment = Period(months=6)
        notional = 100.0
        securitisation_level_val = SecuritizationLevel.NONE

        daycounter = DayCounterType.ACT_ACT

        scheduler = Schedule(
            start_day=issue_date,
            end_day=maturity_date,
            time_period=coupon_payment,
            backwards=True,
            stub=True,
            business_day_convention=RollConvention.UNADJUSTED,
            calendar=ECB(years=range(issue_date.year, maturity_date.year + 1)),
        )

        refdate = issue_date - dt.timedelta(days=1)
        dates = [issue_date, maturity_date]
        rates = [0.02, 0.04]

        discount_curve = DiscountCurve(
            id="test_curve",
            refdate=refdate,
            dates=dates,
            df=rates,
            interpolation=InterpolationType.LINEAR,
            extrapolation=ExtrapolationType.LINEAR,
            daycounter=daycounter,
        )

        value_date = dt.datetime(2024, 6, 1)
        refdate = value_date - dt.timedelta(days=1)
        dates = [value_date, maturity_date]
        rates = [0.03, 0.06]

        ref_curve = DiscountCurve(
            id="test_curve",
            refdate=refdate,
            dates=dates,
            df=rates,
            interpolation=InterpolationType.LINEAR,
            extrapolation=ExtrapolationType.LINEAR,
            daycounter=daycounter,
        )

        # the coupon payment at 2024-01-01 falls between the fixed coupon payment and the start of the coupon reference curve
        with self.assertRaises(ValueError):
            FloatingRateBond(
                obj_id=obj_id,
                schedule=scheduler,
                notional=notional,
                currency="EUR",
                issue_date=issue_date,
                maturity_date=maturity_date,
                coupon_ref_curve=ref_curve,
                margin=0.01,
                fixing_date=dt.datetime(2023, 6, 1),
                fixing_coupon_rate=0.02,
                issuer="Test",
                securitization_level=securitisation_level_val,
                accrual_day_counter_type=daycounter,
            )

    def test_cashflow_dates_fixing_date(self):
        obj_id = "test_bond"
        issue_date = dt.datetime(2023, 1, 1)
        maturity_date = dt.datetime(2025, 1, 1)
        coupon_rate = 0.03
        coupon_payment = Period(months=6)
        notional = 100.0
        securitisation_level_val = SecuritizationLevel.NONE

        daycounter = DayCounterType.ACT_ACT

        scheduler = Schedule(
            start_day=issue_date,
            end_day=maturity_date,
            time_period=coupon_payment,
            backwards=True,
            stub=True,
            business_day_convention=RollConvention.UNADJUSTED,
            calendar=ECB(years=range(issue_date.year, maturity_date.year + 1)),
        )

        refdate = issue_date - dt.timedelta(days=1)
        dates = [issue_date, maturity_date]
        rates = [0.02, 0.04]

        discount_curve = DiscountCurve(
            id="test_curve",
            refdate=refdate,
            dates=dates,
            df=rates,
            interpolation=InterpolationType.LINEAR,
            extrapolation=ExtrapolationType.LINEAR,
            daycounter=daycounter,
        )

        value_date = dt.datetime(2023, 6, 1)
        refdate = value_date - dt.timedelta(days=1)
        dates = [value_date, maturity_date]
        rates = [0.03, 0.06]

        ref_curve = DiscountCurve(
            id="test_curve",
            refdate=refdate,
            dates=dates,
            df=rates,
            interpolation=InterpolationType.LINEAR,
            extrapolation=ExtrapolationType.LINEAR,
            daycounter=daycounter,
        )

        bond = FloatingRateBond(
            obj_id=obj_id,
            schedule=scheduler,
            notional=notional,
            currency="EUR",
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon_ref_curve=ref_curve,
            margin=0.01,
            fixing_date=dt.datetime(2024, 6, 1),  # <- fixing date is after the valuation date, therefore the valuation starts after the fixing date
            fixing_coupon_rate=0.02,
            issuer="Test",
            securitization_level=securitisation_level_val,
            accrual_day_counter_type=daycounter,
        )

        target_payment_dates = [dt.datetime(2024, 7, 1), dt.datetime(2025, 1, 1)]

        true_cashflows = bond.cashflows

        for i in range(len(true_cashflows)):
            self.assertEqual(target_payment_dates[i], true_cashflows[i][0])


if __name__ == "__main__":
    unittest.main()
