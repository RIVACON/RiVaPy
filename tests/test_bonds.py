from unittest import main, TestCase
from rivapy.instruments import (
    ZeroCouponBondSpecification,
    FixedRateBondSpecification,
    FloatingRateNoteSpecification,
    PlainVanillaCouponBondSpecification,
)
from rivapy.tools.enums import RollConvention, SecuritizationLevel, DayCounterType, Currency
from rivapy.pricing import SimpleCashflowPricer
from rivapy.marketdata import DiscountCurveParametrized, ConstantRate
from datetime import date, datetime


class BondSpecificationTests(TestCase):

    def test_bond_specification(self):
        # zero coupon bond
        zero_coupon_bond = ZeroCouponBondSpecification(
            "US500769CH58",
            datetime(2007, 6, 29),
            datetime(2037, 6, 29),
            "USD",
            1000,
            "KfW",
            SecuritizationLevel.SENIOR_UNSECURED,
        )
        self.assertEqual(zero_coupon_bond.obj_id, "US500769CH58")
        self.assertEqual(zero_coupon_bond.issue_date, datetime(2007, 6, 29))
        self.assertEqual(zero_coupon_bond.maturity_date, datetime(2037, 6, 29))
        self.assertEqual(zero_coupon_bond.currency, "USD")
        self.assertEqual(zero_coupon_bond.notional, 1000)
        self.assertEqual(zero_coupon_bond.issuer, "KfW")
        self.assertEqual(zero_coupon_bond.securitization_level, "SENIOR_UNSECURED")
        # fixed rate bond
        fixed_rate_bond = FixedRateBondSpecification.from_master_data(
            "DE000CZ40NT7",
            datetime(2019, 3, 11),
            datetime(2024, 9, 11),
            0.0125,
            "1Y",
            True,
            True,
            RollConvention.FOLLOWING,
            "DE",
            "EUR",
            100000,
            "Commerzbank",
            SecuritizationLevel.NON_PREFERRED_SENIOR,
        )
        self.assertEqual(fixed_rate_bond.obj_id, "DE000CZ40NT7")
        self.assertEqual(fixed_rate_bond.issue_date, datetime(2019, 3, 11))
        self.assertEqual(fixed_rate_bond.maturity_date, datetime(2024, 9, 11))
        self.assertEqual(
            fixed_rate_bond.coupon_payment_dates,
            [
                datetime(2019, 9, 11),
                datetime(2020, 9, 11),
                datetime(2021, 9, 13),
                datetime(2022, 9, 12),
                datetime(2023, 9, 11),
                datetime(2024, 9, 11),
            ],
        )
        self.assertEqual(fixed_rate_bond.coupons, [0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125])
        self.assertEqual(fixed_rate_bond.currency, "EUR")
        self.assertEqual(fixed_rate_bond.notional, 100000)
        self.assertEqual(fixed_rate_bond.issuer, "Commerzbank")
        self.assertEqual(fixed_rate_bond.securitization_level, "NON_PREFERRED_SENIOR")
        fixed_rate_bond = FixedRateBondSpecification(
            "DE000CZ40NT7",
            datetime(2019, 3, 11),
            datetime(2024, 9, 11),
            fixed_rate_bond.coupon_payment_dates,
            fixed_rate_bond.coupons,
            "EUR",
            100000,
            "Commerzbank",
            SecuritizationLevel.NON_PREFERRED_SENIOR,
        )
        self.assertEqual(fixed_rate_bond.obj_id, "DE000CZ40NT7")
        self.assertEqual(fixed_rate_bond.issue_date, datetime(2019, 3, 11))
        self.assertEqual(fixed_rate_bond.maturity_date, datetime(2024, 9, 11))
        self.assertEqual(
            fixed_rate_bond.coupon_payment_dates,
            [
                datetime(2019, 9, 11),
                datetime(2020, 9, 11),
                datetime(2021, 9, 13),
                datetime(2022, 9, 12),
                datetime(2023, 9, 11),
                datetime(2024, 9, 11),
            ],
        )
        self.assertEqual(fixed_rate_bond.coupons, [0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125])
        self.assertEqual(fixed_rate_bond.currency, "EUR")
        self.assertEqual(fixed_rate_bond.notional, 100000)
        self.assertEqual(fixed_rate_bond.issuer, "Commerzbank")
        self.assertEqual(fixed_rate_bond.securitization_level, "NON_PREFERRED_SENIOR")

        # floating rate bond
        floating_rate_note = FloatingRateNoteSpecification.from_master_data(
            "DE000HLB3DU1",
            datetime(2016, 6, 23),
            datetime(2024, 6, 27),
            "3M",
            True,
            False,
            RollConvention.FOLLOWING,
            "DE",
            DayCounterType.ThirtyU360,
            0.0,
            "EURIBOR_3M",
            "EUR",
            1000,
            "Helaba",
            SecuritizationLevel.NON_PREFERRED_SENIOR,
        )
        self.assertEqual(floating_rate_note.obj_id, "DE000HLB3DU1")
        self.assertEqual(floating_rate_note.issue_date, datetime(2016, 6, 23))
        self.assertEqual(floating_rate_note.maturity_date, datetime(2024, 6, 27))
        self.assertEqual(
            floating_rate_note.coupon_period_dates,
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
        self.assertEqual(floating_rate_note.daycount_convention, "30U360")
        self.assertEqual(
            floating_rate_note.spreads,
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )
        self.assertEqual(floating_rate_note.reference_index, "EURIBOR_3M")
        self.assertEqual(floating_rate_note.currency, "EUR")
        self.assertEqual(floating_rate_note.notional, 1000)
        self.assertEqual(floating_rate_note.issuer, "Helaba")
        self.assertEqual(floating_rate_note.securitization_level, "NON_PREFERRED_SENIOR")
        floating_rate_note = FloatingRateNoteSpecification(
            "DE000HLB3DU1",
            datetime(2016, 6, 23),
            datetime(2024, 6, 27),
            floating_rate_note.coupon_period_dates,
            DayCounterType.ThirtyU360,
            floating_rate_note.spreads,
            "EURIBOR_3M",
            "EUR",
            1000,
            "Helaba",
            SecuritizationLevel.NON_PREFERRED_SENIOR,
        )
        self.assertEqual(floating_rate_note.obj_id, "DE000HLB3DU1")
        self.assertEqual(floating_rate_note.issue_date, datetime(2016, 6, 23))
        self.assertEqual(floating_rate_note.maturity_date, datetime(2024, 6, 27))
        self.assertEqual(
            floating_rate_note.coupon_period_dates,
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
        self.assertEqual(floating_rate_note.daycount_convention, "30U360")
        self.assertEqual(
            floating_rate_note.spreads,
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )
        self.assertEqual(floating_rate_note.reference_index, "EURIBOR_3M")
        self.assertEqual(floating_rate_note.currency, "EUR")
        self.assertEqual(floating_rate_note.notional, 1000)
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
        bond_spec = PlainVanillaCouponBondSpecification(
            "PV_BOND",
            issue_date=datetime(2023, 1, 1),
            maturity_date=datetime(2025, 1, 2),
            currency=Currency.EUR,
            notional=100.0,
            issuer="None",
            securitization_level=SecuritizationLevel.SUBORDINATED,
            coupon=0.05,
            coupon_freq="1Y",
            accrual_start=datetime(2023, 2, 10),
        )
        dc = DiscountCurveParametrized("", ref_date, ConstantRate(0.05))
        bond_price = SimpleCashflowPricer.pv_cashflows(ref_date, bond_spec, dc)
        bond_yield = SimpleCashflowPricer.compute_yield(
            target_dirty_price=bond_price, val_date=ref_date, specification=bond_spec
        )
        self.assertAlmostEqual(bond_yield, 0.05, delta=1e-3)


if __name__ == "__main__":
    main()
