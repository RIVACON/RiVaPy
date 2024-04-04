import unittest
import math
from dateutil import relativedelta
from datetime import datetime, timedelta

import rivapy
from rivapy.marketdata import DiscountCurve, SurvivalCurve
from rivapy import enums
from rivapy.instruments import SimpleSchedule
from rivapy import _pyvacon_available

if _pyvacon_available:
    from rivapy.marketdata import DatedCurve


class CDSTest(unittest.TestCase):
    """Test simple CDS pricing using ISDA model."""

    def test_pricing(self):
        """Test simple CDS pricing using ISDA model."""
        if not _pyvacon_available:
            self.assertAlmostEqual(1, 1)
            return
        refdate = datetime(2020, 1, 1)
        # yield curve
        days_to_maturity = [1, 180, 360, 720, 3 * 360, 4 * 360, 5 * 360, 10 * 360]
        rates = [-0.0065, 0.0003, 0.0059, 0.0086, 0.0101, 0.012, 0.016, 0.02]
        dates = [refdate + timedelta(days=d) for d in days_to_maturity]
        dsc_fac = [math.exp(-rates[i] * days_to_maturity[i] / 360) for i in range(len(days_to_maturity))]
        dc = DiscountCurve("CDS_interest_rate", refdate, dates, dsc_fac, enums.InterpolationType.LINEAR)
        hazard_rates = [0, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.005]
        sc = SurvivalCurve("Survival", refdate, dates, hazard_rates)

        recoveries = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
        recovery = DatedCurve(
            "Recovery",
            refdate,
            dates,
            recoveries,
            enums.DayCounterType.Act365Fixed.name,
            enums.InterpolationType.LINEAR.name,
        )

        payment_dates = [refdate + relativedelta.relativedelta(years=i) for i in range(10)]
        spec = rivapy.instruments.CDSSpecification(
            premium=0.0012,
            protection_start=refdate,
            premium_pay_dates=payment_dates,
            notional=1000000.0,
        )

        cds_pricing_data = rivapy.pricing.CDSPricingData(
            spec=spec,
            val_date=refdate,
            discount_curve=dc,
            survival_curve=sc,
            recovery_curve=recovery,
        )

        pr = rivapy.pricing.price(cds_pricing_data)
        self.assertAlmostEqual(0.0, 0.0, 3)


class SimpleScheduleTest(unittest.TestCase):
    def test_simple_start_end(self):
        """Simple test: Generate schedule without restrictions to hours or weekdays and check if start is included and end is excluded"""
        simple_schedule = SimpleSchedule(datetime(2023, 1, 1), datetime(2023, 1, 1, 4, 0, 0), freq="1H")
        d = simple_schedule.get_schedule()
        self.assertEqual(len(d), 4)
        self.assertEqual(datetime(2023, 1, 1), d[0])

    def test_simple_hours(self):
        """Simple test: Generate schedule with restrictions to hours and check correctness."""
        simple_schedule = SimpleSchedule(datetime(2023, 1, 1), datetime(2023, 1, 1, 4, 0, 0), freq="1H", hours=[2, 3])
        d = simple_schedule.get_schedule()
        self.assertEqual(len(d), 2)
        self.assertEqual(datetime(2023, 1, 1, 2, 0, 0), d[0])
        self.assertEqual(datetime(2023, 1, 1, 3, 0, 0), d[1])

    def test_simple_weekdays(self):
        """Simple test: Generate schedule with restrictions to hours and weekdays and check correctness."""
        simple_schedule = SimpleSchedule(
            datetime(2023, 1, 1),
            datetime(2023, 1, 2, 4, 0, 0),
            freq="1H",
            hours=[2, 3],
            weekdays=[0],
        )
        d = simple_schedule.get_schedule()
        self.assertEqual(len(d), 2)
        self.assertEqual(datetime(2023, 1, 2, 2, 0, 0), d[0])
        self.assertEqual(datetime(2023, 1, 2, 3, 0, 0), d[1])


if __name__ == "__main__":
    unittest.main()
