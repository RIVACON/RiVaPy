from unittest import main, TestCase
import unittest
import rivapy
from rivapy.tools.datetools import DayCounter, Schedule, _term_to_period
from rivapy.tools.enums import RollConvention, SecuritizationLevel, DayCounterType, Currency
from rivapy.instruments._logger import logger
from datetime import date, datetime
from rivapy.instruments.components import (
    AmortizationScheme,
    ConstNotionalStructure,
    LinearNotionalStructure,
    ZeroAmortizationScheme,
    LinearAmortizationScheme,
    VariableAmortizationScheme,
)


class ComponentsTests(TestCase):
    amort_scheme = ZeroAmortizationScheme()
    amort_scheme_lin = LinearAmortizationScheme(80, 2)
    amort_scheme_var = VariableAmortizationScheme([30, 50, 20], [1, 2, 3])

    def test_amortization(self):
        self.assertEqual(ComponentsTests.amort_scheme.get_total_amortization(), 0.0)
        self.assertEqual(ComponentsTests.amort_scheme._total_amortization, 0.0)
        self.assertEqual(ComponentsTests.amort_scheme_lin.get_total_amortization(), 80.0)
        self.assertEqual(ComponentsTests.amort_scheme_lin.total_amortization, 80)
        self.assertEqual(ComponentsTests.amort_scheme_lin._n_steps, 2)
        with self.assertRaises(ValueError):
            LinearAmortizationScheme(105, 2)  # negative amortization not allowed
        self.assertEqual(ComponentsTests.amort_scheme_var.get_total_amortization(), 100.0)
        self.assertEqual(ComponentsTests.amort_scheme_var._amortization_amounts, [30, 50, 20])
        self.assertEqual(ComponentsTests.amort_scheme_var._terms, [1, 2, 3])
        self.assertEqual(ComponentsTests.amort_scheme_var.get_nr_of_amortization_steps(), 3)
        with self.assertRaises(ValueError):
            VariableAmortizationScheme([60, 50], [1, 2])  # sum > 100 not allowed

    def test_notional_structures(self):
        notional_const = ConstNotionalStructure(1000000)
        date = datetime(2025, 1, 1)
        self.assertEqual(notional_const.notional, [1000000])
        self.assertEqual(notional_const.get_amount(5), 1000000)
        self.assertEqual(notional_const.get_amount_per_date(date), 1000000)
        self.assertEqual(notional_const.get_size(), 1)
        self.assertEqual(notional_const.get_amortizations_by_index(), [(1, 1000000)])
        self.assertEqual(notional_const.get_amortization_schedule(), [])
        with self.assertLogs(logger, level="ERROR") as cm:
            result = notional_const.get_amortization_schedule()
            # assertLogs returns a list of formatted log messages in cm.output
            self.assertIn("End dates of notional structure are not set.", cm.output[0])

        notional_lin = LinearNotionalStructure(1000000, 500000, 5)
        self.assertEqual(notional_lin._start_notional, 1000000)
        self.assertEqual(notional_lin._end_notional, 500000)
        self.assertEqual(notional_lin._n_steps, 5)
        self.assertEqual(notional_lin.notional, [1000000.0, 875000.0, 750000.0, 625000.0, 500000.0])
        self.assertEqual(notional_lin.get_amount(4), 500000)
        self.assertEqual(notional_lin.get_size(), 5)
        self.assertEqual(notional_lin.get_amortizations_by_index(), [(1, 125000.0), (2, 125000.0), (3, 125000.0), (4, 125000.0)])
        with self.assertLogs(logger, level="ERROR") as cm:
            result = notional_lin.get_amortization_schedule()
            # assertLogs returns a list of formatted log messages in cm.output
            self.assertIn("End dates of notional structure are not set.", cm.output[0])
            # self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
