import unittest
import numpy as np
import pandas as pd
from rivapy.credit.creditmetrics import CreditMetricsModel
from rivapy.instruments.components import Issuer


class TestCreditMetricsModel(unittest.TestCase):
    def setUp(self):
        # Minimal Testdaten
        self.n_simulation = 10
        self.transition_matrix = (
            np.matrix(
                """
                                90.81, 8.33, 0.68, 0.06, 0.08, 0.02, 0.01, 0.01;
                                0.70, 90.65, 7.79, 0.64, 0.06, 0.13, 0.02, 0.01;
                                0.09, 2.27, 91.05, 5.52, 0.74, 0.26, 0.01, 0.06;
                                0.02, 0.33, 5.95, 85.93, 5.30, 1.17, 1.12, 0.18;
                                0.03, 0.14, 0.67, 7.73, 80.53, 8.84, 1.00, 1.06;
                                0.01, 0.11, 0.24, 0.43, 6.48, 83.46, 4.07, 5.20;
                                0.21, 0, 0.22, 1.30, 2.38, 11.24, 64.86, 19.79"""
            )
            / 100
        )
        self.position_data = pd.DataFrame({"IssuerID": ["A", "B"], "Exposure": [100, 200], "RecoveryRate": [0.4, 0.5]})
        self.issuer_data = [
            Issuer(obj_id="A", name="IssuerA", rating="AAA", country="DE", sector="Industrials", esg_rating="AAA"),
            Issuer(obj_id="B", name="IssuerB", rating="BBB", country="DE", sector="Industrials", esg_rating="AAA"),
        ]
        self.stock_data = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=5),
                "Dax": [100, 101, 102, 103, 104],
                "IssuerA": [50, 51, 52, 53, 54],
                "IssuerB": [60, 61, 62, 63, 64],
            }
        )
        self.r = 0.01
        self.t = 1
        self.confidencelevel = 95
        self.seed = 42

        self.model = CreditMetricsModel(
            n_simulation=self.n_simulation,
            transition_matrix=self.transition_matrix,
            position_data=self.position_data,
            issuer_data=self.issuer_data,
            stock_data=self.stock_data,
            r=self.r,
            t=self.t,
            confidencelevel=self.confidencelevel,
            seed=self.seed,
        )

    def test_merge_positions_issuer(self):
        merged = self.model.merge_positions_issuer()
        self.assertIn("IssuerName", merged.columns)
        self.assertIn("RatingID", merged.columns)
        self.assertEqual(len(merged), 2)

    def test_get_correlation(self):
        corr = self.model.get_correlation()
        self.assertTrue("IssuerA" in corr.index)
        self.assertTrue("IssuerB" in corr.index)

    def test_get_cutoffs_rating(self):
        cutoffs = self.model.get_cutoffs_rating()
        self.assertEqual(cutoffs.shape, (8, 7))

    def test_get_expected_value(self):
        ev = self.model.get_expected_value()
        self.assertTrue("EV" in ev.columns)
        self.assertTrue("issuer" in ev.index.names or "issuer" in ev.index)

    def test_get_states(self):
        states = self.model.get_states()
        self.assertTrue("AAA" in states.columns)
        self.assertTrue("issuer" in states.index.names or "issuer" in states.index)

    def test_mc_calculation(self):
        Loss, rr_scenarios, issuer_ids, issuer_names = self.model.mc_calculation()
        self.assertEqual(Loss.shape, (self.n_simulation, len(issuer_ids)))
        self.assertEqual(rr_scenarios.shape, (self.n_simulation, len(issuer_ids)))
        self.assertEqual(len(issuer_ids), 2)
        self.assertEqual(len(issuer_names), 2)

    def test_get_loss_distribution(self):
        Loss, _, _, _ = self.model.mc_calculation()
        loss_distribution = self.model.get_loss_distribution(Loss)
        self.assertEqual(loss_distribution.shape[0], self.n_simulation)

    def test_get_portfolio_VaR(self):
        Loss, _, _, _ = self.model.mc_calculation()
        loss_distribution = self.model.get_loss_distribution(Loss)
        expected = np.sort(loss_distribution)
        expected = (-1) * np.interp((100 - self.confidencelevel) / 100, np.linspace(0, 1, len(expected)), expected)
        var = self.model.get_portfolio_VaR(loss_distribution)
        self.assertIsInstance(var, float)
        self.assertAlmostEqual(var, expected, places=5)

    def test_get_portfolio_ES(self):
        Loss, _, _, _ = self.model.mc_calculation()
        loss_distribution = self.model.get_loss_distribution(Loss)
        expected = min(loss_distribution) * (-1)  # Assuming expected is the minimum loss
        es = self.model.get_portfolio_ES(loss_distribution)
        self.assertIsInstance(es, float)
        self.assertAlmostEqual(es, expected, places=5)


if __name__ == "__main__":
    unittest.main()
