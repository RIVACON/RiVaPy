from __future__ import division
from turtle import position
import pandas as pd
import numpy as np
from scipy.stats import norm
import sys
import math
from scipy.linalg import sqrtm
from random import seed
from random import random
import plotly.express as px
from typing import List, Union as _Union
from rivapy.instruments.components import Issuer
from rivapy.tools.enums import Rating
from numpy.linalg import cholesky


class creditMetricsModel:
    def __init__(
        self,
        n_simulation: int,
        transition_matrix: np.matrix,
        position_data: pd.DataFrame,
        issuer_data: List[Issuer],
        stock_data: pd.DataFrame,
        r: float,
        t: float,
        confidencelevel: int,
        seed: int = None,
    ):
        """_summary_

        Args:
            n_simulation (int): Number of simulation, which should be carried out
            transition_matrix (np.matrix): Transition matrix (format np.matrix). S&P 8x8 matrix is integrated.
            position_data (pd.DataFrame): Dataframe with position data. Specific format is needed.
            issuer_data (pd.DataFrame): Dataframe with issuer data. Specific format is needed.
            stock_data (pd.DataFrame): Dataframe with stock data. Stock data needs to include close values of the different issuers as well as a reference time series (e.g. Dax)
            r (float): Risk-free rate. Needed to comupute expected value of positions as well as different states during transition process.
            t (float): Dipositon horizon for calculation of credit risk.
            confidencelevel (int): Used confidence level in VaR-Calculation. Format Int.
            seed (int, optional): Seed for random number generator. Defaults to None.
        """

        self.n_simulation = n_simulation
        self.transition_matrix = transition_matrix
        self.position_data = position_data
        self.issuer_data = issuer_data
        self.stock_data = stock_data
        self.r = r
        self.t = t
        self.confidencelevel = confidencelevel
        self.seed = seed

    def mergePositionsIssuer(self):
        """
        Merges position dataframe with issuer dataframe to obtain rating-data for each position.
        Maps all +/- Rating variants to the same RatingID.
        Returns:
            DataFrame: Returns adjusted position dataframe.
        """
        # Mapping aller Rating-Varianten (inkl. +/-) auf RatingID
        rating_map = pd.DataFrame(
            {
                "Rating": [
                    "AAA",
                    "AA+",
                    "AA",
                    "AA-",
                    "A+",
                    "A",
                    "A-",
                    "BBB+",
                    "BBB",
                    "BBB-",
                    "BB+",
                    "BB",
                    "BB-",
                    "B+",
                    "B",
                    "B-",
                    "CCC+",
                    "CCC",
                    "CCC-",
                    "D",
                ],
                "RatingID": [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7],
            }
        )

        # issuer_data ist jetzt eine Liste von Issuer-Objekten
        issuer_df = pd.DataFrame(
            [
                {
                    "IssuerID": issuer.obj_id,
                    "IssuerName": issuer.name,
                    "Rating": str(issuer.rating),  # ggf. .value oder .name je nach Enum-Implementierung
                }
                for issuer in self.issuer_data
            ]
        )

        # Mapping anwenden
        issuer_adj = issuer_df.merge(rating_map, on="Rating", how="left")
        positions_adj = self.position_data.merge(issuer_adj[["IssuerID", "IssuerName", "Rating", "RatingID"]], on="IssuerID", how="left")

        return positions_adj

    def get_correlation(self):
        """Calculates correlation pairs for issuer with a specific reference time series.

        Returns:
            DataFrame: Dataframe with correlation coefficient for each issuer.
        """

        mergedData = self.stock_data.drop(["Date"], axis=1)
        returns = mergedData.pct_change()

        correlation_mat = returns.corr()
        corr_pairs = correlation_mat.unstack()["Dax"]
        return corr_pairs

    def get_cutoffs_rating(self):
        """Computes cutoffs for each initial rating based on input transition matrix.
        The inverse function of the standard normal distribution is used to get specific thresholds.

        Returns:
            DataFrame: Dataframe with arrays including thresholds for each initial rating.
        """
        Z = np.cumsum(np.flipud(self.transition_matrix.T), 0)
        Z[Z >= (1 - 1 / 1e12)] = 1 - 1 / 1e12
        Z[Z <= (0 + 1 / 1e12)] = 0 + 1 / 1e12

        CutOffs = norm.ppf(Z, 0, 1)  # compute cut offes by inverting normal distribution
        return CutOffs

    def get_credit_spreads(self, LGD, idx):
        """Computes credit spreads for every rating based on the following formula
        -np.log(1-LGD*PD_t)/1

        Args:
            LGD (_type_): Instrument-specific LGD can be used in this computation.

        Returns:
            DataFrame: Dataframe with Credit spreads for each initial rating.
        """
        # credit spread implied by transmat
        PD_t = self.transition_matrix[:, -1]
        PD_vec = PD_t[idx]
        LGD_np = LGD.to_numpy().reshape(-1, 1)
        credit_spread = -np.log(1 - np.multiply(LGD_np, PD_vec)) / self.t
        return credit_spread

    def get_expected_value(self):
        """Calculates expected value of every position based on exposure and credit spread for initial rating class.

        Returns:
            DataFrame: Dataframe including expected values.
        """
        positions = self.get_issuer_groups()
        exposure = np.matrix(positions["Exposure"]).T
        idx = positions["RatingID"]
        LGD = 1 - positions["RecoveryRate"]
        credit_spread = self.get_credit_spreads(LGD, idx)
        EV = np.multiply(exposure, np.exp(-(self.r + credit_spread) * self.t))
        EV = pd.DataFrame(EV, columns=["EV"])  # keep in same order as credit cutoff
        EV["issuer"] = positions["IssuerName"].to_list()
        EV = EV.groupby("issuer").sum()  # group by issuer to sum up expected values
        return EV

    def get_states(self):
        """Calculates matrix of present values for every position and every possible future rating.

        Returns:
            DataFrame: Dataframe with all possible present values.
        """
        positions = self.get_issuer_groups()
        LGD = 1 - np.array(positions["RecoveryRate"])
        PD_t = self.transition_matrix[:, -1]  # default probability at t
        credit_spread = -np.log(1 - PD_t * LGD.T)
        exposure = np.matrix(positions["Exposure"])
        state = np.multiply(exposure, np.exp(-(self.r + credit_spread) * self.t)).T
        state = np.append(state, np.multiply(exposure, np.matrix(positions["RecoveryRate"])).T, axis=1)  # last column is default case
        states = pd.DataFrame(np.fliplr(state), columns=["D", "C", "B", "BB", "BBB", "A", "AA", "AAA"])  # keep in same order as credit cutoff
        states["issuer"] = positions["IssuerName"].to_list()
        states = states.groupby("issuer").sum()
        return states

    def get_issuer_groups(self):
        df_positions_grouped = self.mergePositionsIssuer()
        df_positions_grouped = df_positions_grouped[["IssuerID", "IssuerName", "RecoveryRate", "Rating", "RatingID", "Exposure"]]
        df_positions_grouped = df_positions_grouped.groupby(["IssuerID", "IssuerName", "RecoveryRate", "Rating", "RatingID"], as_index=False).sum()

        return df_positions_grouped

    def mc_calculation(self):
        """
        Monte-Carlo simulation of portfolio based on positions, issuer, correlation and transition matrix.

        For each simulation step, the return of each issuer is simulated:
        - The return of the benchmark (Y) is simulated and multiplied with the issuer-specific correlation. This random number is consistent for every issuer during one simulation step.
        - Afterwards, the idiosyncratic return of each issuer is simulated and multiplied with the idiosyncratic risk factor sqrt(1-p^2).
        - This results in the simulated return for every issuer in every simulation step:
        r_k = rho * Y + sqrt(1 - rho^2) * Z_k

        For each issuer, the new rating is determined and the loss is calculated as the difference between the new value and the expected value.

        Returns:
            tuple:
                Loss (np.ndarray): Array of shape (n_simulation, n_issuer) with losses for each scenario and issuer.
                issuer_ids (np.ndarray): Array of issuer IDs, order matches Loss columns.
                issuer_names (list): List of issuer names, order matches Loss columns.
        """
        positions = self.get_issuer_groups()
        correlation = self.get_correlation()
        cutOffs = self.get_cutoffs_rating()
        states = self.get_states()
        EV = self.get_expected_value()
        issuer_info = positions[["IssuerName", "IssuerID", "Rating", "RatingID"]].drop_duplicates()
        issuer_ids = issuer_info["IssuerID"].to_numpy()
        issuer_names = issuer_info["IssuerName"].to_list()
        Loss = np.zeros((self.n_simulation, len(issuer_ids)))
        rr_scenarios = np.zeros((self.n_simulation, len(issuer_ids)))
        np.random.seed(self.seed)

        for i in range(self.n_simulation):
            YY = norm.ppf(np.random.rand())
            for idx, k in enumerate(issuer_ids):
                issuer = issuer_names[idx]
                rho = correlation[issuer]
                rr = YY * rho
                YY_ido = norm.ppf(np.random.rand())
                rr_idio = np.sqrt(1 - (rho**2)) * YY_ido
                rr_all = rr + rr_idio
                rating_id = issuer_info.loc[issuer_info["IssuerID"] == k, "RatingID"].iloc[0]
                cutoffs_vec = np.matrix(cutOffs[:, rating_id]).T
                rating = np.array(rr_all < cutoffs_vec)
                rate_idx = len(rating) - np.sum(rating, 0)
                col_idx = rate_idx[0].astype(int)
                V_t = states.loc[issuer].iloc[col_idx]
                Loss_t = V_t - EV.loc[issuer].iloc[0]
                Loss[i, idx] = Loss_t
                rr_scenarios[i, idx] = rr_all

        return Loss, rr_scenarios, issuer_ids, issuer_names

    def get_loss_distribution(self, mc_scenario_values: np.array):
        """Computes loss distribution for portfolio after monte-carlo-simulation.

        Returns:
            Array: Portfolio loss distribution.
        """
        loss_distribution = np.sum(mc_scenario_values, 1)

        return loss_distribution

    def get_portfolio_VaR(self, loss_distribution: np.array):
        """Computes Credit Value at Risk for specific portfolio and confidence level.

        Returns:
            Float: Portfolio Value at Risk of specific confidence level.
        """
        Port_Var = -1 * np.percentile(loss_distribution, self.confidencelevel)

        return Port_Var

    def get_portfolio_ES(self, loss_distribution: np.array):
        """Computes expected shortfall for specific portfolio and confidence level.

        Returns:
            Float: Expected shorfall of porfolio.
        """

        expectedShortfall = -1 * np.mean(loss_distribution[loss_distribution < np.percentile(loss_distribution, self.confidencelevel)])

        return expectedShortfall
