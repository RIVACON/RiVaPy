from __future__ import division
import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import List, Optional as _Optional
from rivapy.instruments.components import Issuer


class CreditMetricsModel:
    def __init__(
        self,
        n_simulation: int,
        transition_matrix: np.matrix,
        position_data: pd.DataFrame,
        issuer_data: List[Issuer],
        stock_data: pd.DataFrame,
        r: float,
        t: float,
        confidencelevel: float,
        seed: _Optional[int] = None,
        list_of_indices: _Optional[List[str]] = ["DAX", "SP"],
        mapping_countries_on_indices: _Optional[dict] = {
            "DE": "DAX",
            "US": "SP",
        },
    ):
        """CreditMetrics model initializer.

        Args:
            n_simulation (int): Number of simulations.
            transition_matrix (np.matrix): Transition matrix (format np.matrix).
                S&P 8x8 matrix is integrated.
            position_data (pd.DataFrame): DataFrame with position data. Specific format is needed.
            issuer_data (pd.DataFrame): List of Issuer objects containing issuer metadata.
            stock_data (pd.DataFrame): DataFrame with stock data. Must include closing
                prices of the different issuers as well as reference indices (e.g. DAX).
            r (float): Risk-free rate. Needed to compute expected value of positions and
                state valuations during the transition process.
            t (float): Time horizon for the credit risk calculation.
            confidencelevel (float): Confidence level used in VaR calculation (percentage).
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
        self.list_of_indices = list_of_indices
        self.mapping_countries_on_indices = mapping_countries_on_indices

    def merge_positions_issuer(self):
        """
        Merges position dataframe with issuer dataframe to obtain rating-data for each position.
        Maps all +/- Rating variants to the same RatingID.
        Returns:
            DataFrame: Returns adjusted position dataframe.
        """
        # Map all rating variants (including +/-) to a single RatingID
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

        # issuer_data is now a list of Issuer objects
        issuer_df = pd.DataFrame(
            [
                {
                    "IssuerID": issuer.obj_id,
                    "IssuerName": issuer.name,
                    "Rating": str(issuer.rating),
                    "Land": issuer.country,
                }
                for issuer in self.issuer_data
            ]
        )

        # Apply mapping
        issuer_adj = issuer_df.merge(rating_map, on="Rating", how="left")
        positions_adj = self.position_data.merge(issuer_adj[["IssuerID", "IssuerName", "Rating", "Land", "RatingID"]], on="IssuerID", how="left")

        return positions_adj

    def get_correlation(self):
        """Calculates correlation pairs for issuer with a specific reference time series.

        Returns:
            DataFrame: Dataframe with correlation coefficient for each issuer.
        """

        mergedData = self.stock_data.drop(["Date"], axis=1)
        returns = mergedData.pct_change()

        # Split returns into indices and stocks
        returns_indices = returns[self.list_of_indices]
        returns_stocks = returns.drop(self.list_of_indices, axis=1)
        # Compute correlations with different indices
        indices_correlation = returns_indices.corr()

        # Create an empty DataFrame for the results
        corr_pairs = pd.DataFrame()

        # Loop over the index columns and compute correlations with stocks
        for col_name in self.list_of_indices:
            # Compute the correlation of the current index series with all stock series
            correlation = returns_stocks.corrwith(returns[col_name])
            # Add the results as a new column to the result DataFrame
            corr_pairs[f"{col_name}"] = correlation
        return indices_correlation, corr_pairs

    def get_cutoffs_rating(self):
        """Compute cutoffs for each initial rating based on the transition matrix.

        The inverse cumulative distribution function of the standard normal is used to
        obtain thresholds corresponding to cumulative transition probabilities.

        Returns:
            np.ndarray: Array with threshold values for each target rating per initial rating.
        """
        Z = np.cumsum(np.flipud(self.transition_matrix.T), 0)
        Z[Z >= (1 - 1 / 1e12)] = 1 - 1 / 1e12
        Z[Z <= (0 + 1 / 1e12)] = 0 + 1 / 1e12

        CutOffs = norm.ppf(Z, 0, 1)  # compute cut offes by inverting normal distribution
        return CutOffs

    def get_credit_spreads(self, LGD, idx):
        """Compute credit spreads implied by the transition matrix.

        Formula used (per time horizon t): spread = -log(1 - LGD * PD_t) / t

        Args:
            LGD (pd.Series): Instrument-specific loss-given-default values (0-1).
            idx (pd.Series or list): RatingID index array aligning instruments to rows of PD_t.

        Returns:
            np.ndarray: Credit spread per instrument (shape: n_instruments x 1).
        """
        # credit spread implied by transmat
        PD_t = self.transition_matrix[:, -1]
        PD_vec = PD_t[idx]
        LGD_np = LGD.to_numpy().reshape(-1, 1)
        credit_spread = -np.log(1 - np.multiply(LGD_np, PD_vec)) / self.t
        return credit_spread

    def get_expected_value(self):
        """Calculate the expected present value of each instrument specified by issuer and recovery rate.

        Uses the instrument exposure, the risk-free rate and the credit spread implied
        by the transition matrix to discount the expected payoff over the time horizon.

        Returns:
            DataFrame: DataFrame including expected values (grouped by issuer).
        """
        positions = self.get_issuer_groups()
        exposure = np.matrix(positions["Exposure"]).T
        idx = positions["RatingID"]
        LGD = 1 - positions["RecoveryRate"]
        credit_spread = self.get_credit_spreads(LGD, idx)
        EV = np.multiply(exposure, np.exp(-(self.r + credit_spread) * self.t))
        EV = pd.DataFrame(EV, columns=["EV"])  # keep in same order as credit cutoff
        EV["issuer"] = positions["IssuerID"].to_list()
        EV = EV.groupby("issuer").sum()  # group by issuer to sum up expected values
        return EV

    def get_states(self):
        """Compute matrix of present values for each issuer's positions under all possible future ratings.

        Each column corresponds to a target rating (including default). The values are the
        discounted present values of the issuer's positions assuming the given rating outcome.

        Returns:
            DataFrame: DataFrame with present values by issuer and rating state.
        """
        positions = self.get_issuer_groups()
        LGD = 1 - np.array(positions["RecoveryRate"])
        PD_t = self.transition_matrix[:, -1]  # default probability at t
        credit_spread = -np.log(1 - PD_t * LGD.T)
        exposure = np.matrix(positions["Exposure"])
        state = np.multiply(exposure, np.exp(-(self.r + credit_spread) * self.t)).T
        state = np.append(state, np.multiply(exposure, np.matrix(positions["RecoveryRate"])).T, axis=1)  # last column is default case
        states = pd.DataFrame(np.fliplr(state), columns=["D", "C", "B", "BB", "BBB", "A", "AA", "AAA"])  # keep in same order as credit cutoff
        states["issuer"] = positions["IssuerID"].to_list()
        states = states.groupby("issuer").sum()
        return states

    def get_issuer_groups(self):
        df_positions_grouped = self.merge_positions_issuer()
        df_positions_grouped = df_positions_grouped[["IssuerID", "IssuerName", "RecoveryRate", "Rating", "RatingID", "Land", "Exposure"]]
        df_positions_grouped = df_positions_grouped.groupby(
            ["IssuerID", "IssuerName", "RecoveryRate", "Rating", "RatingID", "Land"], as_index=False
        ).sum()
        df_positions_grouped["Position_Index"] = [self.mapping_countries_on_indices.get(country) for country in df_positions_grouped["Land"]]

        return df_positions_grouped

    def mc_calculation(self):
        """
        Monte-Carlo simulation of portfolio based on positions, issuer, correlation and transition matrix.

        For each simulation step, the return of each issuer is simulated:
        - The return of the benchmark (Y) is simulated and multiplied with the issuer-specific correlation.
          This random number is consistent for every issuer during one simulation step.
        - Afterwards, the idiosyncratic return of each issuer is simulated and multiplied with the idiosyncratic
          risk factor sqrt(1-p^2).
        - This results in the simulated return for every issuer in every simulation step:
          r_k = rho * Y + sqrt(1 - rho^2) * Z_k

        For each issuer, the new rating is determined and the loss is calculated as the difference between the
        new value and the expected value.

        Returns:
            tuple:
                Loss (np.ndarray): Array of shape (n_simulation, n_issuer) with losses for each scenario and issuer.
                issuer_ids (np.ndarray): Array of issuer IDs, order matches Loss columns.
                issuer_names (list): List of issuer names, order matches Loss columns.
        """
        positions = self.get_issuer_groups()
        indices_correlation, corr_pairs = self.get_correlation()
        indices_cholesky = np.linalg.cholesky(indices_correlation)
        cutOffs = self.get_cutoffs_rating()
        states = self.get_states()
        EV = self.get_expected_value()
        issuer_info = positions[["IssuerName", "IssuerID", "Rating", "RatingID", "Position_Index"]].drop_duplicates()
        issuer_ids = issuer_info["IssuerID"].to_numpy()
        issuer_names = issuer_info["IssuerName"].to_list()
        Loss = pd.DataFrame(np.zeros((self.n_simulation, len(issuer_ids))), columns=issuer_ids, index=range(self.n_simulation))
        rr_scenarios = pd.DataFrame(np.zeros((self.n_simulation, len(issuer_ids))), columns=issuer_ids, index=range(self.n_simulation))
        np.random.seed(self.seed)

        # random numbers for indices
        normal_random_indices = np.random.randn(len(self.list_of_indices), self.n_simulation)
        YY = np.matmul(indices_cholesky, normal_random_indices)
        YY_idio = np.random.randn(len(issuer_ids), self.n_simulation)

        # Calculate Losses for each issuer
        # Iterate
        i = 0
        for idx in issuer_ids:
            # correlation between issuer and its assigned index
            issuer_name = issuer_info.loc[issuer_info["IssuerID"] == idx, "IssuerName"].iloc[0]
            index_name = issuer_info.loc[issuer_info["IssuerID"] == idx, "Position_Index"].iloc[0]
            correlation = corr_pairs.loc[issuer_name, index_name]
            rating_id = issuer_info.loc[issuer_info["IssuerID"] == idx, "RatingID"].iloc[0]
            cutoffs_vec = np.matrix(cutOffs[:, rating_id]).T
            rr = YY[self.list_of_indices.index(index_name), :] * correlation
            rr_idio = np.sqrt(1 - (correlation**2)) * YY_idio[i, :]
            rr_all = rr + rr_idio
            # Determine new rating by comparing the simulated score to cutoffs
            new_ratings = np.array(rr_all < cutoffs_vec)
            new_ratings_idx = len(new_ratings) - np.sum(new_ratings, 0)
            col_idx = new_ratings_idx.astype(int)
            V_t = states.loc[idx].iloc[col_idx]
            Loss_t = V_t.to_numpy() - EV.loc[idx].iloc[0]
            Loss.loc[:, idx] = Loss_t
            rr_scenarios.loc[:, idx] = rr_all
            i += 1

        return Loss, rr_scenarios, issuer_ids, issuer_names

    def get_loss_distribution(self, mc_scenario_values: np.ndarray):
        """Computes loss distribution for portfolio after monte-carlo-simulation.

        Returns:
            Array: Portfolio loss distribution.
        """
        loss_distribution = np.sum(mc_scenario_values, 1)

        return loss_distribution

    def get_portfolio_VaR(self, loss_distribution: np.ndarray):
        """Computes Credit Value at Risk for specific portfolio and confidence level.

        Returns:
            Float: Portfolio Value at Risk of specific confidence level.
        """
        Port_Var = -1.0 * np.percentile(loss_distribution, self.confidencelevel)

        return Port_Var

    def get_portfolio_ES(self, loss_distribution: np.ndarray):
        """Computes expected shortfall for specific portfolio and confidence level.

        Returns:
            Float: Expected shorfall of porfolio.
        """

        expectedShortfall = -1.0 * np.mean(loss_distribution[loss_distribution < np.percentile(loss_distribution, self.confidencelevel)])

        return expectedShortfall
