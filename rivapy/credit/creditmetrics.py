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

from numpy.linalg import cholesky

class creditMetricsModel():
    def __init__(self, 
                # rho : float, 
                # n_issuer : int, 
                n_simulation : int , 
                transition_matrix : np.matrix, 
                position_data : pd.DataFrame,
                issuer_data : pd.DataFrame,
                stock_data: pd.DataFrame, 
                r : float, 
                t : float, 
                RR : float,
                confidencelevel : int,
                seed : int = None):
        """_summary_

        Args:
            n_simulation (int): Number of simulation, which should be carried out
            transition_matrix (np.matrix): Transition matrix (format np.matrix). S&P 8x8 matrix is integrated.
            position_data (pd.DataFrame): Dataframe with position data. Specific format is needed.
            issuer_data (pd.DataFrame): Dataframe with issuer data. Specific format is needed.
            stock_data (pd.DataFrame): Dataframe with stock data. Stock data needs to inlclude close values of the differen issuers as well as a reference time series (e.g. Dax)
            r (float): Risk-free rate. Needed to comupute expected value of positions as well as different states during transition process.
            t (float): Dipositon horizon for calculation of credit risk.
            RR (float): Fix recovery rate for CVaR calaculation
            confidencelevel (int): Used confidence level in VaR-Calculation. Format Int.
            seed (int, optional): Seed for random number generator. Defaults to None.
        """

        # self.rho = rho
        # self.n_issuer = n_issuer
        self.n_simulation = n_simulation
        self.transition_matrix = transition_matrix
        self.position_data = position_data
        self.issuer_data = issuer_data
        self.stock_data = stock_data
        self.r = r
        self.t = t
        self.RR = RR
        self.confidencelevel = confidencelevel
        self.seed = seed
        self.n_issuer = issuer_data["IssuerID"].nunique()

    def mergePositionsIssuer(self):
        """Merges position dataframe with issuer dataframe to obtain rating-data for each position

        Returns:
            DataFrame: Returns adjusted position dataframe.
        """
        rating_map = pd.DataFrame({'Rating': ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"], 'RatingID': [0, 1, 2, 3, 4, 5, 6, 7]})
        issuer_adj = self.issuer_data.merge(rating_map, on = "Rating", how = "left")
        positions_adj = self.position_data.merge(issuer_adj[["IssuerID","Rating","RatingID"]], on = "IssuerID", how = "left")

        return positions_adj
    
    def get_correlation (self):
        """Calculates correlation pairs for issuer with a specific reference time series.

        Returns:
            DataFrame: Dataframe with correlation coefficient for each issuer.
        """

        mergedData = self.stock_data.drop(['Date'], axis=1)
        returns = mergedData.pct_change()

        correlation_mat = returns.corr()
        corr_pairs = correlation_mat.unstack()['Dax']
        return corr_pairs

    
    def get_cutoffs_rating(self):
        """Computes cutoffs for each initial rating based on input transition matrix. 
        The inverse function of the standard normal distribution is used to get specific thresholds.

        Returns:
            DataFrame: Dataframe with arrays including thresholds for each initial rating.
        """
        Z=np.cumsum(np.flipud(self.transition_matrix.T),0)
        Z[Z>=(1-1/1e12)] = 1-1/1e12;
        Z[Z<=(0+1/1e12)] = 0+1/1e12;

        CutOffs=norm.ppf(Z,0,1) # compute cut offes by inverting normal distribution
        return CutOffs

    def get_credit_spreads(self, LGD):
        """Computes credit spreads for every rating based on the following formula
        -np.log(1-LGD*PD_t)/1

        Args:
            LGD (_type_): Instrument-specific LGD can be used in this computation.

        Returns:
            DataFrame: Dataframe with Credit spreads for each initial rating.
        """
        # credit spread implied by transmat
        PD_t = self.transition_matrix[:,-1] # default probability at t
        credit_spread = -np.log(1-LGD*PD_t)/1
        
        return credit_spread
    
    def get_expected_value (self):
        """Calculates expected value of every position based on exposure and credit spread for initial rating class.

        Returns:
            DataFrame: Dataframe including expected values.
        """
        # positions = self.mergePositionsIssuer()
        positions = self.get_issuer_groups()
        exposure = np.matrix(positions["Exposure"]).T
        # print(exposure)
        idx = positions["RatingID"]
        # print(idx)
        LGD = 1-self.RR
        credit_spread = self.get_credit_spreads(LGD)
        # print(credit_spread)
        EV = np.multiply(exposure, np.exp(-(self.r+credit_spread[idx])*self.t))

        return EV
    
    def get_states (self):
        """Calculates matrix of present values for every position and every possible future rating.

        Returns:
            DataFrame: Dataframe with all possible present values.
        """
        positions = self.get_issuer_groups()
        LGD = 1-self.RR
        recover = self.RR
        credit_spread = self.get_credit_spreads(LGD)
        cp = np.tile(credit_spread.T,[positions["IssuerID"].nunique(),1])
        exposure = np.matrix(positions["Exposure"]).T
        state = np.multiply(exposure,np.exp(-(self.r+cp)*self.t))
        state = np.append(state,np.multiply(exposure,recover),axis=1) #last column is default case
        states = np.fliplr(state) # keep in same order as credit cutoff

        return states
    
    def get_issuer_groups(self):
        df_positions_grouped = self.mergePositionsIssuer()
        df_positions_grouped = df_positions_grouped[['IssuerID', 'IssuerName', 'RecoveryRate', 'Rating', 'RatingID','Exposure']]
        df_positions_grouped = df_positions_grouped.groupby(['IssuerID', 'IssuerName', 'RecoveryRate', 'Rating', 'RatingID'], as_index=False).sum()

        return df_positions_grouped
    
    def mc_calculation(self):
        """Monte-Carlo simulation of portfolio based on positions, issuer, correlation and transition matrix. 
         In every simulation step the return of an issuer will be simulated:
         - Therefore the return of the Benchmark (Y) will be simulated and multiplied with the issuer-specific correlation. This random number is consistent for
         every position during one simulation step.
         - Aftewards the return of a specific issuer will be simulated and multiplied with the idiosyncratic risk factor (1-p)
         This results in the simulated return for every position in every simulation step
          𝑟𝑘=𝜌𝑌+√(1−𝜌^2 ) 𝑍𝑘

        Returns:
            Array2d: Occurring losses during simulation for every position.
        """
        # c = get_cholesky_distribution(rho, n_issuer)
        # cut = get_cut_ratings(transition_matrix, position_data)
        # positions = self.mergePositionsIssuer()
        positions = self.get_issuer_groups()
        issuer = self.issuer_data
        correlation = self.get_correlation()
        cutOffs = self.get_cutoffs_rating()
        states = self.get_states ()
        EV = self.get_expected_value ()
        # n_positions = positions["InstrumentID"].nunique()
        n_issuer = positions["IssuerID"].nunique()
        Loss = np.zeros((self.n_simulation,n_issuer))
        np.random.seed(self.seed)

        for i in range(0,self.n_simulation):
            YY = norm.ppf(np.random.rand())
            # rr=c*YY.T
            # rr = YY*self.rho
            for k in range (0, n_issuer):
                # n_positions_issuer = positions['InstrumentID'][positions['IssuerID']==issuer.loc[k,"IssuerID"]].nunique()
                # positions_issuer = positions[positions['IssuerID']==issuer.loc[k,"IssuerID"]]
                # positions_issuer.reset_index(inplace = True, drop = True)
                rho = correlation[positions.loc[k,'IssuerName']]
                rr = YY*rho
                YY_ido = norm.ppf(np.random.rand())
                #corr_idio=np.sqrt((1-(c*c)))
                rr_idio=np.sqrt(1-(rho**2))*YY_ido
                # print(rr_idio)
                rr_all=rr+rr_idio
                # print(rr)
                # for j in range (0,n_positions_issuer):
                    # rho = correlation[positions.loc[j,'IssuerName']]
                    # rr = YY*rho
                    # YY_ido = norm.ppf(np.random.rand())
                    # #corr_idio=np.sqrt((1-(c*c)))
                    # rr_idio=np.sqrt(1-(rho**2))*YY_ido
                    # print(j)
                    # print(n_positions_issuer)
                    # print(positions_issuer)
                    # rr_all=rr+rr_idio
                    # print(rr_all)
                rating = np.array(rr_all<np.matrix(cutOffs[:,positions.loc[k,"RatingID"]]).T)
                # print(rating)
                rate_idx = len(rating) - np.sum(rating,0)
                # print(rate_idx)
                col_idx = rate_idx
                V_t = states[k,col_idx] # retrieve the corresponding state value of the exposure
                Loss_t = V_t-EV.item(k)
                # print(Loss_t)
                Loss[i,k] = Loss_t
                    # print(j)
                    # print(k)
                    # print(Loss)

        # Portfolio_MC_Loss = np.sum(Loss,1)
        return Loss 

    def get_Loss_distribution (self):
        """Computes loss distribution for portfolio after monte-carlo-simulation.

        Returns:
            Array: Portfolio loss distribution.
        """
        Loss = self.mc_calculation()
        Portfolio_MC_Loss = np.sum(Loss,1)

        return Portfolio_MC_Loss

    def get_portfolio_VaR(self):
        """Computes Credit Value at Risk for specific portfolio and confidence level.

        Returns:
            Float: Portfolio Value at Risk of specific confidence level.
        """
        loss_Distribution = self.get_Loss_distribution()
        Port_Var = -1*np.percentile(loss_Distribution,self.confidencelevel)

        return Port_Var

    def get_portfolio_ES(self):
        """Computes expected shortfall for specific portfolio and confidence level.

        Returns:
            Float: Expected shorfall of porfolio.
        """
        loss_Distribution = self.get_Loss_distribution()
        portVar = self.get_portfolio_VaR()

        expectedShortfall = -1*np.mean(loss_Distribution[loss_Distribution<-1*portVar])

        return expectedShortfall


