from __future__ import division
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
                rho : float, 
                n_issuer : int, 
                n_simulation : int , 
                transition_matrix : np.matrix, 
                position_data : pd.DataFrame, 
                r : float, 
                t : float, 
                confidencelevel : int):
        self.rho = rho
        self.n_issuer = n_issuer
        self.n_simulation = n_simulation
        self.transition_matrix = transition_matrix
        self.position_data = position_data
        self.r = r
        self.t = t
        self.confidencelevel = confidencelevel
    
    def get_cutoffs_rating(self):
        Z=np.cumsum(np.flipud(self.transition_matrix.T),0)
        Z[Z>=(1-1/1e12)] = 1-1/1e12;
        Z[Z<=(0+1/1e12)] = 0+1/1e12;

        CutOffs=norm.ppf(Z,0,1) # compute cut offes by inverting normal distribution
        return CutOffs

    def get_credit_spreads(self, LGD):
        # credit spread implied by transmat
        PD_t = self.transition_matrix[:,-1] # default probability at t
        credit_spread = -np.log(1-LGD*PD_t)/1
        
        return credit_spread
    
    def get_expected_value (self):
        exposure = np.matrix(self.position_data["Exposure"]).T
        # print(exposure)
        idx = self.position_data["RatingID"]
        # print(idx)
        LGD = 0.45
        credit_spread = self.get_credit_spreads(self, LGD)
        # print(credit_spread)
        EV = np.multiply(exposure, np.exp(-(self.r+credit_spread[idx])*self.t))

        return(EV)
    
    def get_states (self):
        # bond state variable for security Value
        LGD = 0.45
        recover = 0.55
        credit_spread = self.get_credit_spreads(self, LGD)
        cp = np.tile(credit_spread.T,[self.position_data["InstrumentID"].nunique(),1])
        # print(cp)
        exposure = np.matrix(self.position_data["Exposure"]).T
        # print(exposure)
        state = np.multiply(exposure,np.exp(-(self.r+cp)*self.t))
        # print(state)
        state = np.append(state,np.multiply(exposure,recover),axis=1) #last column is default case
        # print(state)
        states = np.fliplr(state) # keep in same order as credit cutoff
        # print(states)

        return states
    
    def mc_calculation(self):
        # c = get_cholesky_distribution(rho, n_issuer)
        # cut = get_cut_ratings(transition_matrix, position_data)
        cutOffs = self.get_cutoffs_rating(self)
        states = self.get_states (self)
        EV = self.get_expected_value (self)
        n_positions = self.position_data["InstrumentID"].nunique()
        Loss = np.zeros((self.n_simulation,n_positions))
        # np.random.seed(1)

        for i in range(0,self.n_simulation):
            YY = norm.ppf(np.random.rand())
            # rr=c*YY.T
            rr = YY*self.rho
            for j in range (0,n_positions):
                YY_ido = norm.ppf(np.random.rand())
                #corr_idio=np.sqrt((1-(c*c)))
                rr_idio=np.sqrt(1-(self.rho**2))*YY_ido
                # print(rr_idio)
                rr_all=rr+rr_idio
                # print(rr_all)
                rating = np.array(rr_all<np.matrix(cutOffs[:,self.position_data.loc[j,"RatingID"]]).T)
                rate_idx = len(rating) - np.sum(rating,0)
                # print(rate_idx)
                col_idx = rate_idx
                V_t = states[j,col_idx] # retrieve the corresponding state value of the exposure
                Loss_t = V_t-EV.item(j)
                # print(Loss_t)
                Loss[i,j] = Loss_t
                # print(Loss)

        # Portfolio_MC_Loss = np.sum(Loss,1)
        return Loss 

    def get_Loss_distribution (self):
        Loss = self.mc_calculation(self)
        Portfolio_MC_Loss = np.sum(Loss,1)

        return Portfolio_MC_Loss

    def get_portfolio_VaR(self):
        loss_Distribution = self.get_Loss_distribution(self)
        Port_Var = -1*np.percentile(loss_Distribution,self.confidencelevel)

        return Port_Var

    def get_portfolio_ES(self):
        loss_Distribution = self.get_Loss_distribution(self)
        portVar = self.get_portfolio_VaR(self)

        expectedShortfall = -1*np.mean(loss_Distribution[loss_Distribution<-1*portVar])

        return expectedShortfall


