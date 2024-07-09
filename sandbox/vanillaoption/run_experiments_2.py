import datetime as dt
import sys

sys.path.insert(0, "../..")
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from rivapy.tools.datetime_grid import DateTimeGrid
from rivapy.models.gbm import GBM
from rivapy.models.heston_for_DH import HestonForDeepHedging
from rivapy.models.heston_with_jumps import HestonWithJumps
from rivapy.models.barndorff_nielsen_shephard import BNS
from rivapy.instruments.specifications import EuropeanVanillaSpecification
from rivapy.pricing.vanillaoption_pricing import (
    VanillaOptionDeepHedgingPricer,
    DeepHedgeModelwEmbedding,
)
from scipy.special import comb

import analysis

from sys import exit

import ast  
with open('model_params_dict.txt') as f: 
    data = f.read() 
model_params = ast.literal_eval(data) 

model = []

vol_list = [0.1,0.2,0.3,0.4]
loop = 1#len(vol_list)
for i in [47]:#range(loop):
    #model.append(GBM(0.,vol_list[i]))
    #model.append(GBM(drift=model_params['GBM']['drift'][i],volatility=model_params['GBM']['vol'][i]))
    model.append(HestonForDeepHedging(rate_of_mean_reversion = model_params['Heston']['rate_of_mean_reversion'][i],
                                      long_run_average = model_params['Heston']['long_run_average'][i],
                                      vol_of_vol = model_params['Heston']['vol_of_vol'][i], 
                                      correlation_rho = model_params['Heston']['correlation_rho'][i],
                                      v0 = model_params['Heston']['v0'][i]))
    #model.append(HestonWithJumps(rate_of_mean_reversion = model_params['Heston with Jumps']['rate_of_mean_reversion'][i],
    #                             long_run_average = model_params['Heston with Jumps']['long_run_average'][i],
    #                             vol_of_vol = model_params['Heston with Jumps']['vol_of_vol'][i], 
    #                             correlation_rho = model_params['Heston with Jumps']['correlation_rho'][i],
    #                             muj = 0.1791,sigmaj = 0.1346, 
    #                             lmbda = model_params['Heston with Jumps']['lmbda'][i],
    #                             v0 = model_params['Heston with Jumps']['v0'][i]))
    #model.append(BNS(rho =model_params['BNS']['rho'][i],
    #                 lmbda=model_params['BNS']['lmbda'][i],
    #                 b=model_params['BNS']['b'][i],
   #                  a=model_params['BNS']['a'][i],
   #                  v0 = model_params['BNS']['v0'][i]))


#model = [HestonForDeepHedging(rate_of_mean_reversion = 0.06067,long_run_average = 0.0707,
#                  vol_of_vol = 0.2928, correlation_rho = -0.757,v0 = 0.009),
#            HestonForDeepHedging(rate_of_mean_reversion = 0.06067,long_run_average = 0.0707,
#                  vol_of_vol = 0.2928, correlation_rho = -0.757,v0 = 0.04),
#            HestonForDeepHedging(rate_of_mean_reversion = 0.06067,long_run_average = 0.0707,
#                  vol_of_vol = 0.2928, correlation_rho = -0.757,v0 = 0.09),
#            HestonForDeepHedging(rate_of_mean_reversion = 0.06067,long_run_average = 0.0707,
#                  vol_of_vol = 0.2928, correlation_rho = -0.757,v0 = 0.16),
#                GBM(drift=0., volatility=0.1),GBM(drift=0., volatility=0.2),GBM(drift=0., volatility=0.3),GBM(drift=0., volatility=0.4),
#            BNS(rho =-0.4675,lmbda=0.5474,b=18.6075,a=0.6069,v0 = 0.006),
#            BNS(rho =-0.4675,lmbda=0.5474,b=18.6075,a=0.6069,v0 = 0.033),
#            BNS(rho =-0.4675,lmbda=0.5474,b=18.6075,a=0.6069,v0 = 0.08),
#            BNS(rho =-0.4675,lmbda=0.5474,b=18.6075,a=0.6069,v0 = 0.15),
#            HestonWithJumps(rate_of_mean_reversion = 0.04963,long_run_average = 0.065,
#                vol_of_vol = 0.2286, correlation_rho = -0.99,muj = 0.1791,sigmaj = 0.1346, lmbda = 0.1382,v0 = 0.007),
#            HestonWithJumps(rate_of_mean_reversion = 0.04963,long_run_average = 0.065,
#                vol_of_vol = 0.2286, correlation_rho = -0.99,muj = 0.1791,sigmaj = 0.1346, lmbda = 0.1382,v0 = 0.032),
#            HestonWithJumps(rate_of_mean_reversion = 0.04963,long_run_average = 0.065,
#                vol_of_vol = 0.2286, correlation_rho = -0.99,muj = 0.1791,sigmaj = 0.1346, lmbda = 0.1382,v0 = 0.085),
#            HestonWithJumps(rate_of_mean_reversion = 0.04963,long_run_average = 0.065,
#                vol_of_vol = 0.2286, correlation_rho = -0.99,muj = 0.1791,sigmaj = 0.1346, lmbda = 0.1382,v0 = 0.15)]


repo = analysis.Repo(
    "./test_ins"
)

reg = {
    "mean_variance": [0.0],
    "exponential_utility": [5.0, 10.0],  # , 15.0, 20.0] ,
    "expected_shortfall": [0.1],
}

spec = []

strike = [1.]# [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
days = [30]#[20,40, 60, 80, 100, 120]
refdate = dt.datetime(2023, 1, 1)
issuer = "DBK"
seclevel = "COLLATERALIZED"
tpe = "CALL"  # Change to 'PUT' if you want to calculate the price of an european put option.
long_short_flag = 'long'

count = 3
for i in range(len(strike)):
    for j in range(len(days)):
        count = count + 1
        expiry = refdate + dt.timedelta(days=days[j])
        ins = EuropeanVanillaSpecification(
                    "Test_Call"+str(count),
                    tpe,
                    expiry,
                    strike[i],
                    issuer=issuer,
                    sec_lvl=seclevel,
                    curr="EUR",
                    udl_id="ADS",
                    share_ratio=1,
                    long_short_flag=long_short_flag
                )
        spec.append(ins)


n_sims = 64000#loop*16000#*4
for emb_size in [1]:
    for seed in [0]:
        for tc in [0.01]:
            pricing_results = repo.run(
                            refdate,
                            spec,
                            model,
                            rerun=False,
                            depth=3,
                            nb_neurons=64,
                            n_sims=n_sims,
                            regularization=10.,
                            epochs=1000,
                            verbose=1,
                            tensorboard_logdir="logs/"
                            + dt.datetime.now().strftime("%Y%m%dT%H%M%S"),
                            initial_lr=0.005, 
                            decay_steps=16_000,
                            batch_size=256,
                            decay_rate=0.95,
                            seed=seed,
                            days=int(np.max(days)),
                            embedding_size=emb_size,
                            transaction_cost={'ADS':[tc]},
                            loss = "exponential_utility"
                        )