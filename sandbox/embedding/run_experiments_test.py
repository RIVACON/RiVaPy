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
from rivapy.instruments.specifications import EuropeanVanillaSpecification, BarrierOptionSpecification
from rivapy.pricing.vanillaoption_pricing import (
    VanillaOptionDeepHedgingPricer,
    DeepHedgeModelwEmbedding,
)
from scipy.special import comb

import analysis

from sys import exit


import ast  
with open('/home/doeltz/doeltz/development/RiVaPy/sandbox/embedding/model_params_dict.txt') as f: 
    data = f.read() 
model_params = ast.literal_eval(data) 

model = []

loop = 1#32#
for i in range(loop):
    #model.append(HestonForDeepHedging(rate_of_mean_reversion = 0.6067,long_run_average = 0.0707,
    #              vol_of_vol = 0.2928, correlation_rho = -0.757,v0 = 0.0654))
    #model.append(GBM(0.,vol_list[i]))
    model.append(GBM(drift=model_params['GBM']['drift'][i],volatility=model_params['GBM']['vol'][i]))
    # model.append(HestonForDeepHedging(rate_of_mean_reversion = model_params['Heston']['rate_of_mean_reversion'][i],
    #                                   long_run_average = model_params['Heston']['long_run_average'][i],
    #                                   vol_of_vol = model_params['Heston']['vol_of_vol'][i], 
    #                                   correlation_rho = model_params['Heston']['correlation_rho'][i],
    #                                   v0 = model_params['Heston']['v0'][i]))
    # model.append(HestonWithJumps(rate_of_mean_reversion = model_params['Heston with Jumps']['rate_of_mean_reversion'][i],
    #                              long_run_average = model_params['Heston with Jumps']['long_run_average'][i],
    #                              vol_of_vol = model_params['Heston with Jumps']['vol_of_vol'][i], 
    #                              correlation_rho = model_params['Heston with Jumps']['correlation_rho'][i],
    #                              muj = 0.1791,sigmaj = 0.1346, 
    #                              lmbda = model_params['Heston with Jumps']['lmbda'][i],
    #                              v0 = model_params['Heston with Jumps']['v0'][i]))
    # model.append(BNS(rho =model_params['BNS']['rho'][i],
    #                  lmbda=model_params['BNS']['lmbda'][i],
    #                  b=model_params['BNS']['b'][i],
    #                  a=model_params['BNS']['a'][i],
    #                  v0 = model_params['BNS']['v0'][i]))




repo = analysis.Repo(
    "./test"
)

reg = {
    "mean_variance": [0.0],
    "exponential_utility": [5.0, 10.0],  # , 15.0, 20.0] ,
    "expected_shortfall": [0.1],
}

#spec = {}

strike = [0.8]#[0.85,0.9,0.95,1.]#, 0.9, 1.0, 1.1, 1.2]
days = [30]#[20,40, 60, 80, 100, 120]
refdate = dt.datetime(2023, 1, 1)
issuer = "DBK"
seclevel = "COLLATERALIZED"
tpe = "DOB_CALL"  # Change to 'PUT' if you want to calculate the price of an european put option.
long_short_flag = 'long'


spec = []

count = 0
for i in range(len(strike)):
    for j in range(len(days)):
        count = count + 1
        expiry = refdate + dt.timedelta(days=days[j])
        spec.append(BarrierOptionSpecification( #EuropeanVanillaSpecification(
                    'P'+str(count-1)+str(tpe)+'K'+str(strike[i])+'T'+str(days[j]),
                    tpe,
                    expiry,
                    strike[i],
                    barrier=0.95,
                    issuer=issuer,
                    sec_lvl=seclevel,
                    curr="EUR",
                    udl_id="ADS",
                    share_ratio=1,
                    long_short_flag=long_short_flag,
                    portfolioid=count-1
                ))
        

        
n_sims = loop*100#16000*4
for emb_size in [64]:
    for seed in [42]:
        #for tc in [1e-10,0.0001,0.001,0.01]:
        pricing_results = repo.run(
                            refdate,
                            spec,
                            model,
                            rerun=False,
                            depth=3,
                            nb_neurons=128,
                            n_sims=n_sims,
                            regularization=10.,
                            epochs=1,#300,
                            verbose=1,
                            tensorboard_logdir="logs/"
                            + dt.datetime.now().strftime("%Y%m%dT%H%M%S"),
                            initial_lr=0.0005, 
                            decay_steps=16_000,
                            batch_size=2024,
                            decay_rate=0.9,
                            seed=seed,
                            days=int(np.max(days)),
                            embedding_size=emb_size,
                            embedding_size_port=1,
                            transaction_cost={'ADS':[0.01]}#'DOB_ADS':[0.01]},
                            #loss = "exponential_utility"
                        )