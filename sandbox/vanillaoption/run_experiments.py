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
    DeepHedgeModel,
)
from scipy.special import comb

import analysis

from sys import exit


model = [GBM(drift=0.0, volatility=0.25),HestonForDeepHedging(rate_of_mean_reversion = 0.6067,long_run_average = 0.0707,
                  vol_of_vol = 0.2928, correlation_rho = -0.757),HestonWithJumps(rate_of_mean_reversion = 0.4963,long_run_average = 0.065,
                  vol_of_vol = 0.2286, correlation_rho = -0.99,muj = 0.1791,sigmaj = 0.1346, lmbda = 0.1382),BNS(rho =-4.675,lmbda=0.5474,b=18.6075,a=0.6069)]

repo = analysis.Repo(
    "./experiments1"
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

count = 0
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
                udl_id="ADS4",
                share_ratio=1,
                long_short_flag=long_short_flag
            )
        spec.append(ins)



for tc in [0]:#[1.e-10,0.0001,0.001,0.01]:
    pricing_results = repo.run(
                            refdate,
                            spec,
                            model,
                            rerun=False,
                            depth=3,
                            nb_neurons=16,
                            n_sims=200_000,
                            regularization=0.0,
                            epochs=100,
                            verbose=1,
                            tensorboard_logdir="logs/"
                            + dt.datetime.now().strftime("%Y%m%dT%H%M%S"),
                            initial_lr=0.005,  # 5e-4,
                            decay_steps=16_000,
                            batch_size=64,
                            decay_rate=0.95,
                            seed=42,
                            days=int(np.max(days))
                        )