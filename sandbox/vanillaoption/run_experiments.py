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
from rivapy.instruments.specifications import EuropeanVanillaSpecification
from rivapy.pricing.vanillaoption_pricing import (
    VanillaOptionDeepHedgingPricer,
    DeepHedgeModel,
)
from scipy.special import comb

import analysis

from sys import exit



#model = GBM(drift=0.0, volatility=0.2)
model = HestonForDeepHedging(rate_of_mean_reversion = 1.,long_run_average = 0.04, vol_of_vol = 2., correlation_rho = -0.7)

refdate = dt.datetime(2023, 1, 1)
transaction_cost = 0.01
days = 30
issuer = "DBK"
seclevel = "COLLATERALIZED"
tpe = "CALL"  # Change to 'PUT' if you want to calculate the price of an european put option.
expiry = refdate + dt.timedelta(days=30)
strike = 1.0
long_short_flag = 'long'

repo = analysis.Repo(
    "./experiments"
)

reg = {
    "mean_variance": [0.0],
    "exponential_utility": [5.0, 10.0],  # , 15.0, 20.0] ,
    "expected_shortfall": [0.1],
}
spec = EuropeanVanillaSpecification(
    "Test_Call",
    tpe,
    expiry,
    strike,
    issuer=issuer,
    sec_lvl=seclevel,
    curr="EUR",
    udl_id="ADS",
    share_ratio=1,
    long_short_flag=long_short_flag
)


for tc in [0]:#[1.e-10,0.0001,0.001,0.01]:
    pricing_results = repo.run(
                            refdate,
                            spec,
                            model,
                            rerun=False,
                            depth=3,
                            nb_neurons=16,
                            n_sims=100_000,
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
                            transaction_cost={"ADS": [tc]},
                            modelname = 'Heston with Volswap'
                        )