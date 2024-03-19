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
from rivapy.instruments.specifications import EuropeanVanillaSpecification
from rivapy.pricing.vanillaoption_pricing import (
    VanillaOptionDeepHedgingPricer,
    DeepHedgeModel,
)
from scipy.special import comb

import analysis

from sys import exit

# timegrid = np.linspace(0.0, days*1.0/365.0, days*24)
# forward_expiries = [timegrid[-1]]


model = GBM(drift=0.0, volatility=0.2)

refdate = dt.datetime(2023, 1, 1)
transaction_cost = 0.01
days = 30
issuer = "DBK"
seclevel = "COLLATERALIZED"
tpe = "CALL"  # Change to 'PUT' if you want to calculate the price of an european put option.
expiry = refdate + dt.timedelta(days=30)
strike = 1.0

repo = analysis.Repo(
    "/home/doeltz/doeltz/development/RiVaPy/sandbox/vanillaoption/experiments/"
)

reg = {
    "mean_variance": [0.0],
    "exponential_utility": [5.0, 10.0],  # , 15.0, 20.0] ,
    "expected_shortfall": [0.1],
}
spec = EuropeanVanillaSpecification(
    "Test_call",
    tpe,
    expiry,
    strike,
    issuer=issuer,
    sec_lvl=seclevel,
    curr="EUR",
    udl_id="ADS",
    share_ratio=1,
)


for nb_neurons in [64]:  # 16,32,64]:
    for depth in [3]:  #  [2,3,4]:
        for tc in [0]:  # .,1.e-10,0.0001,0.001,0.01]:
            for initial_lr in [5e-4]:  #  [1e-4, 5e-4, 1e-3, 5e-3]:
                for test_weighted_paths in [True]:  #  , False]:
                    for batch_size in [128]:  # ,256,4*256, 4*4*256]:
                        pricing_results = repo.run(
                            refdate,
                            spec,
                            model,
                            rerun=False,
                            depth=depth,
                            nb_neurons=nb_neurons,
                            n_sims=100_000,
                            regularization=0.0,
                            epochs=800,
                            verbose=1,
                            tensorboard_logdir="logs/"
                            + dt.datetime.now().strftime("%Y%m%dT%H%M%S"),
                            initial_lr=initial_lr,  # 5e-4,
                            decay_steps=6_000,
                            batch_size=batch_size,
                            decay_rate=0.95,
                            seed=42,
                            transaction_cost={"ADS": [tc]},
                            test_weighted_paths=test_weighted_paths,
                        )