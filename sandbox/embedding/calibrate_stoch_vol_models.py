import pandas as pd
import scipy.optimize as scpo
import numpy as np
import sys
sys.path.insert(0, "../..")
from rivapy.models.calibration import calibrate
from rivapy.models.heston_for_DH import HestonForDeepHedging
from rivapy.models.gbm import GBM
from rivapy.models.heston_with_jumps import HestonWithJumps
from rivapy.models.barndorff_nielsen_shephard import BNS
import matplotlib.pyplot as plt

x_strikes_ = [0.5,1.0,1.5]
ttm_ = [0.1, 0.5, 1.0, 2.0]
x_strikes=[]
ttm=[]
vol = 0.3
prices = []

for j in range(len(x_strikes_)):
    for i in range(len(ttm_)):
        x_strikes.append(x_strikes_[j])
        ttm.append(ttm_[i])
        model = GBM(0.,vol)
        prices.append(model.compute_call_price(S0=1., K=x_strikes_[j], ttm=ttm_[i]))

model = HestonForDeepHedging(rate_of_mean_reversion=0.6,long_run_average=0.1,vol_of_vol=1.1, correlation_rho=-0.4,v0=0.02)
result = calibrate(model, np.array(prices), np.array(ttm), np.array(x_strikes))
print(result)