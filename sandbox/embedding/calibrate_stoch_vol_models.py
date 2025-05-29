from typing import Tuple
import pandas as pd
import numpy as np
import datetime as dt
import sys
import os
import json
import logging
sys.path.insert(0, "C:/Users/doeltz/development/analytics_tools/install/python_modules")

from rivapy.models.calibration import calibrate
import rivapy.models.factory as factory 
from rivapy.tools.interfaces import _JSONDecoder, _JSONEncoder
from rivapy.models.heston_for_DH import HestonForDeepHedging
from rivapy.models.gbm import GBM
from rivapy.models.heston_with_jumps import HestonWithJumps
from rivapy.models.barndorff_nielsen_shephard import BNS
import matplotlib.pyplot as plt
import pyvacon

logger =  logging.getLogger()
logger.setLevel(logging.INFO)

def transform_to_X_price_and_strike(valDate: dt.datetime, 
                        T: dt.datetime, 
                        fwd: pyvacon.finance.marketdata.ForwardCurve,
                         dc: pyvacon.finance.marketdata.DiscountCurve, 
                         rPrice: float, 
                         rStrike: float)->Tuple[float,float]:
    SV = fwd.SV(valDate, T)
    D = fwd.discountedFutureCashDivs(valDate, T)
    df = 1.0
    df = dc.value(valDate, T)
    F = fwd.value(valDate, T)
    xPrice = rPrice / (df * (F - D) * SV)
    xStrike = pyvacon.finance.utils.computeXStrike(rStrike, F, D)
    return xStrike, xPrice

data_dir = 'C:/Users/doeltz/development/vol_calib_data/deep_hedging/'
calibrated_models = []
info =  []

for filename in os.listdir(data_dir):
    if not filename.endswith(".json"):
        continue
    d = os.path.join(data_dir, filename)
    logger.info('Loading calibration data from %s', data_dir+d)
    calib_data = pyvacon.finance.calibration.BaseCalibrationData.load(d)
    fwd = calib_data.fwdCurve
    dsc = calib_data.dsc
    qt = calib_data.quoteTable
    cal_date = pyvacon.converter.create_datetime(calib_data.calDate)

    expiries_ = [cal_date + dt.timedelta(days=days) for days in [30, 180, 365]]
    x_strikes_ = np.array([[0.9, 1.0, 1.1],
                        [0.8,1.0,1.2],
                        [0.8,1.0,1.2]])
    x_prices = []
    ttm = []
    x_strikes = []
    dc = pyvacon.finance.definition.DayCounter('Act365Fixed')
    for i in range(len(expiries_)):
        ttm_ = dc.yf(cal_date, expiries_[i])
        for j in range(len(x_strikes_[i])):
            ttm.append(ttm_)
            x_strikes.append(x_strikes_[i][j])
            vol = calib_data.startVol.calcImpliedVol(cal_date, expiries_[i], x_strikes_[i][j])
            x_prices.append(pyvacon.finance.utils.calcEuropeanCallPrice(x_strikes_[i][j],ttm_,1.0,1.0,vol))
            
    models = [
        HestonForDeepHedging(rate_of_mean_reversion=0.6,long_run_average=0.2*0.2,
                                vol_of_vol=1.1, correlation_rho=-0.7,v0=0.2*0.2),
        HestonWithJumps(rate_of_mean_reversion=0.6,long_run_average=0.2*0.2,
                            vol_of_vol=1.1,correlation_rho=-0.7,
                            muj = 0.1, sigmaj=0.1, lmbda=0.1,v0 = 0.2*0.2),
        BNS(a=0.1, b=0.1, lmbda=0.1, rho=-0.2, v0=0.1),
    ]

    for m in models:
        logger.info('Calibrating model: %s', m.__class__.__name__)
        result = calibrate(m, x_prices, ttm, x_strikes)
        if not result.success:
            print('Calibration of model not successful!')
        calibrated_models.append(m.to_dict())
        info.append({'model':m.__class__.__name__, 'success':bool(result.success), 'data': d})
        #m = factory.create(m.to_dict())

with open("calibrated_models.json", "w") as f:
    json.dump(calibrated_models, f, cls=_JSONEncoder)
    #model_call_prices = model.compute_call_price(1.0, x_strikes, ttm)
with open("calibration_info.json", "w") as f:
    json.dump(info, f, cls=_JSONEncoder)