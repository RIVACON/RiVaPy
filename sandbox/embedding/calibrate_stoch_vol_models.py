from typing import Tuple
import pandas as pd
import numpy as np
import datetime as dt
import sys
sys.path.insert(0, "C:/Users/doeltz/development/analytics_tools/install/python_modules")
from rivapy.models.calibration import calibrate
from rivapy.models.heston_for_DH import HestonForDeepHedging
from rivapy.models.gbm import GBM
from rivapy.models.heston_with_jumps import HestonWithJumps
from rivapy.models.barndorff_nielsen_shephard import BNS
import matplotlib.pyplot as plt
import pyvacon

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

data_dir = 'C:/Users/doeltz/development/vol_calib_data/'
data = [
    #'CalibrationData_BYND_22393_1572_SSVI_215519.166143_213018.json',
    #'CalibrationData_PROX_931_786_SSVI_092907.157345_092552.json',
    #'CalibrationData_BYND_22393_4022_SSVI_205537.985527_203000.json',
    #'CalibrationData_WBA_100_1522_SSVI_214644.288587_213015.json',
    #'CalibrationData_SEDG_19648_1406_SSVI_215410.606244_213016.json',
    'CalibrationData_NVAX_18374_1905_SSVI_205344.749740_202959.json'
]
calib_data = pyvacon.finance.calibration.BaseCalibrationData.load(data_dir+data[0])
fwd = calib_data.fwdCurve
dsc = calib_data.dsc
qt = calib_data.quoteTable
cal_date = calib_data.calDate
qt_df = qt.getDataFrame()

qt_df_selected = qt_df[(qt_df['BID_IV']>0.0) & (qt_df['ASK_IV']>0.0) & (qt_df['IS_CALL']==1)]
x_strikes = qt_df_selected['STRIKE'].values.copy()
expiries =  qt_df_selected['EXPIRY'].values.copy()
x_prices = qt_df_selected['BID'].values.copy()
ttm = np.empty(x_strikes.shape[0])
dc = pyvacon.finance.definition.DayCounter('Act365Fixed')
for i in range(x_strikes.shape[0]):
    ttm[i] = dc.yf( cal_date, expiries[i])
    x_strike, x_price = transform_to_X_price_and_strike(cal_date, expiries[i], fwd, dsc, x_prices[i], x_strikes[i])
    x_strikes[i] = x_strike
    x_prices[i] = x_price

model = HestonForDeepHedging(rate_of_mean_reversion=0.6,long_run_average=0.2*0.2,vol_of_vol=1.1, correlation_rho=-0.7,v0=0.2*0.2)
result = calibrate(model, x_prices, ttm, x_strikes)
#model_call_prices = model.compute_call_price(1.0, x_strikes, ttm)
print(result)
print(model._to_dict())