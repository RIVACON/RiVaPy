from sys import exit
import datetime as dt
import sys
sys.path.insert(0,'../..')
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from scipy.special import comb
import pandas as pd
#import seaborn
#seaborn.set_style('whitegrid')
#seaborn.reset_orig() #uncomment to get seaborn styles
from rivapy.tools.datetime_grid import DateTimeGrid
from rivapy.models.residual_demand_fwd_model import WindPowerForecastModel, MultiRegionWindForecastModel, LinearDemandForwardModel
from rivapy.instruments.ppa_specification import GreenPPASpecification
from rivapy.models.residual_demand_model import SmoothstepSupplyCurve
from rivapy.models import OrnsteinUhlenbeck
from rivapy.pricing.green_ppa_pricing import GreenPPADeepHedgingPricer, DeepHedgeModel

import analysis
import pprint

repo = None

def set_repo(path:str):
    global repo
    repo = analysis.Repo(path)


def setup(experiments):
    sim_results = {}
    forecast_points={}
    hedge_models={}
    specs = {}
    paths = {}
    pnl = {}
    deltas = {}
    for k in experiments["models"]:
        sim_results[k[1]], forecast_points[k[1]] = repo.simulate_model(k[0], n_sims=100_000)
        hedge_models[k[1]] = repo.get_hedge_model(k[0])
        specs[k[1]] = GreenPPASpecification.from_dict(repo.results[k[0]]['ppa_spec'])
        paths[k[1]] = {l:sim_results[k[1]].get(l, forecast_points[k[1]]) for l in ['Power_Germany_FWD0', 'Onshore_FWD0', 'Offshore_FWD0']}
        pnl['volume_hedge, ' + k[1]] = compute_volume_hedge_pnl(paths[k[1]], specs[k[1]])
            #'static_volume_hedge': compute_static_volume_hedge_pnl(paths[ref_key_volume_hedge], specs[ref_key_volume_hedge]),
        pnl['no_hedge, ' + k[1]] = compute_no_hedge_pnl(paths[k[1]], specs[k[1]])
        #    'static_volume_hedge':compute_static_volume_hedge_pnl(paths[ref_key_volume_hedge], specs[ref_key_volume_hedge])
    for k,v in hedge_models.items():
        pnl[k] = compute_pnl(v, paths[k], specs[k])#specs['ec202973a34dfc5b71a86e0e7b2209a62c29b6a1'])#
        deltas[k] = hedge_models[k].compute_delta(paths[k])
    
    return sim_results, forecast_points, hedge_models, specs, paths, pnl, deltas

def compute_pnl(hedge_model, paths, green_ppa_spec):
    power_price = paths.get('Power_Germany_FWD0', None)
    volume = paths.get(green_ppa_spec.location+'_FWD0')
    print(green_ppa_spec.fixed_price)
    payoff = (power_price[-1,:] -green_ppa_spec.fixed_price)*(volume[-1,:])*green_ppa_spec.max_capacity
    return hedge_model.compute_pnl(paths, payoff)

def compute_volume_hedge_pnl(paths, green_ppa_spec):
    power_price = paths.get('Power_Germany_FWD0', None)
    volume = paths.get(green_ppa_spec.location+'_FWD0')
    pnl = volume[0,:]*power_price[0,:]
    for i in range(1,power_price.shape[0]-1):
        pnl += (volume[i,:]-volume[i-1,:])*power_price[i,:]
    pnl -= volume[-2,:]*power_price[-1,:]
    pnl += (power_price[-1,:] -green_ppa_spec.fixed_price)*(volume[-1,:])*green_ppa_spec.max_capacity
    return pnl

def compute_no_hedge_pnl(paths, green_ppa_spec):
    power_price = paths.get('Power_Germany_FWD0', None)
    volume = paths.get(green_ppa_spec.location+'_FWD0')
    pnl = (power_price[-1,:] -green_ppa_spec.fixed_price)*(volume[-1,:])*green_ppa_spec.max_capacity
    return pnl


def compute_static_volume_hedge_pnl(paths, green_ppa_spec):
    power_price = paths.get('Power_Germany_FWD0', None)
    volume = paths.get(green_ppa_spec.location+'_FWD0')
    pnl = (power_price[-1,:] -green_ppa_spec.fixed_price)*(volume[-1,:]) + (-power_price[-1,:] +power_price[0,:])*(volume[0,:])
    return pnl

def compute_static_volume_hedge_var_pnl(paths, hedge_volume, green_ppa_spec, strike=None):
    power_price = paths.get('Power_Germany_FWD0', None)
    volume = paths.get(green_ppa_spec.location+'_FWD0')
    strike_ = green_ppa_spec.fixed_price
    if strike is not None:
        strike_ = strike
    pnl = (power_price[-1,:] -strike_)*(volume[-1,:]) + (-power_price[-1,:] +power_price[0,:])*(hedge_volume)
    tmp = np.percentile(pnl, 5)
    return pnl[pnl<tmp].mean()#pnl.var()


def compute_statistics(pnl):
    pnl_stat = {'name':[], 'mean': [], 'var':[],'p-skewness':[], 
                #'20%': [], 
                #'5%':[], 
                #'10%':[], 
                '5% ES':[], 
                #'10% ES':[], 
                '20% ES':[], 
               'utility, 0.05': [], 
                'utility, 0.1': [],
                #'utility, 0.15': [], 'utility, 0.2': [], 
               }
    for k,v in pnl.items():
        if ( k == 'static_volume_hedge'):# or (k==): # (k == 'no_hedge') or
            continue
        v_ = 100.0*v
        pnl_stat['name'].append(k)
        pnl_stat['mean'].append(np.mean(v_))
        pnl_stat['var'].append(np.sqrt(np.var(v_)))
        pnl_stat['p-skewness'].append( 3.0*(v_.mean()-np.median(v_))/v_.std())

        tmp = np.percentile(v_, 10)
        #pnl_stat['10%'].append(tmp)
        tmp = v_[v_<tmp].mean()
        #pnl_stat['10% ES'].append(tmp) 

        tmp = np.percentile(v_, 20)
        #pnl_stat['20%'].append(tmp)
        tmp = v_[v_<tmp].mean()
        pnl_stat['20% ES'].append(tmp)
        tmp = np.percentile(v_, 5)
        #pnl_stat['5%'].append(tmp)
        tmp = v_[v_<tmp].mean()
        pnl_stat['5% ES'].append(tmp)
        
        pnl_stat['utility, 0.05'].append(np.mean(np.exp(-0.05*v_)))
        pnl_stat['utility, 0.1'].append(np.mean(np.exp(-0.1*v_)))
        #pnl_stat['utility, 0.15'].append(np.mean(np.exp(-0.15*v_)))
        #pnl_stat['utility, 0.2'].append( np.mean(np.exp(-0.2*v_)))

    pnl_stat = pd.DataFrame(pnl_stat)
    return pnl_stat.set_index('name')#.to_latex(float_format="{:0.3f}".format)

