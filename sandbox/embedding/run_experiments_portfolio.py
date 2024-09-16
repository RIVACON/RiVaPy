import datetime as dt
import sys

sys.path.insert(0, "../..")
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import json
import os

import logging
logging.basicConfig(level=logging.DEBUG)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set loglevel to warning
#import tensorflow as tf
#tf.config.run_functions_eagerly(True)
#tf.keras.backend.set_floatx('float64')

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from rivapy.tools.datetime_grid import DateTimeGrid
from rivapy.models.gbm import GBM
from rivapy.models.historic_sim import HistoricSimulation
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

n_models_per_model_type = 30#
gbm_vols = np.linspace(0.05,1.5,n_models_per_model_type)
for i in range(n_models_per_model_type):
    #model.append(HestonForDeepHedging(rate_of_mean_reversion = 0.6067,long_run_average = 0.0707,
    #              vol_of_vol = 0.2928, correlation_rho = -0.757,v0 = 0.0654))
    #model.append(GBM(0.,vol_list[i]))
    model.append(GBM(drift=0.0,volatility=gbm_vols[i]))
    if False:
        model.append(HestonForDeepHedging(rate_of_mean_reversion = model_params['Heston']['rate_of_mean_reversion'][i],
                                        long_run_average = model_params['Heston']['long_run_average'][i],
                                        vol_of_vol = model_params['Heston']['vol_of_vol'][i], 
                                        correlation_rho = model_params['Heston']['correlation_rho'][i],
                                        v0 = model_params['Heston']['v0'][i]))
        model.append(HestonWithJumps(rate_of_mean_reversion = model_params['Heston with Jumps']['rate_of_mean_reversion'][i],
                                    long_run_average = model_params['Heston with Jumps']['long_run_average'][i],
                                    vol_of_vol = model_params['Heston with Jumps']['vol_of_vol'][i], 
                                    correlation_rho = model_params['Heston with Jumps']['correlation_rho'][i],
                                    muj = 0.1791,sigmaj = 0.1346, 
                                    lmbda = model_params['Heston with Jumps']['lmbda'][i],
                                    v0 = model_params['Heston with Jumps']['v0'][i]))
        model.append(BNS(rho =model_params['BNS']['rho'][i],
                        lmbda=model_params['BNS']['lmbda'][i],
                        b=model_params['BNS']['b'][i],
                        a=model_params['BNS']['a'][i],
                        v0 = model_params['BNS']['v0'][i]))

# Historic Simulation models
if True:
    with open('/home/doeltz/doeltz/development/RiVaPy/sandbox/embedding/yfinance_data/data.json', "r") as f:
        historic_data = json.load(f)
    n_models_old = len(model)
    for i in range(len(historic_data)):
        if len(historic_data[i][1])>0:
            model.append(HistoricSimulation(historic_data[i][1], description=historic_data[i][0]))
    print( "include historic data, number of historic models: ", len(historic_data)-n_models_old)


repo = analysis.Repo(
    '/home/doeltz/doeltz/development/repos/embedding_historic'
    #"C:/Users/doeltz/development/RiVaPy/sandbox/embedding/test"
)

reg = {
    "mean_variance": [0.0],
    "exponential_utility": [5.0, 10.0],  # , 15.0, 20.0] ,
    "expected_shortfall": [0.1],
}

strike = [1.0]#[0.8, 0.9, 1.0, 1.1, 1.2]
days = [30]#20, 40, 60, 80, 100, 120]
refdate = dt.datetime(2023, 1, 1)
issuer = "DBK"
seclevel = "COLLATERALIZED"
tpe = "CALL"  # Change to 'PUT' if you want to calculate the price of an european put option.


spec = []

for i in range(len(strike)):
    j=0
    expiry = refdate + dt.timedelta(days=days[j])
    spec.append(EuropeanVanillaSpecification(
                'C'+str(len(spec))+str(tpe)+'K'+str(strike[i])+'T'+str(days[j]),
                tpe,
                expiry,
                strike[i],
                #barrier=0.95,
                issuer=issuer,
                sec_lvl=seclevel,
                curr="EUR",
                udl_id="ADS",
                share_ratio=1,
            ))

n_sims_per_model = 1000  
n_sims = n_sims_per_model*len(model)
n_portfolios = None # set to None to switch off embedding with respect to portfolios

def objective(trial):
    # Define the hyperparameters to be tuned
    _, params = repo.run(
                                    refdate,
                                    spec,
                                    model,
                                    rerun=True,
                                    depth=trial.suggest_int("depth", 2, 4),
                                    nb_neurons=trial.suggest_categorical("nb_neurons", [32, 64,128,256]),
                                    n_sims=n_sims,
                                    regularization=0.,
                                    epochs=trial.suggest_int("epochs", 30, 200),
                                    verbose=1,
                                    tensorboard_logdir=repo.repo_dir+"/logs/"
                                    + dt.datetime.now().strftime("%Y%m%dT%H%M%S"),
                                    initial_lr=trial.suggest_float("initial_lr", 2e-5, 1e-3),
                                    final_lr=trial.suggest_float("final_lr", 1e-6, 1e-5),
                                    multiplier_lr=trial.suggest_float("multiplier_lr", 1.1, 4.0),
                                    multiplier_batch_size=trial.suggest_int("multiplier_batch_size", 2, 4),
                                    n_increase_batch_size=trial.suggest_int("n_increase_batch_size", 1, 5),
                                    decay_steps=trial.suggest_categorical("decay_steps", [10,50,100,200]),
                                    batch_size=trial.suggest_categorical("batch_size", [16,32,64,128,256]),#2024,
                                    seed=42,
                                    days=int(np.max(days)),
                                    n_portfolios=n_portfolios,
                                    embedding_size=trial.suggest_categorical("embedding_size", [1,2,4,8]),
                                    embedding_size_port=None,
                                    transaction_cost={}#'ADS':[1e-10]},#'DOB_ADS':[0.01]},
                                    #loss = "exponential_utility"
                                )
    return params["pnl_result"]["var"]
    

if __name__=='__main__':
    import optuna
    if False:
        # Add stream handler of stdout to show the messages
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study_name = "GBM-Study"  # Unique identifier of the study.
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
        study.optimize(objective, n_trials=20)
    else:
        for emb_size in [4]:
            for seed in [42]:
                pricing_results, params = repo.run(
                                    refdate,
                                    spec,
                                    model,
                                    rerun=False,
                                    depth=3,
                                    nb_neurons=128,
                                    n_sims=n_sims,
                                    regularization=0.,
                                    epochs=185,
                                    verbose=1,
                                    tensorboard_logdir=repo.repo_dir+"/logs/"
                                    + dt.datetime.now().strftime("%Y%m%dT%H%M%S"),
                                    initial_lr=0.00015, 
                                    final_lr=0.000005,
                                    multiplier_lr=2.8,
                                    multiplier_batch_size=3,
                                    n_increase_batch_size=3,
                                    decay_steps=50,
                                    batch_size=32,#2024,
                                    seed=seed,
                                    days=int(np.max(days)),
                                    n_portfolios=n_portfolios,
                                    embedding_size=emb_size,
                                    embedding_size_port=None,
                                    transaction_cost={}#'ADS':[1e-10]},#'DOB_ADS':[0.01]},
                                    #loss = "exponential_utility"
                                )
                params["pnl_result"]["var"]