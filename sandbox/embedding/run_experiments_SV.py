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
#tf.compat.v1.enable_eager_execution() 
#tf.keras.backend.set_floatx('float64')

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import rivapy.models
from rivapy.models.historic_sim import HistoricSimulation
from rivapy.instruments.specifications import EuropeanVanillaSpecification, BarrierOptionSpecification
from rivapy.pricing.vanillaoption_pricing import (
    VanillaOptionDeepHedgingPricer,
    DeepHedgeModelwEmbedding,
)
from scipy.special import comb
import analysis
from sys import exit

repo = analysis.Repo(
    '/home/doeltz/doeltz/development/repos/embedding_calibrated_sv'
    #"C:/Users/doeltz/development/RiVaPy/sandbox/embedding/test"
)

import ast  
with open('/home/doeltz/doeltz/development/RiVaPy/sandbox/embedding/calibrated_models.json', 'r') as f: 
    models = json.load(f) 

models = [rivapy.models.factory.create(c) for c in models] 
#       model.append(GBM(drift=0.0,volatility=model_params['GBM']['vol'][i]))
# Historic Simulation models
if False:
    with open('/home/doeltz/doeltz/development/RiVaPy/sandbox/embedding/yfinance_data/data.json', "r") as f:
        historic_data = json.load(f)
    n_models_old = len(model)
    for i in range(len(historic_data)):
        if len(historic_data[i][1])>0:
            model.append(HistoricSimulation(historic_data[i][1], description=historic_data[i][0]))
    print( "include historic data, number of historic models: ", len(historic_data)-n_models_old)



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

n_portfolios = None # set to None to switch off embedding with respect to portfolios

def objective(trial):
    # Define the hyperparameters to be tuned
    _, params = repo.run(
                                    refdate,
                                    spec,
                                    model,
                                    rerun=True,
                                    depth=trial.suggest_int("depth", 2, 4),
                                    nb_neurons=trial.suggest_categorical("nb_neurons", [32,64,128,256]),
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
        for n_sims_per_model in [4000]:#100, 200, 400, 800]:#[2000, 4000, 8000, 16000, 32000, 64000]:
            for model in models:
                n_sims = n_sims_per_model*len(models)
                print('nsims: ', n_sims)
                for emb_size in [8]:
                    for seed in [112]:
                        pricing_results, params = repo.run(
                                            refdate,
                                            spec,
                                            models,
                                            rerun=False,
                                            depth=3,
                                            nb_neurons=128,#128,
                                            n_sims=n_sims,
                                            regularization=0.,
                                            epochs=5000,
                                            verbose=1,
                                            tensorboard_logdir=repo.repo_dir+"/logs/"
                                            + dt.datetime.now().strftime("%Y%m%dT%H%M%S"),
                                            initial_lr=1e-3, 
                                            final_lr=1e-5, #1e-4,
                                            multiplier_lr=2.8,
                                            multiplier_batch_size=3,
                                            n_increase_batch_size=1,
                                            decay_steps=1000,#9000,#100,
                                            batch_size=10_000,
                                            seed=seed,
                                            days=int(np.max(days)),
                                            n_portfolios=n_portfolios,
                                            embedding_size=emb_size,
                                            embedding_size_port=None,
                                            #rerun=True,
                                            transaction_cost={}#'ADS':[1e-10]},#'DOB_ADS':[0.01]},
                                            #loss = "exponential_utility"
                                        )
                        params["pnl_result"]["var"]