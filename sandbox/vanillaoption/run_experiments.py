import datetime as dt
import sys
sys.path.insert(0,'../..')
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from rivapy.tools.datetime_grid import DateTimeGrid

from rivapy.models.residual_demand_fwd_model import WindPowerForecastModel, MultiRegionWindForecastModel, LinearDemandForwardModel
from rivapy.instruments.ppa_specification import GreenPPASpecification
from rivapy.models.residual_demand_model import SmoothstepSupplyCurve
from rivapy.models import OrnsteinUhlenbeck
from rivapy.pricing.green_ppa_pricing import GreenPPADeepHedgingPricer, DeepHedgeModel
import numpy as np
from scipy.special import comb

import analysis

from sys import exit

#np.random.seed(42)
#timegrid = np.linspace(0.0,1.0,365) # simulate on daily timegrid over 1 yr horizon
model = OrnsteinUhlenbeck(speed_of_mean_reversion = 5.0, volatility=0.1)

#val_date = dt.datetime(2023,1,1)
#strike = 1.0 #0.22
#transaction_cost = 0.01
#days = 2

refdate = dt.datetime(2023,1,1)
issuer = 'DBK'
seclevel = 'COLLATERALIZED'
currency = 'EUR'
tpe = 'CALL' # Change to 'PUT' if you want to calculate the price of an european put option.
expiry = refdate + dt.timedelta(days=365)
strike = 60

repo = analysis.Repo('./experiments/')


# einmal mit forecast 0.2, 0.5
# delta-vergleich auf pfaden
# 3 dtm

reg ={'mean_variance':[0.0, 0.2, 0.5], 
      'exponential_utility':[5.0, 10.0],#, 15.0, 20.0] , 
        'expected_shortfall':[0.1]} 
for max_capacity in [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 5.0, 10.0]:
    spec = GreenPPASpecification(udl='Power_Germany',
                                technology = 'Wind',
                                location = 'Onshore',
                                schedule = [val_date + dt.timedelta(days=days)], 
                                fixed_price=strike,
                                max_capacity = max_capacity, 
                                id='dummy')
    for loss in ['exponential_utility']:        #'exponential_utility', mean_variance
        for regularization in reg[loss]:#  [0.0, 0.1, 0.2, 0.5]:
            for seed in [42]:
                for power_fwd_price in [1.0]:#, 0.8, 1.2]: 
                    pricing_results = repo.run(val_date, 
                                            spec, model,
                                            rerun=False, 
                                            initial_forecasts={'Onshore': [0.8],
                                                            'Offshore': [0.6]},
                                            power_fwd_prices=[power_fwd_price],
                                            forecast_hours=[10,14,18],#[8, 10, 12, 14, 16, 18, 20],
                                            additional_states=[],#['Offshore'],
                                            depth=3, 
                                            nb_neurons=32, 
                                            n_sims=200_000, 
                                            regularization=regularization,
                                            epochs=10,#400, #FS 
                                            verbose=1,
                                            tensorboard_logdir = 'logs/' + dt.datetime.now().strftime("%Y%m%dT%H%M%S"), 
                                            initial_lr=1e-5,
                                            decay_steps=4_000,
                                            batch_size=2000, 
                                            decay_rate=0.2, 
                                            seed=seed, 
                                            loss=loss
                    )