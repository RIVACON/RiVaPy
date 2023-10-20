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


#timegrid = np.linspace(0.0, days*1.0/365.0, days*24)
#forward_expiries = [timegrid[-1]]

wind_onshore = WindPowerForecastModel(region='Onshore', speed_of_mean_reversion=0.1, volatility=3.0)
wind_offshore = WindPowerForecastModel(region='Offshore', speed_of_mean_reversion=0.1, volatility=3.0)
regions = [ MultiRegionWindForecastModel.Region( 
                                    wind_onshore,
                                    capacity=1000.0,
                                    rnd_weights=[0.8,0.2]
                                ),
           MultiRegionWindForecastModel.Region( 
                                    wind_offshore,
                                    capacity=100.0,
                                    rnd_weights=[0.2,0.8]
                                )
           
          ]
wind = MultiRegionWindForecastModel('Wind_Germany', regions)

model = LinearDemandForwardModel(wind_power_forecast=wind, 
                                 x_volatility = 0.8, 
                                 x_mean_reversion_speed = 0.5,
                                 power_name= 'Power_Germany',
                                additive_correction=False)

val_date = dt.datetime(2023,1,1)
strike = 1.0 #0.22
days = 3
spec = GreenPPASpecification(udl='Power_Germany',
                            technology = 'Wind',
                             location = 'Onshore',
                             schedule = [val_date + dt.timedelta(days=days)], 
                             fixed_price=strike,
                             max_capacity = 1.0, 
                             id='dummy')
repo = analysis.Repo('./experiments/')

# einmal mit forecast 0.2, 0.5
# delta-vergleich auf pfaden
# 3 dtm

reg ={'mean_variance':[0.0, 0.2, 0.5], 'exponential_utility':[10.0, 50.0] } 
for loss in ['mean_variance']:        #'exponential_utility', mean_variance
    for regularization in reg[loss]:#  [0.0, 0.1, 0.2, 0.5]:
        for seed in [42]:
            for power_fwd_price in [1.0, 0.8, 1.2]: 
                pricing_results = repo.run(val_date, 
                                        spec, model,
                                        rerun=False, 
                                        initial_forecasts={'Onshore': [0.2],
                                                        'Offshore': [0.6]},
                                        power_fwd_prices=[power_fwd_price],
                                        forecast_hours=[10,14,18],#[8, 10, 12, 14, 16, 18, 20],
                                        additional_states=[],#['Offshore'],
                                        depth=3, 
                                        nb_neurons=32, 
                                        n_sims=200_000, 
                                        regularization=regularization,
                                        epochs=200, 
                                        verbose=1,
                                        tensorboard_logdir = 'logs/' + dt.datetime.now().strftime("%Y%m%dT%H%M%S"), 
                                        initial_lr=0.5e-4,
                                        decay_steps=4_000,
                                        batch_size=400, 
                                        decay_rate=0.2, 
                                        seed=seed, 
                                        loss=loss#'exponential_utility' #'mean_variance'
                )