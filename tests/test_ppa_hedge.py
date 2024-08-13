import sys
import unittest
import rivapy.instruments.specifications as specs
tf_installed = False
try:
    import tensorflow as tf # exit this test if no tensorflow is installed  
    tf_installed = True
except:
    tf_installed = False

import datetime as dt
import numpy as np
from rivapy.models.residual_demand_fwd_model import MultiRegionWindForecastModel, WindPowerForecastModel, OrnsteinUhlenbeck, ResidualDemandForwardModel
from rivapy.models.residual_demand_model import SmoothstepSupplyCurve
from rivapy.instruments.ppa_specification import GreenPPASpecification
from rivapy.pricing.green_ppa_pricing import GreenPPADeepHedgingPricer, DeepHedgeModel


class SpecificationDeepHedgingInterfaceTest(unittest.TestCase):
    def test_EuropeanVanillaSpecification(self):
        spec = specs.EuropeanVanillaSpecification('TEST_INS', 'CALL', dt.datetime(2020,1,1), 100.0)
        path = np.zeros((10,3)) # 3 paths at 10 timesteps
        # Test if the interface compute_payoff is implemented correctly
        path[:,0] = 100.0 # first path is constant at the strike
        path[:,1] = 110.0 # second path is above the strike
        path[:,2] = 90.0 # third path is below the strike
        payoff,states = spec.compute_payoff(path,path.shape[0]-1)
        self.assertIsNone(states)
        self.assertAlmostEqual(payoff[0], 0.0, places=8)
        self.assertAlmostEqual(payoff[1], 10.0, places=8)
        self.assertAlmostEqual(payoff[2], 0.0, places=8)

    def test_BarrierOptionSpecification(self):
        spec = specs.BarrierOptionSpecification('TEST_INS', 'UIB_CALL', dt.datetime(2020,1,1), 100.0, 110.0)
        path = np.zeros((10,3))
        # Test if the interface compute_payoff is implemented correctly
        # we consider three cases: Barrier not hit, Barrier hit spot stays above barrier, barrier hit and spot drops below barrier
        path[:,0] = 105.0
        path[:,1] = 115.0
        path[:,2] = 115.0
        path[0,2] = 90.0
        path[-5:,2] = 105.0
        payoff,states = spec.compute_payoff(path,path.shape[0]-1)
        self.assertIsNotNone(states)
        self.assertAlmostEqual(payoff[0], 0.0, places=8)
        self.assertAlmostEqual(payoff[1], 15.0, places=8)
        self.assertAlmostEqual(payoff[2], 5.0, places=8)
        self.assertEqual(states[0,0], 0)
        self.assertEqual(states[-1,0], 0)
        self.assertEqual(states[0,1], 1)
        self.assertEqual(states[-1,1], 1)
        self.assertEqual(states[0,2], 0)
        self.assertEqual(states[1,2], 1)
        self.assertEqual(states[-1,2], 1)
        
if tf_installed:
    class DeepHedger(unittest.TestCase):
        def test_simple(self):
            spots = {}
            np.random.seed(42)
            n_sims = 100
            timegrid = np.linspace(0.0,1.0,12, endpoint=True)
            payoff = None
            for i in range(2):
                model = OrnsteinUhlenbeck(1.0,0.2,1.0)
                rnd = np.random.normal(size=(timegrid.shape[0], n_sims))
                s =  model.simulate(timegrid, start_value=1.0, rnd=rnd)
                spots['Asset'+str(i)] = s
                if payoff is None:
                    payoff = np.maximum(s[-1,:]-1.0,0.0)
                else:
                    payoff = np.maximum(np.maximum(s[-1,:]-1.0,0.0), payoff)
            model = DeepHedgeModel([k for k in spots.keys()], None, timegrid=timegrid,
                                    regularization=0.0, depth=3, n_neurons=32)
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.1,#1e-3,
                decay_steps=100,
                decay_rate=0.9)
            model.train(spots, payoff, lr_schedule, 5, 10)

    class GreenPPAHedger(unittest.TestCase):
        def test_hedging(self):
            """Simple test with a perfect forecast 
            """
            val_date = dt.datetime(2023,1,1)
            days = 2
            timegrid = np.linspace(0.0, days*1.0/365.0, days*24)
            #forecast_points = [i for i in range(len(timegrid)) if i%8==0]
            forward_expiries = [timegrid[-1]]
            n_sims = 1_000

            regions = [ MultiRegionWindForecastModel.Region( 
                                                WindPowerForecastModel(speed_of_mean_reversion=0.5, 
                                                                    volatility=0.05, 
                                                                        expiries=forward_expiries,
                                                                        forecasts = [0.8],#*len(forward_expiries)
                                                                        region = 'Onshore'
                                                                        ),
                                                capacity=100.0,
                                                rnd_weights=[1.0,0.0]
                                            ),
                    MultiRegionWindForecastModel.Region( 
                                                WindPowerForecastModel(speed_of_mean_reversion=0.5, 
                                                                    volatility=1.80, 
                                                                        expiries=forward_expiries,
                                                                        forecasts = [0.8],#*len(forward_expiries)
                                                                        region = 'Offshore'
                                                                        ),
                                                capacity=100.0,
                                                rnd_weights=[1.0,0.0]
                                            )
                    
                    ]
            multi_region_wind_foecast_model = MultiRegionWindForecastModel(regions)
            
            highest_price = OrnsteinUhlenbeck(1.0, 0.01, mean_reversion_level=1.0)
            supply_curve = SmoothstepSupplyCurve(1.0, 0)
            rdm = ResidualDemandForwardModel(
                                            #wind_forecast_model, 
                                            multi_region_wind_foecast_model,
                                            highest_price,
                                            supply_curve,
                                            max_price = 1.0,
                                            forecast_hours=[6, 10, 14, 18], 
                                            )

            strike = 0.4#fwd_prices[:,-1].mean()
            spec = GreenPPASpecification(technology = 'Wind',
                                        location = 'Onshore',
                                        udl = 'Power',
                                        schedule = [val_date + dt.timedelta(days=2)], 
                                        fixed_price=strike,
                                        max_capacity = 1.0)
            result = GreenPPADeepHedgingPricer.price(val_date, spec, rdm , 
                depth=3, nb_neurons=64, 
                n_sims = n_sims, regularization= 10.0, 
                epochs = 10, 
                verbose=1, 
                initial_lr = 1e-2,
                batch_size=400,
                decay_rate=0.6,
                tensorboard_logdir=None
                )
            t = 0
            #delta = result.hedge_model.model.predict([fwd_prices[:,t], forecasts[:,t], np.array([timegrid[t]]*forecasts.shape[0])]).reshape((-1))
            delta = result.hedge_model.compute_delta(result.fwd_prices, result.forecasts, t)
            #self.assertAlmostEqual(delta[0], result.forecasts[spec.location][0,0], 1e-3)

    

if __name__ == '__main__':
    unittest.main()