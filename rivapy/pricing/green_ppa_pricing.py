from typing import List, Dict


try:
    import tensorflow as tf
    try:
        tf.config.run_functions_eagerly(False)
    except:
        pass
except:
    import warnings
    warnings.warn('Tensorflow is not installed. You cannot use the PPA Deep Hedging Pricer!')
    
import numpy as np
import sys
sys.path.append('C:/Users/doeltz/development/RiVaPy/')
import datetime as dt
from rivapy.models.base_model import BaseFwdModel
from rivapy.models import ResidualDemandForwardModel
from rivapy.instruments.ppa_specification import GreenPPASpecification
from rivapy.tools.datetools import DayCounter
from rivapy.tools.enums import DayCounterType
from rivapy.tools.datetime_grid import DateTimeGrid
from rivapy.pricing.deep_hedging import DeepHedgeModel


            
def _generate_lr_schedule(initial_learning_rate: float, decay_step: int, decay_rate: float,):
    return tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=initial_learning_rate,#1e-3,
            decay_steps=decay_step,
            decay_rate=decay_rate)
    


class GreenPPADeepHedgingPricer:
    class PricingResults:
        def __init__(self, hedge_model: DeepHedgeModel, paths: np.ndarray, sim_results, payoff):
            self.hedge_model = hedge_model
            self.paths = paths
            self.sim_results = sim_results
            self.payoff = payoff

    @staticmethod
    def _compute_points(val_date: dt.datetime, 
                        green_ppa: GreenPPASpecification, 
                        forecast_hours: List[int]):
        ppa_schedule = green_ppa.get_schedule()
        if ppa_schedule[-1] <= val_date:
            return None
        timegrid = DateTimeGrid(start=val_date, end=ppa_schedule[-1], freq='1H', inclusive=None, daycounter=DayCounterType.Act365Fixed)
        dc = DayCounter(DayCounterType.Act365Fixed)
        fwd_expiries = [dc.yf(val_date, d) for d in ppa_schedule if d>val_date]
        forecast_points = [i for i in range(len(timegrid.dates)) if timegrid.dates[i].hour in forecast_hours]
        return timegrid, fwd_expiries, forecast_points


    @staticmethod
    def compute_payoff(n_sims: int, 
                       hedge_ins: Dict[str, np.ndarray], 
                       additional_states: Dict[str, np.ndarray], 
                       green_ppa: GreenPPASpecification):
        payoff = np.zeros((n_sims,))
        for k,v in hedge_ins.items(): #TODO: We assume that each hedge instruments corresponds to the spot price at the last time step. Make this more explicit!
            expiry = k.split('_')[-1]
            forecast_key = green_ppa.location+'_'+expiry
            payoff += (v[-1,:] -green_ppa.fixed_price)*(additional_states[forecast_key][-1,:])
        return payoff

    @staticmethod
    def generate_paths(green_ppa: GreenPPASpecification,
                power_wind_model: ResidualDemandForwardModel, 
                 initial_forecasts: dict,
                power_fwd_prices: np.ndarray,
                n_sims: int, 
                timegrid: DateTimeGrid=None,
                val_date: dt.datetime=None):
        if timegrid is None:
             timegrid, expiries, _ = GreenPPADeepHedgingPricer._compute_points(val_date, green_ppa, [0])
        rnd = np.random.normal(size=power_wind_model.rnd_shape(n_sims, timegrid.timegrid.shape[0]))
        return power_wind_model.simulate(timegrid.timegrid, rnd, expiries=expiries, 
                                                       initial_forecasts=initial_forecasts,
                                                       power_fwd_prices=power_fwd_prices)

    @staticmethod
    def price( val_date: dt.datetime,
                green_ppa: GreenPPASpecification,
                power_wind_model: ResidualDemandForwardModel, 
                initial_forecasts: dict,
                power_fwd_prices: np.ndarray,
                forecast_hours: List[int],
                depth: int, 
                nb_neurons: int, 
                n_sims: int, 
                regularization: float, 
                epochs: int,
                timegrid: DateTimeGrid=None,
                verbose: bool=0,
                tensorboard_logdir: str=None, 
                initial_lr: float = 1e-4, 
                batch_size: int = 100, 
                decay_rate: float=0.7, 
                decay_steps: int = 100_000,
                seed: int = 42,
                additional_states=None, 
                #paths: Dict[str, np.ndarray] = None
                ):
        """Price a green PPA using deeep hedging

        Args:
            val_date (dt.datetime): Valuation date.
            green_ppa (GreenPPASpecification): Specification of a green PPA.
            power_wind_model (ResidualDemandForwardModel): The model modeling power prices and renewable quantities.
            depth (int): Number of layers of neural network.
            nb_neurons (int): Number of activation functions. 
            n_sims (int): Number of paths used as input for network training.
            regularization (float): The regularization term entering the loss: Loss is defined by -E[pnl] + regularization*Var(pnl)
            timegrid (DateTimeGrid, optional): Timegrid used for simulation and hedging. If None, an hourly timegrid is used. Defaults to None.
            epochs (int): Number of epochs for network training.
            verbose (bool, optional): Verbosity level (0, 1 or 2). Defaults to 0.
            tensorboard_logdir (str, optional): Pah to tensorboard log, if None, no log is written. Defaults to None.
            initial_lr (float, optional): Initial learning rate. Defaults to 1e-4.
            batch_size (int, optional): The batch size. Defaults to 100.
            decay_rate (float, optional): Decay of learning rate after each epoch. Defaults to 0.7.
            seed (int, optional): Seed that is set to make results reproducible. Defaults to 42.

        Returns:
            _type_: _description_
        """
        #print(locals())
        #if paths is None and power_wind_model is None:
        if power_wind_model is None:
            raise Exception('A power wind model must be specified.')
        tf.keras.backend.set_floatx('float32')

        #_validate(val_date, green_ppa,power_wind_model)
        if green_ppa.udl not in power_wind_model.udls():
            raise Exception('Underlying ' + green_ppa.udl + ' not in underlyings of the model ' + str(power_wind_model.udls()))
        tf.random.set_seed(seed)
        np.random.seed(seed+123)
        timegrid, expiries, forecast_points = GreenPPADeepHedgingPricer._compute_points(val_date, green_ppa, forecast_hours)
        if len(expiries) == 0:
            return None
        rnd = np.random.normal(size=power_wind_model.rnd_shape(n_sims, timegrid.timegrid.shape[0]))
        simulation_results = power_wind_model.simulate(timegrid.timegrid, rnd, expiries=expiries, 
                                                       initial_forecasts=initial_forecasts,
                                                       power_fwd_prices=power_fwd_prices)
        
        hedge_ins = {}
        for i in range(len(expiries)):
            key = BaseFwdModel.get_key(green_ppa.udl, i)
            hedge_ins[key] = simulation_results.get(key, forecast_points)
        additional_states_ = {}
        for i in range(len(expiries)):
            key = BaseFwdModel.get_key(green_ppa.location, i)
            additional_states_[key] = simulation_results.get(key, forecast_points)
        if additional_states is not None:
            for a in additional_states:
                for i in range(len(expiries)):
                    key = BaseFwdModel.get_key(a, i)
                    additional_states_[key] = simulation_results.get(key, forecast_points)
        
        hedge_model = DeepHedgeModel(list(hedge_ins.keys()), list(additional_states_.keys()), timegrid.timegrid, 
                                        regularization=regularization,depth=depth, n_neurons=nb_neurons)
        paths = {}
        paths.update(hedge_ins)
        paths.update(additional_states_)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_lr,#1e-3,
                decay_steps=decay_steps,
                decay_rate=decay_rate, 
                staircase=True)

        payoff = GreenPPADeepHedgingPricer.compute_payoff(n_sims, hedge_ins, additional_states_, green_ppa)  
        
        hedge_model.train(paths, payoff,lr_schedule, epochs=epochs, batch_size=batch_size, tensorboard_log=tensorboard_logdir, verbose=verbose)
        return GreenPPADeepHedgingPricer.PricingResults(hedge_model, paths=paths, sim_results=simulation_results, payoff=payoff)

if __name__=='__main__':
    pass