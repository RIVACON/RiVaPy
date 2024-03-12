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
from rivapy.models.gbm import GBM
from rivapy.instruments.specifications import EuropeanVanillaSpecification
from rivapy.tools.datetools import DayCounter
from rivapy.tools.enums import DayCounterType
from rivapy.tools.datetime_grid import DateTimeGrid
from rivapy.pricing.deep_hedging import DeepHedgeModel


            
def _generate_lr_schedule(initial_learning_rate: float, decay_step: int, decay_rate: float,):
    return tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=initial_learning_rate,#1e-3,
            decay_steps=decay_step,
            decay_rate=decay_rate)
    


class VanillaOptionDeepHedgingPricer:
    class PricingResults:
        def __init__(self, hedge_model: DeepHedgeModel, paths: np.ndarray, sim_results, payoff):
            self.hedge_model = hedge_model
            self.paths = paths
            self.sim_results = sim_results
            self.payoff = payoff

    @staticmethod
    def _compute_timegrid():
        n = 30#365
        T = 30/365
        timegrid = np.linspace(0.0,T,n)
        return timegrid, n
    


    @staticmethod
    def compute_payoff(n_sims: int, 
                       hedge_ins: Dict[str, np.ndarray], strike: float):
        payoff = np.zeros((n_sims,))
        for k,v in hedge_ins.items(): 
            payoff -= np.maximum(v[-1,:] - strike,0)
        return payoff

    @staticmethod
    def generate_paths(vanillaoption: EuropeanVanillaSpecification,
                model: GBM, 
                n_sims: int, 
                timegrid: DateTimeGrid):
        tf.random.set_seed(seed)
        np.random.seed(seed+123)
        timegrid = VanillaOptionDeepHedgingPricer._compute_timegrid()
        model = GBM(drift = 0., volatility=0.2)
        #n_sims = 100_000
        S0 = 1.
        return model.simulate(timegrid, start_value=S0,M = n_sims, n=n)

    @staticmethod
    def price( val_date: dt.datetime,
                vanillaoption: EuropeanVanillaSpecification,
                model: GBM, 
                depth: int, 
                nb_neurons: int, 
                n_sims: int, 
                regularization: float, 
                epochs: int,
                verbose: bool=0,
                tensorboard_logdir: str=None, 
                initial_lr: float = 1e-4, 
                batch_size: int = 100, 
                decay_rate: float=0.7, 
                decay_steps: int = 100_000,
                seed: int = 42,
                loss: str = 'mean_variance',
                transaction_cost: dict = {},
                threshold: float = 0.,
                cascading: bool = False
                #paths: Dict[str, np.ndarray] = None
                ):
        """Price a vanilla option using deeep hedging

        Args:
            val_date (dt.datetime): Valuation date.
            green_ppa (EuropeanVanillaSpecification): Specification of a vanilla option.
            power_wind_model (OrnsteinUhlenbeck): The model modeling blackbla
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
            loss (str, optional): Either 'mean_variance' or 'exponential_utility'.
            transaction_cost (dict, optional): Proportional transaction cost dependent on instrument. Default is empty dict.
            threshold(float,optional): Threshold for trading restrictions. Defaults to 0.
            cascading(bool,optiona): Flag if cascading is considered (in timegrid) or not. Defaults to False.
        Returns:
            _type_: _description_

        .. seealso::
        
           DeepHedgeModel
                The general deep hedging model used internally in this pricing method.

        """
        #print(locals())
        #if paths is None and power_wind_model is None:
        if model is None:
            raise Exception('A model must be specified.')
        tf.keras.backend.set_floatx('float32')

        tf.random.set_seed(seed)
        np.random.seed(seed+123)
        timegrid, n = VanillaOptionDeepHedgingPricer._compute_timegrid()
        model = GBM(drift = 0., volatility=0.2)
        #n_sims = 100_000
        S0 = 1.
        simulation_results = model.simulate(timegrid, start_value=S0,M = n_sims, n=n)
        hedge_ins = {}
        key = 'vanillaoption'
        hedge_ins[key] = simulation_results
        additional_states_ = {}
        
        hedge_model = DeepHedgeModel(list(hedge_ins.keys()), list(additional_states_.keys()), timegrid=timegrid, 
                                        regularization=regularization,depth=depth, n_neurons=nb_neurons, loss = loss,
                                          transaction_cost = transaction_cost,threshold = threshold, cascading = cascading)
        paths = {}
        paths.update(hedge_ins)
        #paths.update(additional_states_)
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=initial_lr,#1e-3,
                decay_steps=decay_steps,
                decay_rate=decay_rate, 
                staircase=True)

        payoff = VanillaOptionDeepHedgingPricer.compute_payoff(n_sims, hedge_ins, vanillaoption.strike)  
        
        hedge_model.train(paths, payoff,lr_schedule, epochs=epochs, batch_size=batch_size, tensorboard_log=tensorboard_logdir, verbose=verbose)
        return VanillaOptionDeepHedgingPricer.PricingResults(hedge_model, paths=paths, sim_results=simulation_results, payoff=payoff)

if __name__=='__main__':
    pass