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
import random
sys.path.append('C:/Users/doeltz/development/RiVaPy/')
import datetime as dt
from rivapy.tools.datetime_grid import DateTimeGrid
from rivapy.pricing.deep_hedging_with_embedding import DeepHedgeModelwEmbedding
             

class VanillaOptionDeepHedgingPricer:
    class PricingResults:
        def __init__(self, hedge_model: DeepHedgeModelwEmbedding, paths: np.ndarray, sim_results, payoff):
            self.hedge_model = hedge_model
            self.paths = paths
            self.sim_results = sim_results
            self.payoff = payoff

    @staticmethod
    def _compute_timegrid(days, freq):
        T = days/365
        if freq == '12H':
            timegrid = np.linspace(0.0,T,days*2)
        else:
            timegrid = np.linspace(0.0,T,days)
        return timegrid
    

    @staticmethod
    def compute_payoff(n_sims: int, 
                       hedge_ins: Dict[str, np.ndarray], ins_list: list):
        payoff = np.zeros((n_sims,))
        for i in range(len(ins_list)):
            strike = ins_list[i].strike
            long_short_flag = ins_list[i].long_short_flag
            tpe = ins_list[i].type
            if tpe == 'CALL':
                if long_short_flag == 'short':
                    for k,v in hedge_ins.items(): 
                        payoff -= np.minimum(strike - v[-1,:],0)
                else:
                    for k,v in hedge_ins.items(): 
                        payoff -= np.maximum(v[-1,:] - strike,0)
            if tpe == 'PUT':
                if long_short_flag == 'short':
                    for k,v in hedge_ins.items(): 
                        payoff -= np.minimum(v[-1,:] - strike,0)
                else:
                    for k,v in hedge_ins.items(): 
                        payoff -= np.maximum(strike - v[-1,:],0)
            if tpe == 'UIB_CALL':
                for k,v in hedge_ins.items(): 
                    condition =  v[-1,:] > ins_list[i].barrier
                    payoff -= np.maximum(v - strike,0)[-1,:]*condition
            if tpe == 'UOB_CALL':
                for k,v in hedge_ins.items(): 
                    condition =  v[-1,:] <= ins_list[i].barrier
                    payoff -= np.maximum(v - strike,0)[-1,:]*condition
            if tpe == 'DIB_CALL':
                for k,v in hedge_ins.items(): 
                    condition =  v[-1,:] < ins_list[i].barrier
                    payoff -= np.maximum(v - strike,0)[-1,:]*condition
            if tpe == 'DOB_CALL':
                for k,v in hedge_ins.items(): 
                    condition =  v[-1,:] >= ins_list[i].barrier
                    payoff -= np.maximum(v - strike,0)[-1,:]*condition



        return payoff

    @staticmethod
    def generate_paths(seed: int,
                       model_list: list,
                       timegrid: DateTimeGrid,
                n_sims: int, 
                days: int,
                freq: str):
        tf.random.set_seed(seed)
        np.random.seed(seed+123)

        simulation_results = np.zeros((len(timegrid)+1, n_sims))
        S0 = 1. #ATM option
        emb_vec = np.zeros((n_sims))
        if freq == '12H':
            n = days*2
        else:
            n = days
        n_sims = int(n_sims/len(model_list))
        for i in range(len(model_list)):
            model= model_list[i]
            simulation_results[:,i*n_sims:n_sims*(i+1)] = model.simulate(timegrid, S0=S0, v0=model.v0, M=n_sims,n=n, model_name=model_list[i].modelname)
            emb_vec[i*n_sims:n_sims*(i+1)] = i    
        return simulation_results, emb_vec
    
    @staticmethod
    def get_call_prices(sim_results: np.ndarray, strike: float,
               seed: int,
                model_list: list,
                timegrid: DateTimeGrid,
                n_sims: int, 
                days: int,
                freq: str):
        tf.random.set_seed(seed)
        np.random.seed(seed+123)

        call_prices = np.zeros((len(timegrid)+1, n_sims))

        if freq == '12H':
            n = days*2
        else:
            n = days
        n_sims = int(n_sims/len(model_list))

        for i in range(len(model_list)):
            model= model_list[i]
            for j in range(len(timegrid)):
                ttm = (timegrid[-1] - timegrid[j])
                call_prices[j,i*n_sims:n_sims*(i+1)] = model.compute_call_price(sim_results[j,i*n_sims:n_sims*(i+1)],strike,ttm)
            call_prices[-1,i*n_sims:n_sims*(i+1)] = 0.
        return call_prices
    
    @staticmethod
    def price(val_date: dt.datetime,
        ins_list: list,#EuropeanVanillaSpecification,
                model_list: list, #HestonForDeepHedging,
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
                cascading: bool = False,
                days: int = 30,
                test_weighted_paths: bool = False,
                freq: str = 'D',
                embedding_size: int = 1
                #paths: Dict[str, np.ndarray] = None
                ):
        """Price a vanilla option using deeep hedging

        Args:
            val_date (dt.dtetime): Valuation date
            ins_list (EuropeanVanillaSpecification): Specification of a list of vanilla options
            model (GBM, HestonForDeepHedging): The list of models.
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
            days (int): number of days until expiry, used for time grid. Defaults to 30.
            test_weighted_paths (bool): Option to Test "weighted" paths S(t), currently not possible. Defaults to False.
            parameter_uncertainty (bool): Option to test for parameter uncertainty in models (vol).Defaults to False.
            freq (str): Delta t for timegrid ('D' or '12H'). Defaults to 'D' (i.e., dt = 1day)
        Returns:
            _type_: _description_

        .. seealso::
        
           DeepHedgeModel
                The general deep hedging model used internally in this pricing method.

        """
        if model_list is None:
            raise Exception('A model must be specified.')
        tf.keras.backend.set_floatx('float32')

        tf.random.set_seed(seed)
        np.random.seed(seed+123)
        timegrid = VanillaOptionDeepHedgingPricer._compute_timegrid(days,freq)

        simulation_results,emb_vec = VanillaOptionDeepHedgingPricer.generate_paths(seed,model_list,timegrid,n_sims,days,freq)
        if model_list[0].modelname == 'Heston with Volswap':
            raise Exception('Heston with Volswap currently not running')
            hedge_ins = {}
            hedge_ins['S1'] = simulation_results[:,:,0]
            hedge_ins['S2'] = simulation_results[:,:,1]
        else:
            hedge_ins = {}
            additional_states_ = {}
            additional_states_["emb_key"] = emb_vec
            for i in range(len(ins_list)):
                key = ins_list[i].udl_id
                T = (ins_list[i].expiry - val_date).days
                print(T)
                if freq == '12H':
                    hedge_ins[key] = simulation_results[:int(T*2),:]
                    #hedge_ins['V'] = VanillaOptionDeepHedgingPricer.get_call_prices(simulation_results[:int(T*2),:], ins_list[i].strike, seed,model_list,timegrid[:int(T*2)],n_sims,days,freq)
                else:
                    hedge_ins[key] = simulation_results[:int(T),:]
                    #hedge_ins['V'] = VanillaOptionDeepHedgingPricer.get_call_prices(simulation_results[:int(T),:], ins_list[i].strike, seed,model_list,timegrid[:int(T)],n_sims,days,freq)
                
        hedge_model = DeepHedgeModelwEmbedding(list(hedge_ins.keys()), list(additional_states_.keys()),timegrid=timegrid, 
                                        regularization=regularization,depth=depth, n_neurons=nb_neurons, loss = loss,
                                        transaction_cost = transaction_cost,no_of_unique_model=len(model_list),embedding_size=embedding_size)

        paths = {}
        paths.update(hedge_ins)
        paths.update(additional_states_) 
        print(paths)

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=initial_lr,#1e-3,
                decay_steps=decay_steps,
                decay_rate=decay_rate, 
                staircase=True)
    
        payoff = VanillaOptionDeepHedgingPricer.compute_payoff(n_sims, hedge_ins, ins_list) 

        hedge_model.train(paths, payoff, lr_schedule, epochs=epochs, batch_size=batch_size, tensorboard_log=tensorboard_logdir, verbose=verbose)
        return VanillaOptionDeepHedgingPricer.PricingResults(hedge_model, paths=paths, sim_results=simulation_results, payoff=payoff)

if __name__=='__main__':
    pass