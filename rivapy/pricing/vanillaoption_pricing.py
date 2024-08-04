from typing import List, Dict
import copy

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
                       hedge_ins: Dict[str, np.ndarray], portfolio_list: list, port_vec,days, val_date):
        payoff = np.zeros((n_sims,))

        for i in range(n_sims):
            for j in range(len(portfolio_list)): 
                for k,v in hedge_ins.items():
                    if portfolio_list[j].portfolioid == port_vec[i]:
                        T = (portfolio_list[j].expiry - val_date).days
                        strike = portfolio_list[j].strike
                        long_short_flag = portfolio_list[j].long_short_flag
                        tpe = portfolio_list[j].type
                        if tpe == 'CALL':
                            if long_short_flag == 'short':
                                payoff[i] -= np.minimum(strike - v[T-1,i],0)
                                continue
                            else:
                                payoff[i] -= np.maximum(v[T-1,i] - strike,0)
                                continue
                        if tpe == 'PUT':
                            if long_short_flag == 'short':
                                payoff[i] -= np.minimum(v[T-1,i] - strike,0)
                                continue
                            else:
                                payoff[i] -= np.maximum(strike - v[T-1,i],0)
                                continue

                        if tpe == 'UIB_CALL':
                            condition =  np.max(v[:T,i]) > portfolio_list[i].barrier
                            payoff[i] -= np.maximum(v - strike,0)[T-1,i]*condition
                            continue
                        if tpe == 'UOB_CALL':
                            condition =  np.max(v[:T,i]) <= portfolio_list[i].barrier
                            payoff[i] -= np.maximum(v - strike,0)[T-1,i]*condition
                            continue
                        if tpe == 'DIB_CALL':
                            condition =  np.min(v[:T,i]) < portfolio_list[i].barrier
                            payoff[i] -= np.maximum(v - strike,0)[T-1,i]*condition
                            continue
                        if tpe == 'DOB_CALL':
                            condition =  np.min(v[:T,i]) >= portfolio_list[i].barrier
                            payoff[i] -= np.maximum(v - strike,0)[T-1,i]*condition
                            continue

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
        ## still a constant, fixed to 1.: TODO -> set variable!!
        S0 = 1. 
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
    def price(val_date: dt.datetime,
        portfolio_list: list,#EuropeanVanillaSpecification,
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
                days: int = 30,
                freq: str = 'D',
                embedding_size: int = 1,
                embedding_size_port: int = 1
                #paths: Dict[str, np.ndarray] = None
                ):
        """Price a vanilla option using deeep hedging

        Args:
            val_date (dt.dtetime): Valuation date
            portfolio_list (PortfolioSpecification): Specification of a list of portfolios
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

        maxid_old = 0
        maxid = 0
        for i in range(len(portfolio_list)):
            if maxid_old < portfolio_list[i].portfolioid:
                maxid = portfolio_list[i].portfolioid
    
        print('compute paths:')
        simulation_results,emb_vec = VanillaOptionDeepHedgingPricer.generate_paths(seed,model_list,timegrid,n_sims,days,freq)
        port_vec = np.random.randint(maxid+1, size=len(emb_vec))

        if model_list[0].modelname == 'Heston with Volswap':
            raise Exception('Heston with Volswap currently not running')
            hedge_ins = {}
            hedge_ins['S1'] = simulation_results[:,:,0]
            hedge_ins['S2'] = simulation_results[:,:,1]
        else:
            hedge_ins = {}
            additional_states_ = {}
            additional_states_["emb_key"] = emb_vec
            additional_states_["port_key"] = port_vec
            for i in range(len(portfolio_list)):
                T = days#(portfolio_list[i].expiry - val_date).days
                key = portfolio_list[i].udl_id 
                if freq == '12H':
                    hedge_ins[key] = simulation_results[:int(T*2),:]
                    #hedge_ins['V'] = VanillaOptionDeepHedgingPricer.get_call_prices(simulation_results[:int(T*2),:], ins_list[i].strike, seed,model_list,timegrid[:int(T*2)],n_sims,days,freq)
                else:
                    hedge_ins[key] = simulation_results[:int(T),:]
                    #hedge_ins['V'] = VanillaOptionDeepHedgingPricer.get_call_prices(simulation_results[:int(T),:], ins_list[i].strike, seed,model_list,timegrid[:int(T)],n_sims,days,freq)
        hedge_model = DeepHedgeModelwEmbedding(list(hedge_ins.keys()), list(additional_states_.keys()),timegrid=timegrid, 
                                        regularization=regularization,depth=depth, n_neurons=nb_neurons, loss = loss,
                                        transaction_cost = transaction_cost,no_of_unique_model=len(model_list),embedding_size=embedding_size,
                                        no_of_portfolios=maxid,embedding_size_port=embedding_size_port)
        paths = {}
        paths.update(hedge_ins)
        paths.update(additional_states_) 

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=initial_lr,#1e-3,
                decay_steps=decay_steps,
                decay_rate=decay_rate, 
                staircase=True)
        print('done.')
        print('compute payoff:')
        payoff = VanillaOptionDeepHedgingPricer.compute_payoff(n_sims, hedge_ins, portfolio_list,port_vec,days,val_date) 
        print('done.')
        print('train hedge model:')
        hedge_model.train(paths, payoff, lr_schedule, epochs=epochs, batch_size=batch_size, tensorboard_log=tensorboard_logdir, verbose=verbose)
        print('done.')
        return VanillaOptionDeepHedgingPricer.PricingResults(hedge_model, paths=paths, sim_results=simulation_results, payoff=payoff)

if __name__=='__main__':
    pass