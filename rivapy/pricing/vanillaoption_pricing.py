from typing import List, Dict, Protocol, Tuple
import os
import json
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

import datetime as dt
from rivapy.pricing._logger import logger
from rivapy.tools.datetime_grid import DateTimeGrid
from rivapy.pricing.deep_hedging_with_embedding import DeepHedgeModelwEmbedding
from rivapy.tools.interfaces import FactoryObject

class SpecificationDeepHedging(Protocol):
    """Class to define the interfaces for the specification of a portfolio for deep hedging.
    """
    def compute_payoff(self, paths: Dict[str,np.ndarray], timegrid: DateTimeGrid)->Tuple[np.ndarray, np.ndarray|None]:
        """Compute the payoff of the specification on a set of paths.

        Args:
            paths (np.ndarray): Set of paths.
            timegrid (DateTimeGrid): Timegrid used for simulation and hedging.

        Returns:
            Tuple[np.ndarray, np.ndarray|None]: The payoff and the states of the specification (None if no states exist).
        """
        pass

    expiry: dt.datetime|dt.date

    id: str


class DeepHedgingData:
    def __init__(self, 
                id: str,
                paths: Dict[str, np.ndarray], 
                hedge_ins: List[str], 
                additional_states: List[str],
                payoff: np.ndarray, 
                portfolios: np.ndarray|None = None,):
        """Class to store the data for deep hedging.

        Args:
            paths (Dict[str, np.ndarray]): Dictionary of paths.
            hedge_ins (List[str]): List of identifiers for the hedge instruments. 
            additional_states (List[str]): List of additional states.
            payoff (np.ndarray): The payoff of the portfolio.
        """
        self.id = id
        self.paths: Dict[str, np.ndarray] = paths
        self.hedge_ins: List[str] = hedge_ins
        self.additional_states: List[str] = additional_states
        self.payoff: np.ndarray = payoff
        self.portfolios = portfolios

    def save(self, path: str):
        if not os.path.exists(path):
            raise Exception(f'Path {path} does not exist.')
        tmp = {'id': self.id,
            'paths_keys': list(self.paths.keys()), 
                'hedge_ins_keys': self.hedge_ins, 
                'additional_states_keys': self.additional_states}
        with open(os.path.join(path, 'data.json'), 'w') as f:
            json.dump(tmp, f)
        for k,v in self.paths.items():
            np.save(os.path.join(path, k), v)
        np.save(os.path.join(path, 'payoff'), self.payoff)
        np.save(os.path.join(path, 'portfolios'), self.portfolios)

    @staticmethod
    def load(path: str):
        if not os.path.exists(path):
            raise Exception(f'Path {path} does not exist.')
        with open(os.path.join(path, 'data.json'), 'r') as f:
            tmp = json.load(f)
        paths = {}
        for k in tmp['paths_keys']:
            paths[k] = np.load(os.path.join(path, k+'.npy'))
        portfolios = np.load(os.path.join(path, 'portfolios.npy'))
        payoff = np.load(os.path.join(path, 'payoff.npy'))
        return DeepHedgingData(tmp['id'], paths, 
                            tmp['hedge_ins_keys'], tmp['additional_states_keys'], 
                            payoff, portfolios)

class VanillaOptionDeepHedgingPricer:
    class PricingResults:
        def __init__(self, hedge_model: DeepHedgeModelwEmbedding, paths: np.ndarray, payoff):
            self.hedge_model = hedge_model
            self.paths = paths
            self.payoff = payoff

   
    @staticmethod
    def compute_payoff(paths: Dict[str, np.ndarray], 
                       timegrid: DateTimeGrid,
                       portfolios: np.ndarray, 
                       portfolio_instruments: List[SpecificationDeepHedging],
                       port_vec: np.ndarray|None)->Tuple[np.ndarray, Dict[str, np.ndarray]]:
        
        n_paths = next(iter(paths.values())).shape[1]
        payoff = np.zeros((n_paths,))
        states = {}
        if port_vec is None:
            if portfolios.shape[0]>1:
                raise Exception("For more then one portfolio port_vec must be not None.")
            port_vec = np.zeros((n_paths,))
        for i in range(portfolios.shape[0]): # for each portfolio
            selected = i == port_vec # select the paths that shall be used for the portfolio
            tmp={k:v[:,selected] for k,v in paths.items() if len(v.shape)>1}
            tmp.update({k:v for k,v in paths.items() if len(v.shape)==1})
                
            for j in range(len(portfolio_instruments)):
                ins_payoff, ins_states = portfolio_instruments[j].compute_payoff(tmp, timegrid)
                if ins_states is not None:
                    state_key = portfolio_instruments[j].id+':states'
                    if state_key not in states:
                        states[state_key] = np.zeros((timegrid.shape[0], n_paths))
                    states[state_key][:,selected] = ins_states # TODO: States should be onehot encoded?! Up to now if more than one state is given, it is just encoded by numbers....
                payoff[selected] += portfolios[i,j]*ins_payoff
        return payoff, states

    @staticmethod
    def _generate_data(val_date: dt.datetime,
                portfolios: np.ndarray|None,#EuropeanVanillaSpecification,
                portfolio_instruments: List[SpecificationDeepHedging],
                model_list: list|None, #HestonForDeepHedging,
                timegrid: DateTimeGrid,
                n_sims: int, 
                seed: int = 42,
                days: int = 30,
                freq: str = 'D'):
        logger.info('Generate data for deep hedging.')
        logger.debug('Generate paths.')
        simulation_results,emb_vec = VanillaOptionDeepHedgingPricer._generate_paths(seed, model_list, timegrid, n_sims, days, freq)
        logger.debug('Generation of paths done.')
        hedge_ins = {}
        additional_states_ = {}
        logger.debug('Compute payoff and portfolio embeddings.')
        if len(model_list)>1:
            additional_states_["emb_model"] = emb_vec
        port_vec = None
        if portfolios.shape[0]>1: # if more then one portfolio is given, apply embedding of portfolios
            port_vec_split = np.array_split(np.zeros((n_sims), dtype=int),portfolios.shape[0])
            for i in range(portfolios.shape[0]):
                port_vec_split[i][:] = i
            port_vec = np.concatenate(port_vec_split)
            additional_states_["emb_portfolio"] = port_vec
        key = portfolio_instruments[0].udl_id # TODO: generalize to more then one underlying
        hedge_ins[key] = simulation_results
        
        payoff, ins_states = VanillaOptionDeepHedgingPricer.compute_payoff(hedge_ins, timegrid, portfolios, portfolio_instruments, port_vec) 
        logger.debug('Computation of payoff and portfolio embeddings done.')
        keys_additional_states = list(ins_states.keys())+list(additional_states_.keys())  
        paths = {}
        paths.update(hedge_ins)
        paths.update(ins_states)
        paths.update(additional_states_) 
        # create DeepHedgingData object
        data_params = {'portfolio_list':hash(portfolios.data.tobytes()), 
                        'model_list':[v.to_dict() for v in model_list],
                        'n_sims':n_sims,
                        'seed':seed}
        logger.info('Data for deep hedging generated.')
        return DeepHedgingData(FactoryObject.hash_for_dict(data_params), 
                                paths, list(hedge_ins.keys()), 
                                keys_additional_states, 
                                payoff=payoff, portfolios=portfolios)
    
    @staticmethod
    def _generate_paths(seed: int,
                       model_list: list,
                       timegrid: DateTimeGrid,
                       n_sims: int, 
                       days: int,
                       freq: str)->Tuple[np.ndarray,np.ndarray]:
        np.random.seed(seed+123)

        simulation_results = np.zeros((timegrid.shape[0], n_sims))
        ## TODO: still a constant, fixed to 1.: TODO -> set variable!!
        S0 = 1. 
        emb_vec = np.zeros((n_sims))
        if freq == '12H': # TODO model anpassen und diese Zeilen sowie n entfernen
            n = days*2
        else:
            n = days
        n_sims = int(n_sims/len(model_list))
        for i in range(len(model_list)):
            model= model_list[i]
            simulation_results[:,i*n_sims:n_sims*(i+1)] = model.simulate(timegrid.timegrid, S0=S0, n_sims=n_sims)
            emb_vec[i*n_sims:n_sims*(i+1)] = i    
        return simulation_results, emb_vec
    
    @staticmethod
    def price(val_date: dt.datetime,
                portfolios: np.ndarray|None,#EuropeanVanillaSpecification,
                portfolio_instruments: List[SpecificationDeepHedging],
                model_list: list|None, #HestonForDeepHedging,
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
                embedding_size_port: int = 1,
                data: DeepHedgingData|None = None,
                #paths: Dict[str, np.ndarray] = None
                )->Tuple[PricingResults, DeepHedgingData]:
        """Price a vanilla option using deeep hedging

        Args:
            val_date (dt.dtetime): Valuation date
            portfolios (np.ndarray): Each row contains portfolio weights. Defaults to None.
            portfolio_instruments (List[SpecificationDeepHedging]): List of portfolio instruments.
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
            data (DeepHedgingData): Data object containing paths and additional states. If not None, no data will be generated. Defaults to None.
        Returns:
            _type_: _description_

        .. seealso::
        
           DeepHedgeModel
                The general deep hedging model used internally in this pricing method.

        """
        if (model_list is None) and (data is None):
            raise Exception('Either a list of models or data must be specified.')
        if portfolios is None: #  if no portfolio is given, we assume a single portfolio with weight -1.0
            portfolios=np.ones((1,len(portfolio_instruments)))
        tf.keras.backend.set_floatx('float32')

        tf.random.set_seed(seed)
        np.random.seed(seed+123)
        timegrid = DateTimeGrid(start=val_date, end=val_date+dt.timedelta(days=days), freq=freq, inclusive='both')

        if data is None:
            logger.info('Generate data.')
            data = VanillaOptionDeepHedgingPricer._generate_data(val_date, portfolios, portfolio_instruments, model_list, timegrid, n_sims, seed, days, freq)
        else:
            logger.info('Data given, no data creation necessary.')
        
        embedding_definitions  = {}
        if "emb_model" in data.additional_states:
            embedding_definitions["emb_model"] = (len(model_list)+1,embedding_size)
        if "emb_portfolio" in data.additional_states:
            embedding_definitions["emb_portfolio"] = (data.portfolios.shape[0]+1,embedding_size_port)
        hedge_model = DeepHedgeModelwEmbedding(data.hedge_ins, data.additional_states, timegrid=timegrid.timegrid, 
                                        regularization=regularization,
                                        depth=depth, n_neurons=nb_neurons, loss = loss,
                                        transaction_cost = transaction_cost,
                                        embedding_definitions=embedding_definitions)
        

        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=initial_lr,#1e-3,
                decay_steps=decay_steps,
                decay_rate=decay_rate, 
                staircase=True)
                
        logger.info("Start training of deep hedging model.")
        hedge_model.train(data.paths, data.payoff, lr_schedule, epochs=epochs, batch_size=batch_size, tensorboard_log=tensorboard_logdir, verbose=verbose)
        logger.info("Training done.")
        results = VanillaOptionDeepHedgingPricer.PricingResults(hedge_model, paths=data.paths, 
                                                                payoff=data.payoff)
        return results, data

if __name__=='__main__':
    pass