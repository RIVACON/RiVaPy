from typing import List, Tuple, Union
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import hiplot as hip
from rivapy.tools.interfaces import _JSONEncoder, _JSONDecoder, FactoryObject
from rivapy.tools.datetime_grid import DateTimeGrid
from rivapy.models.gbm import GBM
from rivapy.instruments.specifications import EuropeanVanillaSpecification
from rivapy.pricing.vanillaoption_pricing import VanillaOptionDeepHedgingPricer, DeepHedgeModel 

def _get_entry(path: str, x: dict):
    path_entry = path.split('.')
    y = x
    for i in range(len(path_entry)-1):
        y = y[path_entry[i]]
    return y[path_entry[-1]]

def _fulfills(conditions: List[Tuple[str, Union[str, float, int,Tuple]]], x:dict):
    
    def __fulfills(path, target_value: Union[str, float, int,Tuple], x: dict):
        try:
            entry = _get_entry(path, x)
        except:
            return False
        if isinstance(target_value, tuple):
            return target_value[0] <=entry <= target_value[1]
        return target_value == entry
    fulfills = True
    for condition in conditions:
        fulfills = __fulfills(condition[0], condition[1], x) and fulfills
    return fulfills
    
def _select(conditions: List[Tuple[str, Union[str, float, int,Tuple]]], x:dict)->dict:
    result = {}
    for k,v in x.items():
        if not _fulfills(conditions, v):
            continue
        result[k] = v
    return result


class Repo:
    def __init__(self, repo_dir):
        self.repo_dir = repo_dir
        self.results = {}
        try:
            with open(repo_dir+'/results.json','r') as f:
                self.results = json.load(f, cls=_JSONDecoder)
        except:
            pass
    
    @staticmethod
    def compute_pnl_figures(pricing_results):
        pnl = pricing_results.hedge_model.compute_pnl(pricing_results.paths, pricing_results.payoff)
        inputs = pricing_results.hedge_model._create_inputs(pricing_results.paths)
        loss = pricing_results.hedge_model.evaluate(inputs, pricing_results.payoff)

        return {'mean': pnl.mean(), 'var': pnl.var(), 
                'loss': loss,
                '1%':np.percentile(pnl,1), '99%': np.percentile(pnl,99),
                '5%':np.percentile(pnl,5), '95%': np.percentile(pnl,95)}
    
    def compute_delta_figures(pricing_results, t):
        delta = pricing_results.hedge_model.compute_delta(pricing_results.paths, t).reshape((-1,))
        return {'delta':delta}

    def run(self, val_date, spec, model, rerun=False, **kwargs):
        params = {}
        params['val_date'] = val_date
        params['spec'] = spec._to_dict()
        params['model'] = model.to_dict()
        _kwargs = copy.deepcopy(kwargs)
        _kwargs.pop('tensorboard_logdir', None) #remove  parameters irrelevant for hashing before generating kashkey
        _kwargs.pop('verbose', None)
        params['pricing_param'] = _kwargs
        hash_key = FactoryObject.hash_for_dict(params)
        params['pricing_param'] = kwargs
        params['spec_hash'] = spec.hash()
        params['model_hash'] = model.hash()
        params['pricing_params_hash'] = FactoryObject.hash_for_dict(kwargs) 
        if (hash_key in self.results.keys()) and (not rerun):
            return self.results[hash_key]
        pricing_result = VanillaOptionDeepHedgingPricer.price(val_date, 
                                      spec, 
                                      model, 
                                      **kwargs)
        #params['delta_result'] = Repo.compute_delta_figures(pricing_result,t = -2)
        params['pnl_result'] = Repo.compute_pnl_figures(pricing_result)
        self.results[hash_key] = params
        with open(self.repo_dir+'/results.json','w') as f:
            json.dump(self.results, f, cls=_JSONEncoder)
        pricing_result.hedge_model.save(self.repo_dir+'/'+hash_key+'/')
        return pricing_result
    
    def save(self):
        with open(self.repo_dir+'/results.json','w') as f:
            json.dump(self.results, f, cls=_JSONEncoder)

    def get_hedge_model(self, hashkey:str)->DeepHedgeModel:
        return DeepHedgeModel.load(self.repo_dir+'/'+hashkey+'/')
        
    def get_model(self, hashkey:str)->GBM:
        return GBM.from_dict(self.results[hashkey]['model'])
        
    def simulate_model(self, hashkey: str, n_sims:int, seed: int = 42)->np.ndarray:
        res = self.results[hashkey]
        spec = EuropeanVanillaSpecification.from_dict(res['spec'])
        timegrid,n = VanillaOptionDeepHedgingPricer._compute_timegrid()
        np.random.seed(seed)
        model = self.get_model(hashkey)
        S0 = 1.
        model_result = model.simulate(timegrid, start_value=S0,M = n_sims, n=n)
        return model_result
    
    def select(self, conditions: List[Tuple[str, Union[str, float, int,Tuple]]])->dict:
        return _select(conditions, self.results)


   
        
