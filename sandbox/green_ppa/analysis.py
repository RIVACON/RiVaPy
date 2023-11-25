from typing import List, Tuple, Union
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import hiplot as hip
from rivapy.tools.interfaces import _JSONEncoder, _JSONDecoder, FactoryObject
from rivapy.models.residual_demand_fwd_model import LinearDemandForwardModel, ForwardSimulationResult
from rivapy.instruments import GreenPPASpecification
from rivapy.pricing.green_ppa_pricing import GreenPPADeepHedgingPricer, DeepHedgeModel #,PPAHedgeModel

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

    def run(self, val_date, ppa_spec, model, rerun=False, **kwargs):
        params = {}
        params['val_date'] = val_date
        params['ppa_spec'] = ppa_spec.to_dict()
        params['model'] = model.to_dict()
        _kwargs = copy.deepcopy(kwargs)
        _kwargs.pop('tensorboard_logdir', None) #remove  parameters irrelevant for hashing before generating kashkey
        _kwargs.pop('verbose', None)
        params['pricing_param'] = _kwargs
        hash_key = FactoryObject.hash_for_dict(params)
        params['pricing_param'] = kwargs
        params['ppa_spec_hash'] = ppa_spec.hash()
        params['model_hash'] = model.hash()
        #params['pricing_params_hash'] = FactoryObject.hash_for_dict(kwargs)
        if (hash_key in self.results.keys()) and (not rerun):
            return self.results[hash_key]
        pricing_result =  GreenPPADeepHedgingPricer.price(val_date, 
                                      ppa_spec, 
                                      model,
                                    **kwargs)
        params['result'] = Repo.compute_pnl_figures(pricing_result)
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
        
    def get_model(self, hashkey:str)->LinearDemandForwardModel:
        return LinearDemandForwardModel.from_dict(self.results[hashkey]['model'])
        
    def simulate_model(self, hashkey: str, n_sims:int, seed: int = 42)->np.ndarray:
        res = self.results[hashkey]
        spec = GreenPPASpecification.from_dict(res['ppa_spec'])
        timegrid,expiries ,forecast_points = GreenPPADeepHedgingPricer._compute_points(res['val_date'],
                                                                         spec,
                                                                        forecast_hours=res['pricing_param']['forecast_hours'])
        np.random.seed(seed)
        model = self.get_model(hashkey)
        rnd = np.random.normal(size=model.rnd_shape(n_sims=n_sims, n_timesteps=timegrid.shape[0]))
        model_result = model.simulate(timegrid.timegrid, rnd, expiries=expiries,
                                       initial_forecasts=res['pricing_param']['initial_forecasts'],
                                        power_fwd_prices=res['pricing_param']['power_fwd_prices'])
        return model_result, forecast_points
    
    def select(self, conditions: List[Tuple[str, Union[str, float, int,Tuple]]])->dict:
        return _select(conditions, self.results)
    
    def plot_hiplot(self, error = 'mean_rmse_scaled_recalib', error_train='mean_rmse_train_scaled', conditions: List[Tuple[str, Union[str, float, int,Tuple]]]=None):
        """Plot errorsw.r.t parameters from the given result file with HiPlot

        Args:
            result_file (str): Reultfile
        """
        if conditions is None: 
            conditions = []
        experiments = []
        for k,v in self.results.items():
            if not _fulfills(conditions, v):
                continue
            tmp = copy.deepcopy(v['pricing_param'])
            tmp['x_volatility'] = v['model']['x_volatility'] 
            tmp['loss_type'] = tmp['loss']
            for l,w in tmp['initial_forecasts'].items():
                tmp['Forecast_'+l] = w[0]
            del tmp['initial_forecasts']
            del tmp['power_fwd_prices']
            tmp['dtm'] = (v['ppa_spec']['schedule'][0]-v['val_date']).days
            tmp['n_forecast_hours'] = len( tmp['forecast_hours'])
            del tmp['forecast_hours']
            tmp['n_additional_states'] = len( tmp['additional_states'])
            del tmp['additional_states']
            if 'tensorboard_logdir' in tmp.keys():
                del tmp['tensorboard_logdir']
            d = v['result']
            tmp.update( {x: d[x] for x in  d if x not in ['seed']})
            tmp['key'] = k
            # get relevant model params
            tmp['vol_onshore'] = v['model']['wind_power_forecast']['region_forecast_models'][0]['model']['volatility']
            tmp['idiosyncratic_vol'] = v['model']['x_volatility']
            tmp['ppa_strike'] = v['ppa_spec']['fixed_price']
            tmp['max_capacity'] = v['ppa_spec']['max_capacity']
            tmp['power_fwd'] = v['pricing_param']['power_fwd_prices'][0]
            region_forecast_models = v['model']['wind_power_forecast']['region_forecast_models']
            for r in region_forecast_models:
                if r['model']['region'] == 'onshore':
                    tmp['vol_onshore'] = r['model']['volatility']
            experiments.append(tmp)
        

        exp = hip.Experiment.from_iterable(experiments)
        exp.display_data(hip.Displays.TABLE).update({
                # In the table, order rows by default
                'order_by': [['mean', 'asc']],
                #'order': ['test loss']
        })#exp.display_data(hip.Displays.PARALLEL_PLOT)
        exp.display_data(hip.Displays.PARALLEL_PLOT).update({
                #'order': ['stdev test loss', 'train loss', 'test loss'], 

        })
        exp.display()
    
def plot_paths(paths: ForwardSimulationResult, 
               forecast_points, 
               result_dir:str = None):
    plt.figure(figsize=(16,5))
    i_ = 1
    for i in ['Power_Germany_FWD0', 'Onshore_FWD0', 'Offshore_FWD0']:
        plt.subplot(1,3,i_)
        i_ += 1
        paths_ = paths.get(i, forecast_points) 
        for j in range(200):
            plt.plot(paths_[:,j], '-r', alpha=0.1)
        plt.ylabel(i)
        plt.xlabel('hour')
    if result_dir is not None:
        plt.savefig(result_dir+'paths.png', dpi=300)
    plt.figure(figsize=(16,16))
    i_ = 1
    for i in ['Power_Germany_FWD0', 'Onshore_FWD0', 'Offshore_FWD0']:
        paths_1 = paths.get(i, forecast_points)
        for j in  ['Power_Germany_FWD0', 'Onshore_FWD0', 'Offshore_FWD0']:
            paths_2 = paths.get(j, forecast_points)
            plt.subplot(3,3,i_)
            i_ += 1
            if i==j:
                plt.hist(paths_1[-1,:], bins=100, density=True)
                plt.xlabel(i)
            else:
                plt.plot(paths_1[-1,:], paths_2[-1,:],'.')
                plt.xlabel(i)
                plt.ylabel(j)
    if result_dir is not None:
        plt.savefig(result_dir+'paths_final_scatter.png', dpi=300)

#from rivapy.tools.interfaces import FactoryObject

#def compute_pnl(pricing_results):
#    return pricing_results.hedge_model.compute_pnl(pricing_results.paths, pricing_results.payoff)