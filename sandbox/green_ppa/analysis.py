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

    def run(self, val_date, ppa_spec, model, **kwargs):
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
        if hash_key in self.results.keys():
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
    
    def plot_hiplot(self, error = 'mean_rmse_scaled_recalib', error_train='mean_rmse_train_scaled'):
        """Plot errorsw.r.t parameters from the given result file with HiPlot

        Args:
            result_file (str): Reultfile
        """
        
        experiments = []
        for k,v in self.results.items():
            tmp = copy.deepcopy(v['pricing_param'])
            del tmp['initial_forecasts']
            del tmp['power_fwd_prices']
            tmp['n_forecast_hours'] = len( tmp['forecast_hours'])
            del tmp['forecast_hours']
            tmp['n_additional_states'] = len( tmp['additional_states'])
            del tmp['additional_states']
            if 'tensorboard_logdir' in tmp.keys():
                del tmp['tensorboard_logdir']
            d = v['result']
            tmp.update( {x: d[x] for x in  d if x not in ['seed']})
            tmp['key'] = k
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
    
def plot_paths(paths: ForwardSimulationResult, forecast_points):
    plt.figure(figsize=(14,5))
    i_ = 1
    for i in ['Power_Germany_FWD0', 'Onshore_FWD0', 'Offshore_FWD0']:
        plt.subplot(1,3,i_)
        i_ += 1
        paths_ = paths.get(i, forecast_points) 
        for j in range(200):
            plt.plot(paths_[:,j], '-r', alpha=0.1)
        plt.ylabel(i)

    plt.figure(figsize=(14,14))
    i_ = 1
    for i in ['Power_Germany_FWD0', 'Onshore_FWD0', 'Offshore_FWD0']:
        paths_1 = paths.get(i, forecast_points)
        for j in  ['Power_Germany_FWD0', 'Onshore_FWD0', 'Offshore_FWD0']:
            paths_2 = paths.get(j, forecast_points)
            plt.subplot(3,3,i_)
            i_ += 1
            if i==j:
                plt.hist(paths_1[-1,:], bins=100)
                plt.xlabel(i)
            else:
                plt.plot(paths_1[-1,:], paths_2[-1,:],'.')
                plt.xlabel(i)
                plt.ylabel(j)

#from rivapy.tools.interfaces import FactoryObject

#def compute_pnl(pricing_results):
#    return pricing_results.hedge_model.compute_pnl(pricing_results.paths, pricing_results.payoff)