import datetime as dt
import sys
#sys.path.insert(0,'../..')
sys.path.append('/home/doeltz/doeltz/development/RiVaPy')
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from rivapy.tools.datetime_grid import DateTimeGrid
from rivapy.tools.interfaces import _JSONEncoder
from rivapy.models.gbm import GBM
import rivapy.instruments.factory as spec_fac
from rivapy.models.heston_for_DH import HestonForDeepHedging
from rivapy.models.heston_with_jumps import HestonWithJumps
from rivapy.models.barndorff_nielsen_shephard import BNS
from rivapy.instruments.specifications import EuropeanVanillaSpecification
from rivapy.pricing.vanillaoption_pricing import VanillaOptionDeepHedgingPricer, DeepHedgeModelwEmbedding
import rivapy.pricing.analytics as analytics
import logging
import analysis


valdate = dt.datetime(2023, 1, 1)

logger = logging.getLogger(__name__)

def compute_pnl_bshedge(timegrid: DateTimeGrid, paths:np.ndarray, strike: float, 
                        is_call: bool, implied_vol: float)->Tuple[np.ndarray, np.ndarray]:
    if isinstance(timegrid, DateTimeGrid):
        timegrid = timegrid.timegrid
    deltas = np.empty((timegrid.shape[0], paths.shape[1]))
    pnl = np.zeros((paths.shape[1],))
    for i in range(timegrid.shape[0]):
        t = timegrid[i]
        T = timegrid[-1]-t
        if i==0:
            deltas[i,:] = analytics.compute_european_delta(paths[i,:], strike, 0.0, implied_vol, T, is_call)
            pnl -= deltas[i,:]*paths[i,:]
        if i>0:
            if i < timegrid.shape[0]-1:
                deltas[i,:] = analytics.compute_european_delta(paths[i,:], strike, 0.0, implied_vol, T, is_call)
                pnl -= (deltas[i,:]-deltas[i-1,:])*paths[i,:]
            else:
                pnl += deltas[i-1,:]*paths[i,:]
    if is_call:
        pnl -= np.maximum(paths[-1,:]-strike,0.0)
    else:
        pnl -= np.maximum(strike-paths[-1,:],0.0)
    return pnl, deltas

result_dir = '/home/doeltz/doeltz/development/RiVaPy/sandbox/embedding/figures/'
repo = analysis.Repo('/home/doeltz/doeltz/development/repos/embedding_gbm')

axis_label_fontsize = 12
plot_vol_embeddings = False
statistics_n_sims = False
plot_pnl_distributions = False
single_vs_multi_task = False
sv_embeddings = True

#region ============ plot vol embeddings ====================
if plot_vol_embeddings:
    logger.info('Plotting embeddings vs volatility')
    models = [("4798c45448df2ba298b1db7d93b24df05b585331", "8"), ("4135388f85a4d6d7dfdef7aed65237876f50ca96", "16"), ("b12172ac880c138d601d1065c2be89f00b155f8b", "32")]
    for m in models:
        model = repo.get_hedge_model(m[0])
        emb = model.get_embedding('emb_model')
        vols = [v['volatility'] for v in repo.results[m[0]]['model']]
        plt.plot(vols,emb[0][:-2,0] ,'-o', label=m[1])
    plt.grid(visible=True, linestyle='--',color='0.8')
    plt.legend()
    plt.xlabel('volatility')
    plt.ylabel('embedding')
    plt.savefig(os.path.join(result_dir, 'embedding_vs_volatility_GBM_1D.png'), dpi=300)
#endregion


#region ============ pnl-var w.r.t. different number of simulations  ====================
if statistics_n_sims:
    models = [
        ('aebf9e3e20461ddcba05c2ef05315cf913d5a43b', 3200),
        ('43bc561020a0e760e793208076b361cba69a028b', 6400),
        ('5e30ec38750f3addca6e6f4953239a732df81dcf', 12800),
        ('6a1aefa7f695672add28a53902b142be6219aacc', 25600),
        ( 'a304c06cd5166d0e70d132d364873606f12dc04b', 64000),
        ('98d12f843c208108e64ec021a7998b824e5cf1dc', 128000),
        # ('b12172ac880c138d601d1065c2be89f00b155f8b', 256000),
        # ('7db4b9e5d94ca03c90c168678db35714a713d63b', 512000),
        # ('48b3117b8b577cd95383e6c68ae1d77467c67cc8', 1024000),
        #('cdac6aa560d2ac12d57b7707dc41ec072ecd009c', 2048000),
    ] # n_models = 32
    ref_model = repo.get_hedge_model(models[0][0])
    params = repo.results[models[0][0]]
    spec = spec_fac.create(params['spec'][0])
    timegrid = DateTimeGrid(start=valdate, end=valdate+dt.timedelta(days= params['days']), freq="D", inclusive='both')
    paths, model_embed_vec = repo.simulate_model(valdate, 32*10_000, seed=1783, days=30, model=params['model'])
    paths = {spec.udl_id: paths, 'emb_model': model_embed_vec}
    payoff,_ = VanillaOptionDeepHedgingPricer.compute_payoff(paths, timegrid, np.array([[-1.0]]), [spec], port_vec=None)

    results = {}
    max_embeddings = int(model_embed_vec.max())+1
    paths_tmp = paths[spec.udl_id][:,model_embed_vec[:]==0]
    print(params['model'][0])
    pnl_bs_min, deltas = compute_pnl_bshedge(ref_model.timegrid, paths_tmp, 1.0, True, params['model'][0]['volatility'])
    pnl_bs_min = np.std(pnl_bs_min)
    paths_tmp = paths[spec.udl_id][:,model_embed_vec[:]==max_embeddings-1]
    pnl_bs_max, deltas = compute_pnl_bshedge(ref_model.timegrid, paths_tmp, 1.0, True, params['model'][-1]['volatility'])
    pnl_bs_max = np.std(pnl_bs_max)
    std_dev = []
    std_dev_min = []
    std_dev_max = []
    n_sims= [m[1] for m in models]
    for m in models:
        model = repo.get_hedge_model(m[0])
        pnl = model.compute_pnl(paths, payoff)
        pnl_std_single = []
        for i in range(max_embeddings):
            pnl_std_single.append(np.std(pnl[model_embed_vec[:]==i]))
        results[m[1]] = {'mean': np.mean(pnl), 
                         'std': np.mean(pnl_std_single), 
                         'std min': np.min(pnl_std_single), 
                        'std max': np.max(pnl_std_single),
                         '1\% quantile': np.quantile(pnl, 0.01), 
                         '10\% quantile': np.quantile(pnl, 0.1)}
        std_dev.append(results[m[1]]['std'])
        std_dev_min.append(results[m[1]]['std min'])
        std_dev_max.append(results[m[1]]['std max'])
    result = pd.DataFrame(results).transpose()
    result = result.style.format(decimal=',', thousands='.', precision=2)
    result.to_latex(os.path.join(result_dir, 'pnl_var_n_simulations.csv'))
    plt.plot(n_sims, std_dev, '-o', label='mean of std dev of PnL')
    plt.plot(n_sims, [pnl_bs_max]*len(n_sims), '--', color='k', label='PnL BS hedge, vol=0.1')
    plt.plot(n_sims, [pnl_bs_min]*len(n_sims), ':', color='k', label='PnL BS hedge, vol=0.8')
    plt.grid(visible=True, linestyle='--',color='0.8')
    plt.fill_between(n_sims, std_dev_min, std_dev_max, color='b', alpha=.15)
    plt.xlabel('number of simulations')
    plt.ylabel('standard deviation of PnL')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'pnl_var_n_simulations.png'), dpi=300)

#endregion

#region =============pnl distribution ========================

if plot_pnl_distributions:
    models = [
        ('aebf9e3e20461ddcba05c2ef05315cf913d5a43b', 3200),
        ('98d12f843c208108e64ec021a7998b824e5cf1dc', 128000),
    ] # n_models = 32
    ref_model = repo.get_hedge_model(models[0][0])
    params = repo.results[models[0][0]]
    spec = spec_fac.create(params['spec'][0])
    timegrid = DateTimeGrid(start=valdate, end=valdate+dt.timedelta(days= params['days']), freq="D", inclusive='both')
    paths, model_embed_vec = repo.simulate_model(valdate, 32*10_000, seed=1783, days=30, model=params['model'])
    paths = {spec.udl_id: paths, 'emb_model': model_embed_vec}
    payoff,_ = VanillaOptionDeepHedgingPricer.compute_payoff(paths, timegrid, np.array([[-1.0]]), [spec], port_vec=None)

    for vol in [10,20]: 
        plt.figure()
        paths_tmp = paths[spec.udl_id][:,model_embed_vec[:]==vol]
        v = params['model'][vol]
        print(params['model'][vol])
        pnl_bs, deltas = compute_pnl_bshedge(ref_model.timegrid, paths_tmp, 1.0, True, v['volatility'])
        plt.hist(pnl_bs, bins=100, alpha=1.0, density=True, label='BS hedge, vol={:.2f}'.format(v['volatility']))
        for m in models:
            model = repo.get_hedge_model(m[0])
            pnl = model.compute_pnl(paths, payoff)
            plt.hist(pnl[model_embed_vec[:]==vol], bins=100, alpha=0.5, density=True,label='model: {} sims'.format(m[1]))
        plt.grid(visible=True, linestyle='--',color='0.8')
        plt.xlabel('PnL')
        plt.ylabel('frequency')
        plt.legend()
        plt.savefig(os.path.join(result_dir, f'pnl_dist_n_simulations_{v["volatility"] }.png'), dpi=300)

#endregion

#region ============== retraining comparison multitask vs normal =================
if single_vs_multi_task:
    model_key = 'cdac6aa560d2ac12d57b7707dc41ec072ecd009c'
    #single_task_model_key = ('f40c1cfe45bb2d6af6cd1e8f99307be89ffe2623',10)
    #single_task_model_key =('64d1b8a2dd2ac44ed16b4843e23246a04fa97884',100) 
    single_task_model_key = ('3c28eba8bceef8978276fc25572c3dfa9fbf0ea9',1000)
    #('4b44d9bbf9cbe85bfe6da4273c810d670571b6ca', 100)# ##
     #  100 sims
    # 
    model = repo.get_hedge_model(model_key) # 128000
    single_task_model = repo.get_hedge_model(single_task_model_key[0])
    params = repo.results[model_key]
    n_tasks = len(params['model'])
    vol = repo.results[single_task_model_key[0]]['model'][0]['volatility'] 
    spec = spec_fac.create(params['spec'][0])
    timegrid = DateTimeGrid(start=valdate, end=valdate+dt.timedelta(days= params['days']), freq="D", inclusive='both')

    stoch_model = GBM(drift=0.0, volatility=vol)
    X=stoch_model.simulate(timegrid.timegrid, 1.0, n_sims=single_task_model_key[1] , seed=112)
    paths = {'ADS': X, 'emb_model': np.full((X.shape[1],), n_tasks+1)}
    X=stoch_model.simulate(timegrid.timegrid, 1.0, n_sims=10_000, seed=645)
    paths_test = {'ADS': X, 'emb_model': np.full((X.shape[1],),n_tasks+1)}
    payoff, states = VanillaOptionDeepHedgingPricer.compute_payoff(paths, timegrid, np.array([[-1.0]]), [spec] , port_vec=None)
    payoff_test, states_test = VanillaOptionDeepHedgingPricer.compute_payoff(paths_test, timegrid, np.array([[-1.0]]), [spec], port_vec=None)
    pnl_train, pnl_test = model.train_task(model, paths, payoff, paths_test, payoff_test,
                                        initial_lr=1e-3,
                                            decay_steps=200,
                                            decay_rate=1.0,
                                            epochs=500,
                                            batch_size=1000,)
    pnl_bs, delta_bs = compute_pnl_bshedge(timegrid, X, spec.strike, True, stoch_model.volatility)
    plt.hist(pnl_bs, bins=100, alpha=1.0, density=True, label='BS hedge')
    paths_test = {'ADS': X, 'emb_model': np.full((X.shape[1],),0)}
    pnl_single_task = single_task_model.compute_pnl(paths_test, payoff_test)
    #print(np.var(pnl_bs), np.var(pnl_test), np.var(pnl_single_task))
    plt.hist(pnl_single_task, bins=100, alpha=0.5, density=True, label='single-task')
    plt.hist(pnl_test, bins=100, alpha=0.5, density=True, label='multi-task')
    
    plt.grid(visible=True, linestyle='--',color='0.8')
    plt.xlabel('PnL')
    plt.ylabel('frequency')
    plt.legend()
    plt.savefig(os.path.join(result_dir, f'pnl_dist_single_vs_multi_task_{single_task_model_key[1]}.png'), dpi=300)
    ### delta plot
    t = -10
    plt.figure()
    plt.plot(X[t-1,:],delta_bs[t-1,:], '.', label='BS delta')
    delta = single_task_model.compute_delta(paths_test, t=t).reshape((-1,))
    plt.plot(X[t,:],delta, '.', label='single-task delta')
    paths_test['emb_model'] = np.full((X.shape[1],),n_tasks+1)
    delta = model.compute_delta(paths_test, t=t).reshape((-1,))
    plt.plot(X[t,:],delta, '.', label='multi-task delta')
    plt.grid(visible=True, linestyle='--',color='0.8')
    plt.xlabel('spot')
    plt.ylabel('delta')
    plt.legend()
    plt.savefig(os.path.join(result_dir, f'delta_single_vs_multi_task_{single_task_model_key[1]}.png'), dpi=300)

#endregion   

# region create serialized models 
if False:
    import ast  
    with open('/home/doeltz/doeltz/development/RiVaPy/sandbox/embedding/model_params_dict.txt') as f: 
        data = f.read() 
    model_params = ast.literal_eval(data) 

    models =[] 
    for n_models_per_model_type in [20]:#[2, 8, 16, 32]:#
        model = []
        for i in range(n_models_per_model_type):
            #model.append(HestonForDeepHedging(rate_of_mean_reversion = 0.6067,long_run_average = 0.0707,
            #              vol_of_vol = 0.2928, correlation_rho = -0.757,v0 = 0.0654))
            #model.append(GBM(0.,vol_list[i]))
            model.append(GBM(drift=0.0,volatility=model_params['GBM']['vol'][i]).to_dict())
            model.append(HestonForDeepHedging(rate_of_mean_reversion = model_params['Heston']['rate_of_mean_reversion'][i],
                                            long_run_average = model_params['Heston']['long_run_average'][i],
                                            vol_of_vol = model_params['Heston']['vol_of_vol'][i], 
                                            correlation_rho = model_params['Heston']['correlation_rho'][i],
                                            v0 = model_params['Heston']['v0'][i]).to_dict())
            model.append(HestonWithJumps(rate_of_mean_reversion = model_params['Heston with Jumps']['rate_of_mean_reversion'][i],
                                        long_run_average = model_params['Heston with Jumps']['long_run_average'][i],
                                        vol_of_vol = model_params['Heston with Jumps']['vol_of_vol'][i], 
                                        correlation_rho = model_params['Heston with Jumps']['correlation_rho'][i],
                                        muj = 0.1791,sigmaj = 0.1346, 
                                        lmbda = model_params['Heston with Jumps']['lmbda'][i],
                                        v0 = model_params['Heston with Jumps']['v0'][i]).to_dict())
            model.append(BNS(rho =model_params['BNS']['rho'][i],
                            lmbda=model_params['BNS']['lmbda'][i],
                            b=model_params['BNS']['b'][i],
                            a=model_params['BNS']['a'][i],
                            v0 = model_params['BNS']['v0'][i]).to_dict())
        models.append(model) 
        with open("models.json", "w") as f:
            json.dump(models, f, cls=_JSONEncoder)   
#endregion

#region sv models histogram, different embedding sizes,
if sv_embeddings:
    repo = analysis.Repo('/home/doeltz/doeltz/development/repos/embedding_sv')
    sv_result_dir = os.path.join(result_dir,'sv')
    models = [
            ('ed59c329abfc2053a5bd1a76f80c9546cf2731ab', 4),
            ('a942c45b39ead4a609d33a307175fa23929b7820', 8),
            ('6c2b9753b1853ce3a5c8c744f3d857c870a48e36', 16)
        ] 
    ref_model = repo.get_hedge_model(models[0][0])
    params = repo.results[models[0][0]]
    spec = spec_fac.create(params['spec'][0])
    timegrid = DateTimeGrid(start=valdate, end=valdate+dt.timedelta(days= params['days']), freq="D", inclusive='both')
    stoch_models = [params['model'][0]]
    paths, model_embed_vec = repo.simulate_model(valdate, 10_000*len(stoch_models), seed=1783, days=30, model=stoch_models )
    paths = {spec.udl_id: paths, 'emb_model': model_embed_vec}
    payoff,_ = VanillaOptionDeepHedgingPricer.compute_payoff(paths, timegrid, np.array([[-1.0]]), [spec], port_vec=None)

    for m in models:
        model = repo.get_hedge_model(m[0])
        pnl = model.compute_pnl(paths, payoff)
        plt.hist(pnl, bins=100, alpha=0.3, density=True,label='embedding size: {}'.format(m[1]))
    plt.xlabel('PnL', fontsize=axis_label_fontsize)
    plt.savefig(os.path.join(sv_result_dir, 'pnl_dist_sv_embeddings.png'), dpi=300)
    
#endregion
