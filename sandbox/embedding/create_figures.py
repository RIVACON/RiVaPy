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
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from rivapy.tools.datetime_grid import DateTimeGrid
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


plot_vol_embeddings = False
statistics_n_sims = False
plot_pnl_distributions = False
single_vs_multi_task = True
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
    model_key = '98d12f843c208108e64ec021a7998b824e5cf1dc'
    single_task_model_key = ('ad368106ec8989601fa9c2b67e37e99037f068fe', 100)#('a13d113fa5f60e2023483178291ef8d6b4d7d86d',10)
     #  100 sims
    # 
    model = repo.get_hedge_model(model_key) # 128000
    single_tak_model = repo.get_hedge_model(single_task_model_key[0])
    params = repo.results[model_key]
    n_tasks = len(params['model'])
    vol = repo.results[single_task_model_key[0]]['model'][0]['volatility'] 
    spec = spec_fac.create(params['spec'][0])
    timegrid = DateTimeGrid(start=valdate, end=valdate+dt.timedelta(days= params['days']), freq="D", inclusive='both')

    stoch_model = GBM(drift=0.0, volatility=vol)
    X=stoch_model.simulate(timegrid.timegrid, 1.0, n_sims=single_task_model_key[1] , seed=112)
    paths = {'ADS': X, 'emb_model': np.full((X.shape[1],), 32)}
    X=stoch_model.simulate(timegrid.timegrid, 1.0, n_sims=10_000, seed=645)
    paths_test = {'ADS': X, 'emb_model': np.full((X.shape[1],),n_tasks)}
    payoff, states = VanillaOptionDeepHedgingPricer.compute_payoff(paths, timegrid, np.array([[-1.0]]), [spec] , port_vec=None)
    payoff_test, states_test = VanillaOptionDeepHedgingPricer.compute_payoff(paths_test, timegrid, np.array([[-1.0]]), [spec], port_vec=None)
    pnl_train, pnl_test = model.train_task(model, paths, payoff, paths_test, payoff_test,
                                        initial_lr=0.00001,
                                            decay_steps=200,
                                            decay_rate=0.95,
                                            epochs=100,
                                            batch_size=5,)
    pnl_bs, delta_bs = compute_pnl_bshedge(timegrid, X, spec.strike, True, stoch_model.volatility)
    plt.hist(pnl_bs, bins=100, alpha=1.0, density=True, label='BS hedge')
    paths_test = {'ADS': X, 'emb_model': np.full((X.shape[1],),1)}
    pnl_single_task = single_tak_model.compute_pnl(paths_test, payoff_test)
    plt.hist(pnl_test, bins=100, alpha=0.5, density=True, label='multi-task')
    plt.hist(pnl_single_task, bins=100, alpha=0.5, density=True, label='single-task')
    plt.grid(visible=True, linestyle='--',color='0.8')
    plt.xlabel('PnL')
    plt.ylabel('frequency')
    plt.legend()
    plt.savefig(os.path.join(result_dir, f'pnl_dist_single_vs_multi_task_{single_task_model_key[1]}.png'), dpi=300)
#endregion