from typing import List, Tuple, Union
import copy
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import hiplot as hip
from rivapy.tools.interfaces import _JSONEncoder, _JSONDecoder, FactoryObject
from rivapy.tools.datetime_grid import DateTimeGrid
from rivapy.models.gbm import GBM
from rivapy.pricing.vanillaoption_pricing import (
    VanillaOptionDeepHedgingPricer,
    DeepHedgeModelwEmbedding,
)

def _get_entry(path: str, x: dict):
    path_entry = path.split(".")
    y = x
    for i in range(len(path_entry) - 1):
        y = y[path_entry[i]]
    return y[path_entry[-1]]

def _fulfills(conditions: List[Tuple[str, Union[str, float, int, Tuple]]], x: dict):

    def __fulfills(path, target_value: Union[str, float, int, Tuple], x: dict):
        try:
            entry = _get_entry(path, x)
        except:
            return False
        if isinstance(target_value, tuple):
            return target_value[0] <= entry <= target_value[1]
        return target_value == entry

    fulfills = True
    for condition in conditions:
        fulfills = __fulfills(condition[0], condition[1], x) and fulfills
    return fulfills


def _select(
    conditions: List[Tuple[str, Union[str, float, int, Tuple]]], x: dict
) -> dict:
    result = {}
    for k, v in x.items():
        if not _fulfills(conditions, v):
            continue
        result[k] = v
    return result


class Repo:
    def __init__(self, repo_dir):
        self.repo_dir = repo_dir
        self.results = {}
        try:
            with open(repo_dir + "/results.json", "r") as f:
                self.results = json.load(f, cls=_JSONDecoder)
        except:
            pass

    @staticmethod
    def compute_pnl_figures(pricing_results):
        pnl = pricing_results.hedge_model.compute_pnl(
            pricing_results.paths, pricing_results.payoff
        )
        inputs = pricing_results.hedge_model._create_inputs(pricing_results.paths)
        loss = pricing_results.hedge_model.evaluate(inputs, pricing_results.payoff)
        # delta = pricing_results.hedge_model.compute_delta(pricing_results.paths, -2).reshape((-1,))

        return {
            "mean": pnl.mean(),
            "var": pnl.var(),
            "loss": loss,
            "1%": np.percentile(pnl, 1),
            "99%": np.percentile(pnl, 99),
            "5%": np.percentile(pnl, 5),
            "95%": np.percentile(pnl, 95),
        }

    def run(self, val_date, spec, model, rerun=False, **kwargs):
        params = {}
        params["val_date"] = val_date
        params["spec"] = [
            spec[k].to_dict() for k in range(len(spec))
        ] #{spec[k].id: spec[k]._to_dict() for k in range(len(spec))}
        params["model"] = [
            model[k].to_dict() for k in range(len(model))
        ]  # model.to_dict()
        _kwargs = copy.deepcopy(kwargs)
        _kwargs.pop(
            "tensorboard_logdir", None
        )  # remove  parameters irrelevant for hashing before generating kashkey
        _kwargs.pop("verbose", None)
        params["pricing_param"] = _kwargs
        hash_key = FactoryObject.hash_for_dict(params)
        params["pricing_param"] = kwargs
        params["spec_hash"] = {spec[k].id: spec[k].hash() for k in range(len(spec))}
        params["model_hash"] = {
            model[k].modelname: model[k].hash() for k in range(len(model))
        }
        params["pricing_params_hash"] = FactoryObject.hash_for_dict(kwargs)
        if (hash_key in self.results.keys()) and (not rerun):
            return self.results[hash_key]
        pricing_result = VanillaOptionDeepHedgingPricer.price(
            val_date, spec, model, **kwargs
        )
        params["pnl_result"] = Repo.compute_pnl_figures(pricing_result)
        self.results[hash_key] = params
        with open(self.repo_dir + "/results.json", "w") as f:
            json.dump(self.results, f, cls=_JSONEncoder)
        pricing_result.hedge_model.save(self.repo_dir + "/" + hash_key + "/")
        return pricing_result

    def save(self):
        with open(self.repo_dir + "/results.json", "w") as f:
            json.dump(self.results, f, cls=_JSONEncoder)

    def get_hedge_model(self, hashkey: str) -> DeepHedgeModelwEmbedding:
        return DeepHedgeModelwEmbedding.load(self.repo_dir + "/" + hashkey + "/")

    def get_model(self, hashkey: str) -> GBM:
        return GBM.from_dict(self.results[hashkey]["model"])
    
    def train_embedding(self, hashkey: str):
        return DeepHedgeModelwEmbedding.load(self.repo_dir + "/" + hashkey + "/")

    def simulate_model(
        self,
        hashkey: str,
        n_sims: int,
        seed: int = 42,
        days: int = 30,
        freq: str = "D",
        parameter_uncertainty: bool = False,
        model: list = [GBM(drift=0.0, volatility=0.25)],
        emb: int = 0
    ) -> np.ndarray:
        # res = self.results[hashkey]
        # spec = EuropeanVanillaSpecification.from_dict(res['spec'])
        timegrid = VanillaOptionDeepHedgingPricer._compute_timegrid(days, freq)
        np.random.seed(seed)
        # model = self.get_model(hashkey)
        simulation_results = np.zeros((len(timegrid)+1, n_sims))
        S0 = 1. #ATM option
        emb_vec = np.zeros((n_sims))
        if freq == '12H':
            n = days*2
        else:
            n = days
        model_list = [model]
        n_sims = int(n_sims/len(model_list))
        for i in range(len(model_list)):
            model= model_list[i]
            simulation_results[:,i*n_sims:n_sims*(i+1)] = model.simulate(timegrid, S0=S0, v0=model.v0, M=n_sims,n=n, model_name=model_list[i].modelname)
            emb_vec[i*n_sims:n_sims*(i+1)] = emb    
        return simulation_results, emb_vec

    def get_call_price(self,
        hashkey: str,
            sim_results: np.ndarray, 
               seed: int = 42,
                model: list = [GBM(drift=0.0, volatility=0.25)],
                n_sims: int = 10000, 
                days: int = 30,
                freq: str = 'D')-> np.ndarray:
        tf.random.set_seed(seed)
        np.random.seed(seed+123)

        timegrid = VanillaOptionDeepHedgingPricer._compute_timegrid(days, freq)
        if freq == '12H':
            n = days*2
        else:
            n = days
        model_list = [model]
        n_sims = int(n_sims/len(model_list))
        for i in range(len(model_list)):
            model= model_list[i]
            ttm = (timegrid[-1] - timegrid[0])
            call_price = model.compute_call_price(1.,model.v0,1.,ttm)
        return call_price


    def select(
        self, conditions: List[Tuple[str, Union[str, float, int, Tuple]]]
    ) -> dict:
        return _select(conditions, self.results)

    def plot_hiplot(
        self,
        conditions: List[Tuple[str, Union[str, float, int, Tuple]]] = None,
    ):
        """Plot errorsw.r.t parameters from the given result file with HiPlot

        Args:
            result_file (str): Reultfile
        """
        if conditions is None:
            conditions = []
        experiments = []
        for k, v in self.results.items():
            if not _fulfills(conditions, v):
                continue
            tmp = copy.deepcopy(v["pricing_param"])
            # tmp["x_volatility"] = v["model"]["x_volatility"]
            tmp.update(v["pnl_result"])

            if "tensorboard_logdir" in tmp.keys():
                del tmp["tensorboard_logdir"]

            tmp["key"] = k
            # get relevant model params
            experiments.append(tmp)

        exp = hip.Experiment.from_iterable(experiments)
        exp.display_data(hip.Displays.TABLE).update(
            {
                # In the table, order rows by default
                "order_by": [["mean", "asc"]],
                #'order': ['test loss']
            }
        )  # exp.display_data(hip.Displays.PARALLEL_PLOT)
        exp.display_data(hip.Displays.PARALLEL_PLOT).update(
            {
                #'order': ['stdev test loss', 'train loss', 'test loss'],
            }
        )
        exp.display()
