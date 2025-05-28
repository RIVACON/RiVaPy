import datetime as dt
import numpy as np
import sys
sys.path.append('C:/Users/doeltz/development/RiVaPy/')
from rivapy.instruments import GreenPPASpecification
from rivapy.pricing.pricing_request import GreenPPAPricingRequest
from rivapy.pricing.pricing_results import PricingResultKeys as PRK
from rivapy.models.base_model import BaseModel
from rivapy.tools import DateTimeGrid
from rivapy.pricing._logger import logger

def price(val_date: dt.datetime, 
          spec: GreenPPASpecification, 
          model: BaseModel,
          pr: GreenPPAPricingRequest,
          n_sims: int = 1000, 
          seed: int = 42) -> float:
    """Compute expected value of Green PPA by MC Simulation for a given model.

    Args:
        spec (GreenPPASpecification): Green PPA specification
        model (BaseModel): Model to use for pricing

    Returns:
        float: Price of the Green PPA
    """    
    def round_to_hour(t):
        # Rounds to nearest hour by adding a timedelta hour if minute >= 30
        return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour) + dt.timedelta(hours=t.minute//30))
    
    if spec.udl not in model.udls():
        raise ValueError('Model does not support underlying '+spec.udl)
    schedule = spec.get_schedule()
    if schedule[-1] <= val_date:
        return 0.0 # specification is expired
    dg = DateTimeGrid(start=round_to_hour(val_date), 
                      end=schedule[-1], 
                      freq='1H') #TODO: What to do with the frequency?
    logger.debug('Start model simulation with %d paths.', n_sims)
    results = model.simulate(dg, n_sims=n_sims, start_value_wind=0.5, 
                                start_value_load=0.5, start_value_solar=0.5, 
                                seed=seed)
    logger.debug('Finished model simulation with %d paths.', n_sims)
    logger.debug('Compute indices for schedule.')
    df = dg.df
    logger.debug('Finished computing indices for schedule.')
    index = df[df.dates.isin(schedule)].index.values
    result = {}
    quantity = results[spec.technology][index,:]# overall total quantity produced. This must be scaled by capacity of the plant in relation to the total capacity

    cf = np.multiply(quantity,results['price'][index,:]-spec.fixed_price)
    if pr.theo_val:
        result[PRK.theo_val] = np.mean(cf)
    if pr.cf_expected:
        result[PRK.cf_expected] = np.mean(cf,axis=1)
    if pr.cf_paths:
        result[PRK.cf_paths] = cf
    
    return result

if __name__=="__main__":
    import rivapy.sample_data.residual_demand_models as rdm_sample
    from rivapy.tools import SimpleSchedule
    simple_schedule = SimpleSchedule(dt.datetime(2022,12,1), dt.datetime(2023,12,1,4,0,0), freq='1H')
    green_ppa = GreenPPASpecification(simple_schedule, 
                                    fixed_price = 10.0, 
                                    max_capacity=10, 
                                    technology = 'wind', 
                                    udl = 'power',
                                    location='')
    rd_model = rdm_sample.WagnerModel.residual_demand_model()
    price(dt.datetime(2022,12,1), green_ppa, rd_model)