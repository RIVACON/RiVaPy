from typing import Union, Callable
import numpy as np
import scipy
from rivapy.tools.interfaces import FactoryObject

class HistoricSim(FactoryObject):
    def __init__(self, historic_values:np.ndarray, description:str):
        """Historic simulation model. It simulates price based on the discrete log-return distribution of the given historic values.

        Args:
            historic_values (np.ndarray): The historic values of the asset.
            description (str): Description of the model.
        """
        if not isinstance(historic_values, np.ndarray):
            self.historic_values = np.array(historic_values)
        else:
            self.historic_values = historic_values
        self.description = description

    def _to_dict(self) -> dict:
        return {'historic_values': list(self.historic_values), 
                'description': self.description}
    
    def simulate(self, timegrid, S0, n_sims: int)->np.ndarray:
        """ Simulate the historic values

        It computes the log-returns of the historic values and simulates the paths based on the log-returns.

        Args:
            timegrid (np.ndarray): One dimensional array containing the time points where the process will be simulated (containing 0.0 as the first timepoint).
            S0 (Union[float, np.ndarray]): Either a float or an array (for each path) with the start value of the simulation.
            n_sims = number of simulations
        """
        log_returns = np.log(self.historic_values[1:]/self.historic_values[:-1])
        rnd = np.random.choice(log_returns, size=(timegrid.shape[0]-1,n_sims))
        result = np.empty((timegrid.shape[0],n_sims))
        result[0,:] = S0
        for i in range(1,timegrid.shape[0]):
            result[i,:] = result[i-1,:]*np.exp(rnd[i-1,:])
        return result