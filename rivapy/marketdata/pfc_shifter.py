import pandas as pd
import numpy as np
import rivapy.tools.interfaces as interfaces
from rivapy.instruments import EnergyFutureSpecifications
from typing import Dict, Set, List

# TODO:
# 1. add logic for creating synthetic contracts
# 2. add possibility to also include peak contracts

def validate_class_input(func):
    def validate_wrapper(self, shape: pd.DataFrame, contracts: Dict[str, EnergyFutureSpecifications]):
        if isinstance(shape, pd.DataFrame):
            if not isinstance(shape.index, pd.DatetimeIndex):
                raise TypeError('The index of the shape DataFrame is not of type pd.DatetimeIndex!')
        else:
            raise TypeError('The shape argument is not of type pd.DataFrame!') 

        contract_scheduled_dates = set(np.concatenate([contract.get_schedule() for contract in contracts.values()]))
        expected_dates = set(shape.index)
        date_diff = expected_dates - contract_scheduled_dates
        if len(date_diff) != 0:
            raise ValueError("The contract dates do not cover each date provided by the shape DataFrame!")
        func(self, shape, contracts) 
    return validate_wrapper

class PFCShifter(interfaces.FactoryObject):
    @validate_class_input
    def __init__(self, shape: pd.DataFrame, contracts: Dict[str, EnergyFutureSpecifications]) -> None:
        self.shape = shape
        self.contracts = contracts
        
        
    def _get_contract_start_end_dates(self) -> List:
        dates = set()
        for contract_schedule in self.contracts.values():
            dates.update(contract_schedule.get_start_end())
        return sorted(list(dates))
    
    
    def _get_forward_price_vector(self) -> np.ndarray:
        return np.array([contract.get_price() for contract in self.contracts.values()]).reshape(-1,1)
    
    
    def generate_transition_matrix(self) -> pd.DataFrame:
        contract_start_and_end_dates = np.array(self._get_contract_start_end_dates())
        
        transition_df = pd.DataFrame(data=np.zeros((len(self.contracts.keys()), len(contract_start_and_end_dates))), 
                                     index=list(self.contracts.keys()), columns=contract_start_and_end_dates)
        
        for contract_name, contract_schedule in self.contracts.items():
            idx = contract_start_and_end_dates.searchsorted(list(contract_schedule.get_start_end()), "right") - 1
                    
            if idx[0] == idx[1]: 
                transition_df.iloc[transition_df.index == contract_name, idx[0]] = 1
            else:
                transition_df.iloc[transition_df.index == contract_name, idx[0]:idx[1]] = 1

        return transition_df.iloc[:, :-1] # drop the last column for the transition matrix
    
    
    def generate_synthetic_contracts(self) -> pd.DataFrame:
        pass
    
    
    def detect_redundant_contracts(self, transition_matrix: pd.DataFrame) -> pd.DataFrame:
        potential_redundant_contracts = []
        np_transition_matrix = transition_matrix.to_numpy()
        for i in range(len(transition_matrix)):
            lst = list(range(len(transition_matrix)))
            lst.remove(i)
            if np.linalg.matrix_rank(np_transition_matrix[lst,:]) == np.linalg.matrix_rank(np_transition_matrix):
                potential_redundant_contracts.append(i)

        base_matrix = np.delete(np_transition_matrix, potential_redundant_contracts, axis=0)

        detected_redundant_contracts = []
        if len(potential_redundant_contracts) != 0:
            for contract_idx in potential_redundant_contracts:
                _temp_matrix = np.concatenate([base_matrix, np_transition_matrix[contract_idx,:].reshape(1,-1)], axis=0)
                if np.linalg.matrix_rank(_temp_matrix) > np.linalg.matrix_rank(base_matrix):
                    base_matrix = _temp_matrix
                else:
                    print(f'Found redundant contract: {transition_matrix.index[contract_idx]}')
                    detected_redundant_contracts.append(transition_matrix.index[contract_idx])

        return transition_matrix.loc[~transition_matrix.index.isin(detected_redundant_contracts), :]
    
    
    def shift(self, transition_matrix: pd.DataFrame) -> pd.DataFrame:
        contract_start_and_end_dates = np.array(self._get_contract_start_end_dates())
        date_tpls = list(zip(contract_start_and_end_dates[:-1], contract_start_and_end_dates[1:]))
        hours_btwn_dates = pd.Series(contract_start_and_end_dates[1:] - contract_start_and_end_dates[:-1]).dt.days.to_numpy().reshape(1,-1) * 24 # 24 only for HPFC
        
        transition_matrix = transition_matrix.to_numpy() * hours_btwn_dates
        transition_matrix = transition_matrix/np.sum(transition_matrix, axis=1).reshape(-1,1)
        fwd_price_vec = self._get_forward_price_vector()

        fwd_price_noc = np.linalg.inv(transition_matrix) @ fwd_price_vec
        pfc = self.shape.copy()
        for i, date_tpl in enumerate(date_tpls):
            if i == len(date_tpls)-1:
                row_filter = (pfc.index >= date_tpl[0]) & (pfc.index <= date_tpl[1])
            else:
                row_filter =  (pfc.index >= date_tpl[0]) & (pfc.index < date_tpl[1])
                
            pfc.iloc[row_filter, 0] = pfc.iloc[row_filter, 0]/np.sum(pfc.iloc[row_filter, 0]) * len(pfc.iloc[row_filter, 0]) * fwd_price_noc[i, 0]
        return pfc


    ## TODO
    def _to_dict(self)->dict:
        self.to_dict()
