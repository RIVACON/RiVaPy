import pandas as pd
import numpy as np
import rivapy.tools.interfaces as interfaces
from rivapy.instruments import EnergyFutureSpecifications
from typing import Dict, Set, List

class PFCShifter(interfaces.FactoryObject):
    def __init__(self, shape: pd.DataFrame, contracts: Dict[str, EnergyFutureSpecifications]) -> None:
        self._validate_inputs(shape=shape, contracts=contracts)
        self.shape = shape
        self.contracts = contracts
    
    @staticmethod
    def _validate_inputs(shape:pd.DataFrame, contracts: Dict[str, EnergyFutureSpecifications]):
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
        return None
        
    def _get_contract_start_end_dates(self) -> List:
        dates = set()
        for contract_schedule in self.contracts.values():
            dates.update(contract_schedule.get_start_end())
        return sorted(list(dates))
    
    def generate_transition_matrix(self) -> pd.DataFrame:
        contract_start_and_end_dates = np.array(self._get_contract_start_end_dates())
        
        transition_df = pd.DataFrame(data=np.zeros((len(self.contracts.keys()), len(contract_start_and_end_dates)-1)), 
                                     index=list(self.contracts.keys()), columns=contract_start_and_end_dates[:-1])
        
        for contract_name, constract_schedule in self.contracts.items():
            idx = contract_start_and_end_dates[:-1].searchsorted(list(constract_schedule.get_start_end()), "right") - 1

            if idx[0] == idx[1]: 
                transition_df.iloc[transition_df.index == contract_name, idx[0]] = 1
            else:
                transition_df.iloc[transition_df.index == contract_name, idx[0]:idx[1]] = 1

        return transition_df
    
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

    def non_overlapping_structure(self):
        pass

    def get_nonredundant_transition_matrix(self, transition_matrix:pd.DataFrame):
        pass


    ## TODO
    def _to_dict(self)->dict:
        self.to_dict()

