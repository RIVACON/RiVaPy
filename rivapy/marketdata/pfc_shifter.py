import itertools
import pandas as pd
import numpy as np
import rivapy.tools.interfaces as interfaces
from rivapy.instruments import EnergyFutureSpecifications
from typing import Dict, Set, List
from collections import defaultdict


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
        self._redundant_contracts: Dict[str, EnergyFutureSpecifications] = {}
        self._synthetic_contracts: Dict[str, EnergyFutureSpecifications] = {}
        
    def _get_contract_start_end_dates(self) -> List:
        dates = set()
        for contract_schedule in self.contracts.values():
            dates.update(contract_schedule.get_start_end())
        return sorted(list(dates))
    
    
    def _get_forward_price_vector(self) -> np.ndarray:
        _dict = {**self.contracts, **self._synthetic_contracts}
        return np.array([contract.get_price() for contract in  _dict.values()]).reshape(-1,1)
    
    
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
        
        # update the contracts dictionary, but still keep te information about the redundant contracts
        self._redundant_contracts = {}
        for contract in detected_redundant_contracts:
            self._redundant_contracts[contract] = self.contracts[contract]
            del self.contracts[contract] # MAYBE THIS IS WRONG, SINCE THIS COULD LEAD TO DELETION OF TIME GRIND POINTS 
        #self._redundant_contracts = {contract: self.contracts[contract] for contract in detected_redundant_contracts}
        return transition_matrix.loc[~transition_matrix.index.isin(detected_redundant_contracts), :]
    
    
    def generate_synthetic_contracts(self, transition_matrix:pd.DataFrame) -> pd.DataFrame:
        m,n = transition_matrix.shape
        target_rank = max(m,n)
        transition_matrix = transition_matrix.copy()
        
        np_transition_matrix = transition_matrix.to_numpy()
        current_rank = np.linalg.matrix_rank(np_transition_matrix)
        if current_rank == target_rank:
            return transition_matrix
        else:
            synthetic_contracts = defaultdict(list)
            for i in range(target_rank-m):
                # compute the most current rank
                updated_rank = np.linalg.matrix_rank(np_transition_matrix)
                linear_dep_candidates = []
                
                for j in range(n):
                    lst = list(range(n))
                    lst.remove(j)
                    tmp_rank = np.linalg.matrix_rank(np_transition_matrix[:,lst])
                    if tmp_rank == updated_rank:
                        # linear dependent
                        linear_dep_candidates.append(j)
                        
                # iteratively test if, adding a further row with a '1' entry for the specific column
                # yields a larger matrix rank
                tmp_matrix = np.concatenate([np_transition_matrix,np.zeros((1,n))],axis=0)
                tmp_rank = updated_rank
                for ld_id in linear_dep_candidates:
                    tmp_matrix[-1, ld_id] = 1
                    test_rank = np.linalg.matrix_rank(tmp_matrix)
                    if test_rank > tmp_rank:
                        tmp_rank = test_rank
                        synthetic_contracts[i].append(ld_id)
                    else:
                        # if the column does not yield a higher matrix rank, revoke the changes
                        tmp_matrix[-1, ld_id] = 0
                # set the new matrix, such that the most current rank can be computed
                np_transition_matrix = tmp_matrix
            
            # get reference contract information to calculate a price for the synthetic contracts
            reference_contract = list(self.contracts.keys())[0]
            reference_mean_shape = self.shape.loc[self.contracts[reference_contract].get_schedule(),:].mean()
            reference_price = self.contracts[reference_contract].get_price()
            
            date_list = self._get_contract_start_end_dates()
            for row_id, column_ids in dict(synthetic_contracts).items():
                _temp_df_shape = None
                for column_id in column_ids:
                    cond1 = self.shape.index >= date_list[column_id]
                    if column_id == n:
                        cond2 = self.shape.index <= date_list[column_id+1]
                    else:
                        cond2 = self.shape.index < date_list[column_id+1]
                    
                    if _temp_df_shape is None:
                        _temp_df_shape = self.shape.loc[(cond1) & (cond2),: ]
                    else:
                        _temp_df_shape = pd.concat([_temp_df_shape, self.shape.loc[(cond1) & (cond2),: ]], axis=0)

                mean_shape = np.mean(_temp_df_shape)
                name = f'Synth_Con_{row_id+1}'
                self._synthetic_contracts[name] = EnergyFutureSpecifications(schedule=None, price=(mean_shape*reference_price/reference_mean_shape)[0], name=name)
                
                _data = np.zeros((n))
                _data[column_ids] = 1
                _df = pd.DataFrame([_data], index=[name], columns=transition_matrix.columns)
                transition_matrix = pd.concat([transition_matrix, _df], axis=0)
            return transition_matrix
                        
                
     
    def shift(self, transition_matrix: pd.DataFrame) -> pd.DataFrame:
        contract_start_and_end_dates = np.array(self._get_contract_start_end_dates())
        contract_schedules = np.unique(list(itertools.chain(*[contract.get_schedule() for contract in self.contracts.values()])))
        
        # starting after the first start date, since we want to get the delivery ticks until the next starting date
        # side='left since we do not want to consider a match as a delivery tick
        delivery_ticks = np.searchsorted(contract_schedules, contract_start_and_end_dates[1:], side='left')
        delivery_ticks_per_period = np.concatenate([np.array([delivery_ticks[0]]),(delivery_ticks[1:] - delivery_ticks[:-1])])
        print(delivery_ticks_per_period)
        date_tpls = list(zip(contract_start_and_end_dates[:-1], contract_start_and_end_dates[1:]))
        # hours_btwn_dates = (pd.Series(contract_start_and_end_dates[1:] - contract_start_and_end_dates[:-1])/pd.Timedelta(hours=timeperiod_fraction)).to_numpy().reshape(1,-1) * delivery_period_units_per_day # 24 only for HPFC Base
        
        transition_matrix = transition_matrix.to_numpy() * delivery_ticks_per_period
        print(np.sum(transition_matrix, axis=1))
        transition_matrix = transition_matrix/np.sum(transition_matrix, axis=1).reshape(-1,1)
        fwd_price_vec = self._get_forward_price_vector()

        fwd_price_noc = np.linalg.inv(transition_matrix) @ fwd_price_vec
        pfc = self.shape.copy()
        print(date_tpls)
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
