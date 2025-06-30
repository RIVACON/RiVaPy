import itertools
import pandas as pd
import numpy as np
import datetime as dt
import rivapy.tools.interfaces as interfaces
import rivapy.tools._validators as validators
from rivapy.instruments import EnergyFutureSpecifications
from typing import Dict, Set, List, Any
from collections import defaultdict


def validate_class_input(func):
    def validate_wrapper(self, shape: pd.DataFrame, contracts: List[EnergyFutureSpecifications]):
        validators._check_pandas_index_for_datetime(dataframe=shape)
        # if isinstance(shape, pd.DataFrame):
        #     if not isinstance(shape.index, pd.DatetimeIndex):
        #         raise TypeError("The index of the shape DataFrame is not of type pd.DatetimeIndex!")
        # else:
        #     raise TypeError("The shape argument is not of type pd.DataFrame!")

        contract_scheduled_dates = set(np.concatenate([contract.get_schedule() for contract in contracts]))
        expected_dates = set(shape.index)
        date_diff = expected_dates - contract_scheduled_dates
        if len(date_diff) != 0:
            raise ValueError("The contract dates do not cover each date provided by the shape DataFrame!")
        func(self, shape, contracts)

    return validate_wrapper


class PFCShifter(interfaces.FactoryObject):
    """A shifting methodology for PFC shapes. This class gets a PFC shape as an input and shifts it in such a way, that the resulting PFC contains the future prices defined in the ``contracts`` dictionary.
    We follow the methodology described here: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2706366

    Args:
        shape (pd.DataFrame): PFC shape, where the ``DataFrame`` index are ``datetime`` objects.
        contracts (Dict[str, EnergyFutureSpecifications]): Dictionary containing the future contracts (``EnergyFutureSpecifications`` objects)

    Usage:

    .. highlight:: python
    .. code-block:: python

            # iterative usage
            pfc_shifter = PFCShifter(shape=shape, contracts=contracts)
            transition_matrix = pfc_shifter.generate_transition_matrix()
            transition_matrix = pfc_shifter.detect_redundant_contracts(transition_matrix)
            transition_matrix = pfc_shifter.generate_synthetic_contracts(transition_matrix)
            pfc = pfc_shifter.shift(transition_matrix)

            # direct call
            pfc_shifter = PFCShifter(shape=shape, contracts=contracts)
            pfc = pfc_shifter.compute()

    """

    @validate_class_input
    def __init__(self, shape: pd.DataFrame, contracts: List[EnergyFutureSpecifications]) -> None:
        self.shape = shape
        self.contracts = {contract.name: contract for contract in contracts}
        self._redundant_contracts: Dict[str, EnergyFutureSpecifications] = {}
        self._synthetic_contracts: Dict[str, EnergyFutureSpecifications] = {}

    def _get_contract_start_end_dates(self) -> List[dt.datetime]:
        """Returns a sorted list combined of all start and end ``datetime`` objects for each contract.

        Returns:
            List[dt.datetime]: Sorted list of start and end dates
        """
        dates = set()
        for contract_schedule in self.contracts.values():
            dates.update(contract_schedule.get_start_end())
        return sorted(list(dates))

    def _get_forward_price_vector(self) -> np.ndarray:
        """Returns a vector containing the forward/future prices of the contracts and potentiall synthetic contracts.

        Returns:
            np.ndarray: Numpy array of forward/future prices
        """
        _dict = {**self.contracts, **self._synthetic_contracts}
        return np.array([contract.get_price() for contract in _dict.values()]).reshape(-1, 1)

    def compute(self) -> pd.DataFrame:
        """Compute method to directly call all the individual steps involved for the shifting

        Returns:
            pd.DataFrame: Shifted PFC shape
        """
        transition_matrix = self.generate_transition_matrix()
        transition_matrix = self.detect_redundant_contracts(transition_matrix=transition_matrix)
        transition_matrix = self.generate_synthetic_contracts(transition_matrix=transition_matrix)
        return self.shift(transition_matrix=transition_matrix)

    def generate_transition_matrix(self) -> pd.DataFrame:
        """The transition matrix is the basis of the shifting algorithm. This method generates a (n x m) matrix with zero and one entries, where n is the number of contracts and m are start and end dates for the delivery periods.
        Hence, the matrix row vectors indicate the delivery periods of each contract. Note that the latest delivery end date is not displayed in the transition matrix.


        Returns:
            pd.DataFrame: Transition matrix containing zeros and ones indicating delivery periods of individual contracts.
        """
        contract_start_and_end_dates = np.array(self._get_contract_start_end_dates())

        transition_df = pd.DataFrame(
            data=np.zeros((len(self.contracts.keys()), len(contract_start_and_end_dates))),
            index=list(self.contracts.keys()),
            columns=contract_start_and_end_dates,
        )

        for contract_name, contract_schedule in self.contracts.items():
            idx = contract_start_and_end_dates.searchsorted(list(contract_schedule.get_start_end()), "right") - 1

            if idx[0] == idx[1]:
                transition_df.iloc[transition_df.index == contract_name, idx[0]] = 1
            else:
                transition_df.iloc[transition_df.index == contract_name, idx[0] : idx[1]] = 1

        return transition_df.iloc[:, :-1]  # drop the last column for the transition matrix

    def detect_redundant_contracts(self, transition_matrix: pd.DataFrame) -> pd.DataFrame:
        """In order to obtain an invertable matrix, the matrix must be of full rank. Linear dependent contracts will yield linear dependent row vectors.
        This is the case if e.g. a Cal Base and all four quarter contracts are provided. This method finds all redundant (linear dependent) contracts and
        omits the last found linear dependent contract in order to make sure that the row vectors are linearly independent.

        Args:
            transition_matrix (pd.DataFrame): Transition matrix generated by the ``generate_transition_matrix`` method.

        Returns:
            pd.DataFrame: Transition matrix without linearly dependent row vectors.
        """
        potential_redundant_contracts = []
        np_transition_matrix = transition_matrix.to_numpy()
        for i in range(len(transition_matrix)):
            lst = list(range(len(transition_matrix)))
            lst.remove(i)
            if np.linalg.matrix_rank(np_transition_matrix[lst, :]) == np.linalg.matrix_rank(np_transition_matrix):
                potential_redundant_contracts.append(i)

        base_matrix = np.delete(np_transition_matrix, potential_redundant_contracts, axis=0)

        detected_redundant_contracts = []
        if len(potential_redundant_contracts) != 0:
            for contract_idx in potential_redundant_contracts:
                _temp_matrix = np.concatenate([base_matrix, np_transition_matrix[contract_idx, :].reshape(1, -1)], axis=0)
                # in case all contracts are potentially redundant
                if base_matrix.shape[0] == 0:
                    ref_rank = 0
                else:
                    ref_rank = np.linalg.matrix_rank(base_matrix)
                if np.linalg.matrix_rank(_temp_matrix) > ref_rank:
                    base_matrix = _temp_matrix
                else:
                    print(f"Found redundant contract: {transition_matrix.index[contract_idx]}")
                    detected_redundant_contracts.append(transition_matrix.index[contract_idx])

        # update the contracts dictionary, but still keep the information about the redundant contracts
        self._redundant_contracts = {}
        for contract in detected_redundant_contracts:
            self._redundant_contracts[contract] = self.contracts[contract]
            del self.contracts[contract]  # <- keep an eye on that line
        return transition_matrix.loc[~transition_matrix.index.isin(detected_redundant_contracts), :]

    def generate_synthetic_contracts(self, transition_matrix: pd.DataFrame) -> pd.DataFrame:
        """In order to fulfill the requirement of an invertable transition matrix, not only the row vectors but also the
        column vectors must generate a basis. In cases where m > n, we need to additionally generate synthetic contracts.
        The delivery period for the synthetic contracts are chosen in such a way that the column vectors become linearly independent.
        The forward price for each synthetic contract is computed based on the rations of the average shape values over the corresponding delivery period of the synthetic contract and a reference contract.
        The shape ratio is multiplied with the forward price of the reference contract in order to obtain a forward price for the synthetic contract.
        The reference contract is implemented to be always the first contract in the ``contracts`` dictionary.

        Args:
            transition_matrix (pd.DataFrame): Transition matrix generated by the ``detect_redundant_contracts`` method.

        Returns:
            pd.DataFrame: Full rank transition matrix
        """
        m, n = transition_matrix.shape
        target_rank = max(m, n)
        transition_matrix = transition_matrix.copy()

        np_transition_matrix = transition_matrix.to_numpy()
        current_rank = np.linalg.matrix_rank(np_transition_matrix)
        if current_rank == target_rank:
            return transition_matrix
        else:
            synthetic_contracts = defaultdict(list)
            for i in range(target_rank - m):
                # compute the most current rank
                updated_rank = np.linalg.matrix_rank(np_transition_matrix)
                linear_dep_candidates = []

                for j in range(n):
                    lst = list(range(n))
                    lst.remove(j)
                    tmp_rank = np.linalg.matrix_rank(np_transition_matrix[:, lst])
                    if tmp_rank == updated_rank:
                        # linear dependent
                        linear_dep_candidates.append(j)

                # iteratively test if, adding a further row with a '1' entry for the specific column
                # yields a larger matrix rank
                tmp_matrix = np.concatenate([np_transition_matrix, np.zeros((1, n))], axis=0)
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
            reference_mean_shape = self.shape.loc[self.contracts[reference_contract].get_schedule(), :].mean(axis=0)
            reference_price = self.contracts[reference_contract].get_price()

            date_list = self._get_contract_start_end_dates()
            for row_id, column_ids in dict(synthetic_contracts).items():
                _temp_df_shape = None
                for column_id in column_ids:
                    cond1 = self.shape.index >= date_list[column_id]
                    if column_id == n:
                        cond2 = self.shape.index <= date_list[column_id + 1]
                    else:
                        cond2 = self.shape.index < date_list[column_id + 1]

                    if _temp_df_shape is None:
                        _temp_df_shape = self.shape.loc[(cond1) & (cond2), :]
                    else:
                        _temp_df_shape = pd.concat([_temp_df_shape, self.shape.loc[(cond1) & (cond2), :]], axis=0)

                mean_shape = np.mean(_temp_df_shape, axis=0)
                name = f"Synth_Contr_{row_id+1}"
                self._synthetic_contracts[name] = EnergyFutureSpecifications(
                    schedule=None, price=(mean_shape * reference_price / reference_mean_shape).iloc[0], name=name
                )

                _data = np.zeros((n))
                _data[column_ids] = 1
                _df = pd.DataFrame([_data], index=[name], columns=transition_matrix.columns)
                transition_matrix = pd.concat([transition_matrix, _df], axis=0)
            return transition_matrix

    def shift(self, transition_matrix: pd.DataFrame) -> pd.DataFrame:
        """This method is the final step in the shifting algorithm. The transition matrix is inversed and multiplied with the forward price vector to obtain a non overlapping forward price vector.

        .. math::

            f^{no} = T^{-1}\cdot f

        where:\n
        :math:`f^{no}` is the Non-overlapping forward price vector\n
        :math:`T` is the Transition matrix\n
        :math:`f` is the Forward price vector\n

        Afterwards the PFC :math:`S(t)` is obtained from the shape :math:`s(t)` by the follwoing formular:

        .. math::
            S(t) = s(t)\cdot \\frac{\sum_{u=T_s}^{T_e} f^{no}(u)}{\sum_{u=T_s}^{T_e} s(u)}

        with :math:`T_s` and :math:`T_e` being the start and end dates of the individual delivery periods.

        Args:
            transition_matrix (pd.DataFrame): Full rank transition matrix generated by the ``generate_synthetic_contracts`` method

        Returns:
            pd.DataFrame: Shifted shape.
        """
        contract_start_and_end_dates = np.array(self._get_contract_start_end_dates())
        contract_schedules = np.unique(list(itertools.chain(*[contract.get_schedule() for contract in self.contracts.values()])))

        # starting after the first start date, since we want to get the delivery ticks until the next starting date
        # side='left since we do not want to consider a match as a delivery tick
        delivery_ticks = np.searchsorted(contract_schedules, contract_start_and_end_dates[1:], side="left")
        delivery_ticks_per_period = np.concatenate([np.array([delivery_ticks[0]]), (delivery_ticks[1:] - delivery_ticks[:-1])])

        date_tpls = list(zip(contract_start_and_end_dates[:-1], contract_start_and_end_dates[1:]))

        transition_matrix = transition_matrix.to_numpy() * delivery_ticks_per_period
        transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1).reshape(-1, 1)
        fwd_price_vec = self._get_forward_price_vector()

        fwd_price_noc = np.linalg.inv(transition_matrix) @ fwd_price_vec
        pfc = self.shape.copy()
        # print(date_tpls)
        for i, date_tpl in enumerate(date_tpls):
            if i == len(date_tpls) - 1:
                row_filter = (pfc.index >= date_tpl[0]) & (pfc.index <= date_tpl[1])
            else:
                row_filter = (pfc.index >= date_tpl[0]) & (pfc.index < date_tpl[1])

            pfc.iloc[row_filter, 0] = pfc.iloc[row_filter, 0] / np.sum(pfc.iloc[row_filter, 0]) * len(pfc.iloc[row_filter, 0]) * fwd_price_noc[i, 0]
        return pfc

    def _to_dict(self) -> dict:
        return {**{"shape": self.shape}, **{"contracts": [v.to_dict() for v in self.contracts.values()]}}
