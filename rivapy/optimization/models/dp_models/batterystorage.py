import time
import abc
import numpy as np
import pandas as pd
from rivapy.optimization.models.dp_models.base_model import DPBaseModel
from typing import Optional, Union, List, Dict


class _SimpleBatteryStorage(abc.ABC):
    def __init__(
        self, states: np.ndarray, actions: np.ndarray, prices: np.ndarray, start_level: Optional[int] = None, end_level: Optional[int] = None
    ):
        self.states = states.reshape(-1, 1)
        self.actions = actions.reshape(1, -1)
        self.prices = prices
        self.start_level = start_level
        self.end_level = end_level

        self.number_states = self.states.shape[0]

    @abc.abstractmethod
    def get_state_transition_matrix(self):
        state_base = (np.ones(shape=(self.states.shape[-1] * self.actions.shape[-1])) * self.states).reshape(-1, 1)

        _actions = np.resize(self.actions, (state_base.shape[0], 1))

        next_state = state_base + _actions
        timestep_state_transition = np.concat([state_base, _actions, next_state], axis=1)

        actions_clmn = timestep_state_transition[:, 1][..., np.newaxis]
        prices = self.prices.T
        costs = (prices * actions_clmn).T.reshape(-1, 1)

        state_transition = np.resize(timestep_state_transition, (costs.shape[0], timestep_state_transition.shape[1]))

        discharge_violation_ids = np.argwhere(state_transition[:, 2] < np.min(self.states)).squeeze()
        charge_violation_ids = np.argwhere(state_transition[:, 2] > np.max(self.states)).squeeze()

        state_transition[discharge_violation_ids, 2] = np.min(self.states)
        state_transition[charge_violation_ids, 2] = np.max(self.states)

        _state_transition = np.concat([state_transition, costs], axis=1)
        state_transition = np.delete(_state_transition, np.unique(np.concat([discharge_violation_ids, charge_violation_ids])), axis=0)

        return state_transition.reshape((len(self.prices), state_transition.shape[0] // len(self.prices), state_transition.shape[-1]))

    @abc.abstractmethod
    def get_start_state(self) -> Union[List[int], int]:
        return self.start_level

    @abc.abstractmethod
    def get_end_state(self) -> Union[List[int], int]:
        return self.end_level

    @abc.abstractmethod
    def get_state_number(self) -> int:
        return self.number_states

    @abc.abstractmethod
    def create_state_mapping(self, state_transition_matrix):
        return state_transition_matrix

    @abc.abstractmethod
    def reverse_mapping(self, mapped_states: Union[List, np.ndarray]):
        return mapped_states


class _ChargeRestrictedBatteryStorage(_SimpleBatteryStorage):
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        prices: np.ndarray,
        max_charges: int,
        start_level: Optional[int] = None,
        end_level: Optional[int] = None,
        start_charges: Optional[int] = None,
        end_charges: Optional[int] = None,
    ):
        super().__init__(states, actions, prices, start_level, end_level)
        self.max_charges = max_charges
        self.start_charges = start_charges
        self.end_charges = end_charges
        self.number_states = self.states.shape[0] * (self.max_charges + 1)
        self.state_mapping_dict = None

    def get_state_transition_matrix(self):
        skeleton_array = np.ones(shape=(self.states.shape[-1] * (self.max_charges + 1) * self.actions.shape[-1]))

        state_base = (skeleton_array * self.states).reshape(-1, 1)

        charges_base = np.tile(np.arange(0, self.max_charges + 1).reshape(-1, 1), (self.states.shape[0], self.actions.shape[-1])).reshape(-1, 1)
        _actions = np.resize(self.actions, (charges_base.shape[0], 1))

        next_state = np.concat([state_base + _actions, charges_base + np.abs(_actions)], axis=1)
        timestep_state_transition = np.concat([state_base, charges_base, _actions, next_state], axis=1)

        actions_clmn_id = 2

        actions_clmn = timestep_state_transition[:, actions_clmn_id][..., np.newaxis]
        prices = self.prices.T
        costs = (prices * actions_clmn).T.reshape(-1, 1)

        state_transition = np.resize(timestep_state_transition, (costs.shape[0], timestep_state_transition.shape[1]))

        discharge_violation_ids = np.argwhere(state_transition[:, actions_clmn_id + 1] < np.min(self.states)).squeeze()
        charge_violation_ids = np.argwhere(state_transition[:, actions_clmn_id + 1] > np.max(self.states)).squeeze()

        charges_violation_ids = np.argwhere(state_transition[:, actions_clmn_id + 2] > self.max_charges).squeeze()

        state_transition[discharge_violation_ids, actions_clmn_id + 1] = np.min(self.states)
        state_transition[charge_violation_ids, actions_clmn_id + 1] = np.max(self.states)
        state_transition[charges_violation_ids, actions_clmn_id + 2] = self.max_charges

        _state_transition = np.concat([state_transition, costs], axis=1)
        state_transition = np.delete(
            _state_transition, np.unique(np.concat([discharge_violation_ids, charge_violation_ids, charges_violation_ids])), axis=0
        )
        return state_transition.reshape((len(self.prices), state_transition.shape[0] // len(self.prices), state_transition.shape[-1]))

    def create_state_mapping(self, state_transition_matrix: np.ndarray) -> np.ndarray:
        # Extract current and next state pairs
        _state_transition_matrix = state_transition_matrix.reshape(-1, state_transition_matrix.shape[-1])

        current_states = _state_transition_matrix[:, [0, 1]]
        next_states = _state_transition_matrix[:, [3, 4]]

        # Combine both for unified unique indexing
        all_states = np.vstack((current_states, next_states))

        # np.unique is slower than using pandas here
        df_states = pd.DataFrame(all_states).astype(int)
        unique_states_df = df_states.drop_duplicates().reset_index(drop=True).reset_index()
        inverse_indices = df_states.merge(unique_states_df, how="left", on=[0, 1], sort=False)["index"].to_numpy()

        # self.state_mapping = {tuple(state): idx for idx, state in enumerate(unique_states)}
        self.state_mapping_dict = pd.Series(unique_states_df["index"].values, index=list(zip(unique_states_df[0], unique_states_df[1]))).to_dict()

        # Split back the indices
        current_state_ids = inverse_indices[: len(current_states)]
        next_state_ids = inverse_indices[len(current_states) :]

        # Construct the final mapped matrix
        return np.column_stack((current_state_ids, _state_transition_matrix[:, 2], next_state_ids, _state_transition_matrix[:, 5:])).reshape(
            (state_transition_matrix.shape[0], state_transition_matrix.shape[1], 4)
        )

    def __return_states(self, level: Optional[int], charge_level: Optional[int]) -> Union[List[int], int]:
        if self.state_mapping_dict is None:
            raise ValueError("No mapping dictionary initialized. Consider running create_state_mapping first")

        if (level is None) and (charge_level is None):
            return None

        if level is None:
            return [self.state_mapping_dict[(state, charge_level)] for state in self.states.squeeze()]

        if charge_level is None:
            return [self.state_mapping_dict[(level, charge)] for charge in np.arange(0, self.max_charges + 1)]

        return self.state_mapping_dict[(level, charge_level)]

    def get_start_state(self) -> Union[List[int], int]:
        return self.__return_states(self.start_level, self.start_charges)

    def get_end_state(self) -> Union[List[int], int]:
        return self.__return_states(self.end_level, self.end_charges)

    def get_state_number(self) -> int:
        return self.number_states

    def reverse_mapping(self, mapped_states):
        reverse_mapping_df = pd.concat(
            [pd.Series(self.state_mapping_dict.keys(), name="States"), pd.Series(self.state_mapping_dict.values(), name="Mapped State")], axis=1
        )
        return reverse_mapping_df.iloc[mapped_states, :]["States"].to_list()


class BatteryStorage(DPBaseModel):
    # TODO: implement check for unrealistic start and end point configuration
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        prices: np.ndarray,  # TODO: change to PFC object
        max_charges: Optional[int],
        start_level: Optional[int] = None,
        end_level: Optional[int] = None,
        start_charges: Optional[int] = None,
        end_charges: Optional[int] = None,
    ):
        self.states = states
        self.actions = actions
        self.prices = prices
        self.start_level = start_level
        self.end_level = end_level
        self.max_charges = max_charges
        self.start_charges = start_charges
        self.end_charges = end_charges

        if self.max_charges is not None:
            self._internal_model = _ChargeRestrictedBatteryStorage(
                states=states,
                actions=actions,
                prices=prices,
                max_charges=max_charges,
                start_level=start_level,
                end_level=end_level,
                start_charges=start_charges,
                end_charges=end_charges,
            )
        else:
            self._internal_model = _SimpleBatteryStorage(states=states, actions=actions, prices=prices, start_level=start_level, end_level=end_level)

    def get_state_transition_matrix(self) -> np.ndarray:
        return self._internal_model.get_state_transition_matrix()

    def state_mapping(self, state_transition_matrix: np.ndarray):
        return self._internal_model.create_state_mapping(state_transition_matrix=state_transition_matrix)

    def get_states_number(self) -> int:
        return self._internal_model.get_state_number()

    def get_start_state(self):
        return self._internal_model.get_start_state()

    def get_end_state(self):
        return self._internal_model.get_end_state()

    def create_output(self, actions, states, values):
        return super().create_output(actions, states, values)
