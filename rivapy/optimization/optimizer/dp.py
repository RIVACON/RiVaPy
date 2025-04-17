import numpy as np
from rivapy.optimization.models.dp_models.base_model import DPBaseModel
from typing import List

# TODO
# 1. Backward pass muss auch mit zwischen States umgehen können (Effizienz)
# 2. Forward pass muss auch mit zwischen States umgehen können (Effizienz)
#     2.1. Bei zwischen States im FP muss auch gewährleistet werden, dass keine falschen States erlaubt sind
#     2.2. Zwischen States werden durch Interpolation bewertet.
#     2.3. Bei der Interpolation muss berücksichtigt werden, dass alles gleich bleibt und nur über den Füllstand interpoliert wird.
#         (Beispielsweise muss der Counting State für die Max Discharges in dem Fall konstant gehalten werden)
#     2.4. Klappt das mit dem State mapping oder funktioniert das nur mit den State-Tuples?
#     2.5. Ist dies in rein NumPy umsetzbar oder muss hier numba hinzugezogen werden?


class DPOptimizer:
    def __init__(self, model: DPBaseModel):
        self.model = model
        self.state_transition_matrix: np.ndarray = model.state_mapping(model.get_state_transition_matrix())

        self._value_list: List[np.ndarray] = []
        self._chosen_states: List[int] = []
        self._chosen_actions: List[int] = []

    def backward(self):
        # backward
        self._value_list = []
        end_state = self.model.get_end_state()
        for i, t in enumerate(reversed(range(self.state_transition_matrix.shape[0]))):
            if i == 0:
                if end_state is not None:
                    previous_state_values = np.full((1, self.model.get_states_number()), np.inf)
                    previous_state_values[0, end_state] = 0
                else:
                    previous_state_values = np.zeros(shape=(1, self.model.get_states_number()))
                self._value_list.append(previous_state_values)
                continue
            else:
                previous_state_values = self._value_list[i - 1]

            # t_id = np.argwhere(bs_matrix[:, 0]==t).squeeze()
            # temp_array = bs_matrix[t_id,...]

            temp_array = self.state_transition_matrix[t, ...]

            temp_array = temp_array[np.argsort(temp_array[:, 0]), ...]

            temp_state_values = previous_state_values[:, np.int32(temp_array[:, -2])].squeeze()

            temp_state_values = temp_state_values + temp_array[:, -1]

            unique_states, idx_start = np.unique(temp_array[:, 0].astype(int), return_index=True)
            self._value_list.append(np.minimum.reduceat(temp_state_values, idx_start).reshape((1, self.model.get_states_number())))

    def forward(self):
        # forward
        self._chosen_states = []
        self._chosen_actions = []
        forward_value_list = list(reversed(self._value_list))
        for t in range(self.state_transition_matrix.shape[0] - 1):
            # t_id = np.argwhere(bs_matrix[:, 0]==t).squeeze()
            temp_array = self.state_transition_matrix[t, ...]

            if t == 0:
                start_state = self.model.get_start_state()

                if start_state is not None:
                    if isinstance(start_state, list) or isinstance(start_state, np.ndarray):
                        temp_state = start_state[np.argmin((forward_value_list[t].squeeze())[start_state])]
                    elif isinstance(start_state, int):
                        temp_state = start_state
                    else:
                        raise ValueError(f"Not supported start_state type of type {type(start_state)}. Supported types are int, list and np.ndarray.")
                else:
                    temp_state = int(np.argmin(forward_value_list[t].squeeze()))
                self._chosen_states.append(temp_state)
            else:
                temp_state = self._chosen_states[t]  # not t-1 since we already have the initial state in the chosen states list

            temp_array = temp_array[np.argwhere(temp_array[:, 0] == temp_state).squeeze(), ...].reshape((-1, temp_array.shape[-1]))

            # if t == self.state_transition_matrix.shape[0] - 1:
            #     next_state_value = np.zeros(shape=(1, self.model.get_states_number()))
            # else:
            next_state_value = forward_value_list[t + 1]

            costs = temp_array[:, -1]
            diff_arr = (next_state_value[:, np.int32(temp_array[:, -2])] + costs).squeeze()

            action_selection = np.argmin(diff_arr).squeeze()

            self._chosen_actions.append(int(temp_array[action_selection, 1]))
            self._chosen_states.append(int(temp_array[action_selection, 2]))
