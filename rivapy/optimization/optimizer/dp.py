import numpy as np


class DPOptimizer:
    def __init__(self):
        pass

    def backward(self):
        # backward
        value_list = []
        for i, t in enumerate(reversed(range(bs_matrix.shape[0]))):
            if i == 0:
                previous_state_values = np.zeros(shape=(1, bs.number_states))
            else:
                previous_state_values = value_list[i - 1]

            # t_id = np.argwhere(bs_matrix[:, 0]==t).squeeze()
            # temp_array = bs_matrix[t_id,...]

            temp_array = bs_matrix[t, ...]

            temp_array = temp_array[np.argsort(temp_array[:, 0]), ...]

            temp_state_values = previous_state_values[:, np.int32(temp_array[:, -2])].squeeze()

            temp_state_values = temp_state_values + temp_array[:, -1]

            unique_states, idx_start = np.unique(temp_array[:, 0].astype(int), return_index=True)
            value_list.append(np.minimum.reduceat(temp_state_values, idx_start).reshape((1, bs.number_states)))
            # value_list.append(np.min(temp_state_values.reshape(bs.number_states, -1), axis=1).reshape((1, bs.number_states)))

    def forward(self):
        # forward
        forward_value_list = list(reversed(value_list))
        chosen_states = []
        chosen_actions = []
        for t in range(bs_matrix.shape[0]):
            # t_id = np.argwhere(bs_matrix[:, 0]==t).squeeze()
            temp_array = bs_matrix[t, ...]

            if t == 0:
                if bs.start_state is not None:
                    temp_state = bs.start_state

                else:
                    temp_state = np.argmin(forward_value_list[t].squeeze())
                chosen_states.append(temp_state)
            else:
                temp_state = chosen_states[t]  # not t-1 since we already have the initial state in the chosen states list

            temp_array = temp_array[np.argwhere(temp_array[:, 0] == temp_state).squeeze(), ...].reshape((-1, temp_array.shape[-1]))

            if t == len(prices) - 1:
                next_state_value = np.zeros(shape=(1, bs.number_states))
            else:
                next_state_value = forward_value_list[t + 1]

            costs = temp_array[:, -1]
            diff_arr = (next_state_value[:, np.int32(temp_array[:, -2])] + costs).squeeze()

            action_selection = np.argmin(diff_arr).squeeze()

            chosen_actions.append(int(temp_array[action_selection, 1]))
            chosen_states.append(int(temp_array[action_selection, 2]))
