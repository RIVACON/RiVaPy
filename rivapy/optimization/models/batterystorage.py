import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from numba import njit
from numba import float32 as numba_float32, int32 as numba_int32
from rivapy.marketdata import EnergyPriceForwardCurve
from typing import Union, Optional, Literal, Dict


@njit
def _value_wrapper(
    state: float,
    charge: int,
    action: float,
    max_state: float,
    min_state: float,
    prices: numba_float32[:],  # np.ndarray,
    t: int,
    states: numba_float32[:],  # np.ndarray,
    value_matrix: numba_float32[:, :, :],  # np.ndarray,
    eff_out: float,
    max_capacity: float,
    mode: int,
    penalty: float,
) -> float:
    if state < min_state or state > max_state:
        value = penalty
        return value
    if mode == 1:
        reward = (-1) * action / 100.0 * prices[t - 1] * max_capacity

    elif mode == -1:
        reward = (-1) * action / 100.0 * eff_out * prices[t - 1] * max_capacity

    elif mode == 0:
        reward = 0.0

    value = __value(state=state, charge=charge, t=t, states=states, value_matrix=value_matrix, reward=reward, mode=mode, penalty=penalty)
    return value


@njit
def __value(
    state: float,
    charge: int,
    t: int,
    states: numba_float32[:],  # np.ndarray,
    value_matrix: numba_float32[:, :, :],  # np.ndarray,
    reward: float,
    mode: int,
    penalty: float,
) -> float:

    idx = np.searchsorted(states, state)
    mode_value = np.abs(mode)
    next_charge = charge * (1 - mode_value) + (charge + 1) * mode_value
    # if mode == 0:
    #     next_charge = charge
    # else:
    #     next_charge = charge + 1

    if np.abs(states[idx] - state) < 10 ** (-8):
        value_next_state = value_matrix[t, idx, next_charge]

    # elif np.abs(states[idx-1] - state) < 10**(-8):

    # if state_mapping.get(state, -1) != -1:
    #     value_next_state = value_matrix[t, state_mapping[state], next_charge]
    else:
        # ceil = state_mapping[states[idx]]
        # floor = state_mapping[states[idx - 1]]
        ceil = idx
        floor = idx - 1

        if np.abs(value_matrix[t, ceil, next_charge] - penalty) < 10 ** (-8) or np.abs(value_matrix[t, floor, next_charge] - penalty) < 10 ** (-8):
            # value_next_state = (-1) * 10**(12)
            # reward = (-1) * 10**(12)
            return penalty
        else:
            m = (value_matrix[t, ceil, next_charge] - value_matrix[t, floor, next_charge]) / (states[idx] - states[idx - 1])
            b = value_matrix[t, floor, next_charge] - m * states[idx - 1]
            value_next_state = m * state + b

    return value_next_state + reward


@njit
def backward(
    eff_in: float,
    eff_out: float,
    max_capacity: float,
    states: numba_float32[:],  # np.ndarray,
    actions: numba_float32[:],  # np.ndarray,
    prices: numba_float32[:],  # np.ndarray,
    max_charges: int,
    end_state: Optional[float] = None,
    penalty: float = -1e12,
):
    T = len(prices)

    max_state = np.max(states)
    min_state = np.min(states)
    max_action = np.max(actions)
    min_action = np.min(actions)

    max_state_id = len(states) - 1

    if end_state is None:
        value_matrix = np.zeros((T, len(states), max_charges + 1), dtype=np.float32)
    else:
        end_state_id = np.searchsorted(states, end_state)
        value_matrix = np.ones((T, len(states), max_charges + 1), dtype=np.float32) * penalty
        value_matrix[:, end_state_id, :] = 0.0

    for i in range(1, T):
        t = T - i - 1
        for state_id in range(len(states)):
            state = states[state_id]
            for charge in range(max_charges + 1):

                if charge == max_charges:
                    value_matrix[t, state_id, charge] = value_matrix[t + 1, state_id, charge]
                    continue

                min_value = penalty
                for action_id in range(len(actions)):
                    action = actions[action_id]
                    if action > 0:
                        next_state = state + eff_in * action
                        mode = 1

                    elif action < 0:
                        next_state = state + action
                        mode = -1

                    else:
                        next_state = state
                        mode = 0

                    value = _value_wrapper(
                        state=next_state,
                        charge=charge,
                        action=action,
                        max_state=max_state,
                        min_state=min_state,
                        prices=prices,
                        t=t + 1,
                        states=states,
                        value_matrix=value_matrix,
                        eff_out=eff_out,
                        max_capacity=max_capacity,
                        mode=mode,
                        penalty=penalty,
                    )
                    if min_value < value:
                        min_value = value

                # Check if continuous action could fill the storage
                if (max_state - state) <= (max_action * eff_in):
                    value = value_matrix[t + 1, max_state_id, charge + 1] - (max_state - state) / 100.0 * (1 / eff_in) * prices[t] * max_capacity

                    if min_value < value:
                        min_value = value

                # Check if continuous action could empty the storage
                if np.abs((min_state - state)) <= np.abs(min_action):
                    value = value_matrix[t + 1, 0, charge + 1] + np.abs((min_state - state)) / 100.0 * eff_out * prices[t] * max_capacity

                    if min_value < value:
                        min_value = value

                # check if reaching the end state is possible and optimal
                if end_state is not None:
                    if end_state > state:
                        if (end_state - state) <= (max_action * eff_in):
                            value = (
                                value_matrix[t + 1, end_state_id, charge + 1] - (end_state - state) / 100.0 * (1 / eff_in) * prices[t] * max_capacity
                            )

                        if min_value < value:
                            min_value = value

                    elif end_state < state:
                        if np.abs((end_state - state)) <= np.abs(min_action):
                            value = (
                                value_matrix[t + 1, end_state_id, charge + 1]
                                + np.abs((end_state - state)) / 100.0 * eff_out * prices[t] * max_capacity
                            )

                        if min_value < value:
                            min_value = value

                value_matrix[t, state_id, charge] = min_value

    return value_matrix


@njit
def forward(
    eff_in: float,
    eff_out: float,
    max_capacity: float,
    value_matrix: np.ndarray,
    states: numba_float32[:],  # np.ndarray,
    actions: numba_float32[:],  # np.ndarray,
    prices: numba_float32[:],  # np.ndarray,
    max_charges: int,
    start_state: Optional[float] = None,
    start_charges: Optional[int] = None,
    end_state: Optional[float] = None,
    penalty: float = -1e12,
):
    T = len(prices)

    state_choices = np.zeros(T, dtype=np.float32)
    charges_choices = np.zeros(T, dtype=np.int32)
    objective = np.zeros(T - 1, dtype=np.float32)
    action_choices = np.zeros(T - 1, dtype=np.float32)

    max_state = np.max(states)
    min_state = np.min(states)
    max_action = np.max(actions)
    min_action = np.min(actions)

    max_state_id = len(states) - 1

    prev_state = None
    prev_charge = None

    if end_state is not None:
        end_state_id = np.searchsorted(states, end_state)

    for t in range(T):
        value_matrix_slice = value_matrix[t]
        if t == 0:
            if start_state is None and start_charges is None:
                # min_index = np.argmin(value_matrix_slice)
                # prev_state, prev_charge = np.unravel_index(min_index, value_matrix_slice.shape)
                min_index = np.argmax(value_matrix_slice)
                rows, cols = value_matrix_slice.shape
                prev_state = min_index // cols  # row index
                prev_charge = min_index % cols

                state_choices[t] = states[prev_state]
                charges_choices[t] = prev_charge

            elif start_charges is None:
                idx = np.searchsorted(states, start_state)
                prev_state = start_state
                prev_charge = np.argmax(value_matrix_slice[idx, :])

                state_choices[t] = prev_state
                charges_choices[t] = prev_charge

            elif start_state is None:
                prev_state_idx = np.argmax(value_matrix_slice[:, start_charges])
                prev_charge = start_charges

                state_choices[t] = states[prev_state_idx]
                charges_choices[t] = prev_charge
            else:
                prev_state = start_state
                prev_charge = start_charges

                state_choices[t] = prev_state
                charges_choices[t] = prev_charge

            continue

        prev_state = state_choices[t - 1]
        prev_charge = int(charges_choices[t - 1])

        if prev_charge == max_charges:
            action_choices[t - 1] = 0
            state_choices[t] = prev_state
            charges_choices[t] = prev_charge
            continue

        best_value = penalty

        chosen_state = None
        chosen_action = None

        for action_id in range(len(actions)):
            action = actions[action_id]
            if action > 0:
                next_state = prev_state + eff_in * action
                next_charge = prev_charge + 1
                mode = 1

            elif action < 0:
                next_state = prev_state + action
                next_charge = prev_charge + 1
                mode = -1

            else:
                next_state = prev_state
                next_charge = prev_charge
                mode = 0

            value = _value_wrapper(
                state=next_state,
                charge=prev_charge,
                action=action,
                max_state=max_state,
                min_state=min_state,
                prices=prices,
                t=t,
                states=states,
                value_matrix=value_matrix,
                eff_out=eff_out,
                max_capacity=max_capacity,
                mode=mode,
                penalty=penalty,
            )

            if best_value < value:
                best_value = value
                chosen_state = next_state
                chosen_action = action
                chosen_charge = next_charge

        # Check if continuous action could fill the storage
        if (max_state - prev_state) <= (max_action * eff_in):
            value = value_matrix[t, max_state_id, prev_charge + 1] - (max_state - prev_state) / 100.0 * (1 / eff_in) * prices[t - 1] * max_capacity

            next_state = max_state
            action = (max_state - prev_state) * (1 / eff_in)
            next_charge = prev_charge + 1
            if best_value < value:
                best_value = value
                chosen_state = next_state
                chosen_action = action
                chosen_charge = next_charge

        # Check if continuous action could empty the storage
        if np.abs((min_state - prev_state)) <= np.abs(min_action):
            value = value_matrix[t, 0, prev_charge + 1] + np.abs((min_state - prev_state)) / 100.0 * eff_out * prices[t - 1] * max_capacity

            next_state = min_state
            action = (-1) * np.abs((min_state - prev_state))
            next_charge = prev_charge + 1
            if best_value < value:
                best_value = value
                chosen_state = next_state
                chosen_action = action
                chosen_charge = next_charge

        # check if reaching the end state is possible and optimal
        if end_state is not None:
            if end_state > prev_state:
                if (end_state - prev_state) <= (max_action * eff_in):
                    value = (
                        value_matrix[t, end_state_id, prev_charge + 1]
                        - (end_state - prev_state) / 100.0 * (1 / eff_in) * prices[t - 1] * max_capacity
                    )

                    next_state = end_state
                    action = (end_state - prev_state) * (1 / eff_in)
                    next_charge = prev_charge + 1
                    if best_value < value:
                        best_value = value
                        chosen_state = next_state
                        chosen_action = action
                        chosen_charge = next_charge

            elif end_state < prev_state:
                if np.abs((end_state - prev_state)) <= np.abs(min_action):
                    value = (
                        value_matrix[t, end_state_id, prev_charge + 1]
                        + np.abs((end_state - prev_state)) / 100.0 * eff_out * prices[t - 1] * max_capacity
                    )

                    next_state = end_state
                    action = end_state - prev_state
                    next_charge = prev_charge + 1
                    if best_value < value:
                        best_value = value
                        chosen_state = next_state
                        chosen_action = action
                        chosen_charge = next_charge

        state_choices[t] = chosen_state
        charges_choices[t] = chosen_charge
        action_choices[t - 1] = chosen_action

        if chosen_action > 0:
            objective[t - 1] = chosen_action / 100.0 * prices[t - 1] * (-1) * max_capacity
        else:
            objective[t - 1] = chosen_action / 100.0 * eff_out * prices[t - 1] * (-1) * max_capacity

        # print(print_value)
    return state_choices, charges_choices, action_choices, objective


class BatteryStorage:

    def __init__(
        self,
        eff_in: float,
        eff_out: float,
        max_capacity: float,
        pfc: EnergyPriceForwardCurve,
        max_charges: int,
        states: np.ndarray,
        actions: np.ndarray,
        start_state: Optional[float] = None,
        start_charges: Optional[int] = None,
        end_state: Optional[float] = None,
        start_date: Optional[dt.datetime] = None,
        end_date: Optional[dt.datetime] = None,
        precompile: bool = True,
        penalty: float = -1e12,
    ):
        self._eff_in = eff_in
        self._eff_out = eff_out
        self._max_capacity = max_capacity

        self._states = states
        self._actions = actions

        self._pfc = pfc

        self.pfc = self._pfc.get_pfc()

        if (start_date is None) and (end_date is not None):
            filtered_df = self.pfc.iloc[self.pfc.index < end_date, 0]

        elif (start_date is not None) and (end_date is None):
            filtered_df = self.pfc.iloc[start_date <= self.pfc.index, 0]

        elif (start_date is not None) and (end_date is not None):
            filtered_df = self.pfc.iloc[(start_date <= self.pfc.index) & (self.pfc.index < end_date), 0]

        else:
            filtered_df = self.pfc.iloc[:, 0]

        self.prices = filtered_df.values
        self.datetimes = list(filtered_df.index)

        self._max_charges = max_charges

        self._start_state = start_state

        self._start_charges = start_charges

        self._end_state = end_state

        self._penalty = penalty

        if precompile is True:
            self.__precompile()

        self.__optimized: bool = False

    def __precompile(self):
        def state_check(state: Optional[Union[float, int]], value: Union[float, int]):
            if state is None:
                return None
            else:
                return value

        eff_in = self._eff_in
        eff_out = self._eff_out
        states = np.array([0, 100], dtype=np.float32)
        actions = np.array([-100, 0, 100], dtype=np.float32)
        prices = np.array([1.0, 1.0], dtype=np.float32)
        max_charges = 1
        max_capacity = 100.0

        start_state = state_check(self._start_state, value=0.0)
        start_charges = state_check(self._start_charges, value=0)
        end_state = state_check(self._end_state, 0.0)

        value_matrix = backward(
            eff_in,
            eff_out,
            max_capacity,
            states,
            actions,
            prices,
            max_charges,
            end_state=end_state,
            penalty=self._penalty,
        )
        _ = forward(
            eff_in,
            eff_out,
            max_capacity,
            value_matrix,
            states,
            actions,
            prices,
            max_charges,
            start_state=start_state,
            start_charges=start_charges,
            end_state=end_state,
            penalty=self._penalty,
        )

    def optimize(self):
        value_matrix = backward(
            eff_in=self._eff_in,
            eff_out=self._eff_out,
            max_capacity=self._max_capacity,
            states=self._states,
            actions=self._actions,
            prices=self.prices,
            max_charges=self._max_charges,
            end_state=self._end_state,
            penalty=self._penalty,
        )

        self._state_choices, self._charges_choices, self._action_choices, self._objective = forward(
            eff_in=self._eff_in,
            eff_out=self._eff_out,
            max_capacity=self._max_capacity,
            value_matrix=value_matrix,
            states=self._states,
            actions=self._actions,
            prices=self.prices,
            max_charges=self._max_charges,
            start_state=self._start_state,
            start_charges=self._start_charges,
            end_state=self._end_state,
            penalty=self._penalty,
        )

        self.__optimized = True

    def create_output(self) -> pd.DataFrame:
        if not self.__optimized:
            raise ValueError("Not able to create an output. Consider running the optimization routine first!")

        obj_array = np.zeros(len(self.prices))
        action_array = np.zeros(len(self.prices))

        obj_array[:-1] = self._objective
        action_array[:-1] = self._action_choices
        data_dict = {
            "DateTime": self.datetimes,
            "Price": self.prices,
            "SOC": self._state_choices,
            "ChargeCycle": self._charges_choices,
            "Charging": action_array,
            "Value": obj_array,
        }
        print(np.sum(obj_array))
        return pd.DataFrame(data_dict)


if __name__ == "__main__":
    import pandas as pd
    import time

    # from rivapy.sample_data.dummy_power_spot_price import spot_price_model
    eff_in = 0.97
    eff_out = 0.97
    np.random.seed(25)
    # states = np.array([0, 1])
    states = np.arange(0, 100.5, step=0.5, dtype=np.float32)
    # states = np.arange(0, 101)
    # actions = np.array([-25, 0,25])
    actions = np.arange(-25, 26)

    timesteps = 100

    prices = np.random.uniform(low=1, high=10, size=timesteps).astype(np.float32)
    max_charges = 200
    max_capacity = 100.0

    dates = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(timesteps)]
    pfc = EnergyPriceForwardCurve.from_existing_pfc(id="1", pfc=pd.DataFrame(data=prices, index=dates), refdate=dt.datetime.today())

    start = time.time()
    batterystorage = BatteryStorage(
        eff_in=eff_in, eff_out=eff_out, max_capacity=max_capacity, pfc=pfc, max_charges=max_charges, states=states, actions=actions, precompile=False
    )

    batterystorage.optimize()
    end = time.time()
    print(end - start)
    print(batterystorage.create_output())
