import numpy as np
import pandas as pd
import datetime as dt
from numba import njit as numba_njit

from rivapy.marketdata import EnergyPriceForwardCurve
from typing import Union, Optional, Callable


USE_NUMBA = True
print(f"USE NUMBA: {USE_NUMBA}")


def _default_decorator(func: Callable):
    def _wraper(*args, **kwargs):
        return func(*args, **kwargs)

    return _wraper


def njit(func: Callable):
    if USE_NUMBA:
        return numba_njit(func)
    else:
        return _default_decorator(func)


@njit
def _value_wrapper(
    state: float,
    charge: float,
    action: float,
    max_state: float,
    min_state: float,
    discharge_gain: float,
    charge_costs: float,
    t: int,
    states: np.ndarray,
    max_charges: np.ndarray,
    max_charge: float,
    value_matrix: np.ndarray,
    mode: int,
    penalty: float,
    tolerance: float,
) -> float:

    if state < min_state or state > max_state:
        value = penalty
        return value

    if charge > max_charge:
        return penalty

    if mode == 1:
        reward = (-1) * action * charge_costs

    elif mode == -1:
        reward = (-1) * action * discharge_gain

    elif mode == 0:
        reward = 0.0

    value = __value(
        state=state,
        charge=charge,
        t=t,
        states=states,
        max_charges=max_charges,
        value_matrix=value_matrix,
        reward=reward,
        penalty=penalty,
        tolerance=tolerance,
    )
    return value


@njit
def __value(
    state: float,
    charge: float,
    t: int,
    states: np.ndarray,
    max_charges: np.ndarray,
    value_matrix: np.ndarray,
    reward: float,
    penalty: float,
    tolerance: float,
) -> float:

    idx_state = np.searchsorted(states, state)

    idx_charge = np.searchsorted(max_charges, charge)

    ceil_state = idx_state
    floor_state = idx_state - 1

    ceil_charge = idx_charge
    floor_charge = idx_charge - 1
    # check if both points are on the grid
    if np.abs(states[ceil_state] - state) < tolerance and np.abs(max_charges[ceil_charge] - charge) < tolerance:
        if np.abs(value_matrix[t, ceil_state, ceil_charge] - penalty) < tolerance:
            return penalty

        return value_matrix[t, ceil_state, ceil_charge] + reward
    # check if state is on the grid
    elif np.abs(states[ceil_state] - state) < tolerance:

        value = __linear_interpolate_charge(
            value_matrix=value_matrix,
            t=t,
            ceil_idx=ceil_charge,
            floor_idx=floor_charge,
            state_idx=ceil_state,
            charge=charge,
            charges=max_charges,
            reward=reward,
            penalty=penalty,
            tolerance=tolerance,
        )
        return value
    # check if charge is on the grid
    elif np.abs(max_charges[ceil_charge] - charge) < tolerance:

        value = __linear_interpolate_state(
            value_matrix=value_matrix,
            t=t,
            ceil_idx=ceil_state,
            floor_idx=floor_state,
            state=state,
            charge_idx=ceil_charge,
            states=states,
            reward=reward,
            penalty=penalty,
            tolerance=tolerance,
        )
        return value
    else:
        # 2d interpolation
        if (
            np.abs(value_matrix[t, floor_state, floor_charge] - penalty) < tolerance
            or np.abs(value_matrix[t, floor_state, ceil_charge] - penalty) < tolerance
            or np.abs(value_matrix[t, ceil_state, floor_charge] - penalty) < tolerance
            or np.abs(value_matrix[t, ceil_state, ceil_charge] - penalty) < tolerance
        ):
            return penalty
        else:
            charge_numerator = max_charges[ceil_charge] - max_charges[floor_charge]
            factor_ceil_charge = (max_charges[ceil_charge] - charge) / charge_numerator
            factor_floor_charge = (charge - max_charges[floor_charge]) / charge_numerator

            state_numerator = states[ceil_state] - states[floor_state]
            factor_ceil_state = (states[ceil_state] - state) / state_numerator
            factor_floor_state = (state - states[floor_state]) / state_numerator

            state_part = factor_ceil_charge * (
                factor_ceil_state * value_matrix[t, floor_state, floor_charge] + factor_floor_state * value_matrix[t, ceil_state, floor_charge]
            )
            charge_part = factor_floor_charge * (
                factor_ceil_state * value_matrix[t, floor_state, ceil_charge] + factor_floor_state * value_matrix[t, ceil_state, ceil_charge]
            )
            value_next_state = state_part + charge_part
            return value_next_state + reward


@njit
def __linear_interpolate_charge(
    value_matrix: np.ndarray,
    t: int,
    ceil_idx: int,
    floor_idx: int,
    state_idx: int,
    charge: float,
    charges: np.ndarray,
    reward: float,
    penalty: float,
    tolerance: float,
) -> float:
    if ceil_idx == 0:
        floor_idx = ceil_idx
        ceil_idx = ceil_idx + 1

    if np.abs(value_matrix[t, state_idx, ceil_idx] - penalty) < tolerance or np.abs(value_matrix[t, state_idx, floor_idx] - penalty) < tolerance:
        return penalty

    m = (value_matrix[t, state_idx, ceil_idx] - value_matrix[t, state_idx, floor_idx]) / (charges[ceil_idx] - charges[floor_idx])
    b = value_matrix[t, state_idx, floor_idx] - m * charges[floor_idx]
    return m * charge + b + reward


@njit
def __linear_interpolate_state(
    value_matrix: np.ndarray,
    t: int,
    ceil_idx: int,
    floor_idx: int,
    state: float,
    charge_idx: int,
    states: np.ndarray,
    reward: float,
    penalty: float,
    tolerance: float,
) -> float:
    if ceil_idx == 0:
        floor_idx = ceil_idx
        ceil_idx = ceil_idx + 1

    if np.abs(value_matrix[t, ceil_idx, charge_idx] - penalty) < tolerance or np.abs(value_matrix[t, floor_idx, charge_idx] - penalty) < tolerance:
        return penalty

    m = (value_matrix[t, ceil_idx, charge_idx] - value_matrix[t, floor_idx, charge_idx]) / (states[ceil_idx] - states[floor_idx])
    b = value_matrix[t, floor_idx, charge_idx] - m * states[floor_idx]
    return m * state + b + reward


@njit
def backward(
    eff_in: float,
    eff_out: float,
    max_capacity: float,
    states: np.ndarray,
    actions: np.ndarray,
    # prices: np.ndarray,
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
    max_charges: np.ndarray,
    base_dispatch: np.ndarray,
    end_state: Optional[float] = None,
    penalty: float = -1e12,
    tolerance: float = 1e-8,
):
    T = len(bid_prices)

    max_state = np.max(states)
    min_state = np.min(states)
    max_action = np.max(actions)
    min_action = np.min(actions)

    max_charge = np.max(max_charges)
    max_charge_id = len(max_charges) - 1

    max_state_id = len(states) - 1

    _range_states = range(len(states))
    _range_charges = range(len(max_charges))

    if end_state is None:
        value_matrix = np.zeros((T, len(states), len(max_charges)), dtype=np.float64)
    else:
        end_state_id = np.searchsorted(states, end_state)
        value_matrix = np.ones((T, len(states), len(max_charges)), dtype=np.float64) * penalty
        value_matrix[:, end_state_id, :] = 0.0

    for i in range(1, T):
        t = T - i - 1

        # price = prices[t]

        bid_price = bid_prices[t]
        ask_price = ask_prices[t]

        base_dispatch_action = base_dispatch[t]

        discharge_price = eff_out * ask_price * max_capacity / 100.0
        charge_price_volume = (1 / eff_in) * bid_price * max_capacity / 100.0
        charge_price = bid_price * max_capacity / 100.0

        for state_id in _range_states:
            state = states[state_id]

            for charge_id in _range_charges:
                charge = max_charges[charge_id]

                # if charge == max_charge:
                #     value_matrix[t, state_id, charge_id] = value_matrix[t + 1, state_id, charge_id]
                #     continue

                min_value = penalty
                for action_id in range(len(actions)):

                    action = actions[action_id] + base_dispatch_action

                    if action > max_action or action < min_action:
                        continue

                    if action > 0:
                        next_state = state + eff_in * action
                        next_charge = charge + eff_in * action
                        mode = 1

                    elif action < 0:
                        next_state = state + action
                        next_charge = charge
                        mode = -1

                    else:
                        next_state = state
                        next_charge = charge
                        mode = 0

                    value = _value_wrapper(
                        state=next_state,
                        charge=next_charge,
                        action=action,
                        max_state=max_state,
                        min_state=min_state,
                        discharge_gain=discharge_price,
                        charge_costs=charge_price,
                        t=t + 1,
                        states=states,
                        max_charges=max_charges,
                        max_charge=max_charge,
                        value_matrix=value_matrix,
                        mode=mode,
                        penalty=penalty,
                        tolerance=tolerance,
                    )
                    if min_value < value:
                        min_value = value

                dsptch_max_action = min(base_dispatch_action + max_action, max_action)
                dsptch_min_action = max(base_dispatch_action + min_action, min_action)

                # Check if continuous action could fill the storage
                if (max_state - state) <= (dsptch_max_action * eff_in):
                    next_charge = charge + max_state - state

                    if next_charge <= max_charge:
                        idx_charge = np.searchsorted(max_charges, next_charge)
                        ceil = idx_charge
                        floor = idx_charge - 1
                        reward = (-1) * (max_state - state) * charge_price_volume
                        value = __linear_interpolate_charge(
                            value_matrix=value_matrix,
                            t=t + 1,
                            ceil_idx=ceil,
                            floor_idx=floor,
                            state_idx=max_state_id,
                            charge=next_charge,
                            charges=max_charges,
                            reward=reward,
                            penalty=penalty,
                            tolerance=tolerance,
                        )

                        if min_value < value:
                            min_value = value

                # Check if continuous action could empty the storage
                if np.abs((min_state - state)) <= np.abs(dsptch_min_action):
                    value = value_matrix[t + 1, 0, charge_id] + np.abs((min_state - state)) * discharge_price

                    if min_value < value:
                        min_value = value

                # Check if maxing out the max_charges is optimal (consider state of charge)
                if (max_charge - charge <= dsptch_max_action * eff_in) and (state + (max_charge - charge) * eff_in <= max_state):
                    reward = (-1) * (max_charge - charge) * charge_price_volume
                    next_state = state + (max_charge - charge) * eff_in

                    idx_next_state = np.searchsorted(states, next_state)

                    value = __linear_interpolate_state(
                        value_matrix=value_matrix,
                        t=t + 1,
                        ceil_idx=idx_next_state,
                        floor_idx=idx_next_state - 1,
                        state=next_state,
                        charge_idx=max_charge_id,
                        states=states,
                        reward=reward,
                        penalty=penalty,
                        tolerance=tolerance,
                    )

                    if min_value < value:
                        min_value = value

                # check if reaching the end state is possible and optimal
                if end_state is not None:
                    if end_state > state:
                        if (end_state - state) <= (dsptch_max_action * eff_in):
                            next_charge = charge + end_state - state

                            if next_charge <= max_charge:
                                idx_charge = np.searchsorted(max_charges, next_charge)
                                ceil = idx_charge
                                floor = idx_charge - 1
                                reward = (-1) * (end_state - state) * charge_price_volume
                                value = __linear_interpolate_charge(
                                    value_matrix=value_matrix,
                                    t=t + 1,
                                    ceil_idx=ceil,
                                    floor_idx=floor,
                                    state_idx=end_state_id,
                                    charge=next_charge,
                                    charges=max_charges,
                                    reward=reward,
                                    penalty=penalty,
                                    tolerance=tolerance,
                                )

                                if min_value < value:
                                    min_value = value

                    elif end_state < state:
                        if np.abs((end_state - state)) <= np.abs(dsptch_min_action):
                            value = value_matrix[t + 1, end_state_id, charge_id] + np.abs((end_state - state)) * discharge_price

                        if min_value < value:
                            min_value = value

                value_matrix[t, state_id, charge_id] = min_value

    return value_matrix


@njit
def forward(
    eff_in: float,
    eff_out: float,
    max_capacity: float,
    value_matrix: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    # prices: np.ndarray,
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
    max_charges: np.ndarray,
    base_dispatch: np.ndarray,
    start_state: Optional[float] = None,
    start_charges: Optional[int] = None,
    end_state: Optional[float] = None,
    penalty: float = -1e12,
    tolerance: float = 1e-8,
):
    T = len(bid_prices)

    state_choices = np.zeros(T, dtype=np.float32)
    charges_choices = np.zeros(T, dtype=np.float32)
    objective = np.zeros(T - 1, dtype=np.float32)
    action_choices = np.zeros(T - 1, dtype=np.float32)

    max_state = np.max(states)
    min_state = np.min(states)
    max_action = np.max(actions)
    min_action = np.min(actions)

    max_charge = np.max(max_charges)
    max_charge_id = len(max_charges) - 1

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
                charges_choices[t] = max_charges[prev_charge]

            elif start_charges is None:
                idx = np.searchsorted(states, start_state)
                prev_state = start_state
                prev_charge = np.argmax(value_matrix_slice[idx, :])

                state_choices[t] = prev_state
                charges_choices[t] = max_charges[prev_charge]

            elif start_state is None:
                idx = np.searchsorted(max_charges, start_charges)
                prev_charge = start_charges
                prev_state_idx = np.argmax(value_matrix_slice[:, idx])

                state_choices[t] = states[prev_state_idx]
                charges_choices[t] = prev_charge
            else:
                prev_state = start_state
                prev_charge = start_charges

                state_choices[t] = prev_state
                charges_choices[t] = prev_charge

            continue
        # price = prices[t - 1]

        bid_price = bid_prices[t - 1]
        ask_price = ask_prices[t - 1]

        discharge_price = eff_out * ask_price * max_capacity / 100.0
        charge_price_volume = (1 / eff_in) * bid_price * max_capacity / 100.0
        charge_price = bid_price * max_capacity / 100.0

        prev_state = state_choices[t - 1]
        prev_charge = charges_choices[t - 1]

        prev_idx = np.searchsorted(max_charges, prev_charge)
        prev_ceil = prev_idx
        prev_floor = prev_idx - 1

        prev_state_idx = np.searchsorted(states, prev_state)

        base_dispatch_action = base_dispatch[t - 1]

        # if prev_charge == max_charge:
        #     action_choices[t - 1] = 0
        #     state_choices[t] = prev_state
        #     charges_choices[t] = prev_charge
        #     continue

        best_value = penalty

        chosen_state = None
        chosen_action = None

        for action_id in range(len(actions)):

            action = actions[action_id] + base_dispatch_action

            if action > max_action or action < min_action:
                continue

            if action > 0:
                next_state = prev_state + eff_in * action
                next_charge = prev_charge + eff_in * action
                mode = 1

            elif action < 0:
                next_state = prev_state + action
                next_charge = prev_charge
                mode = -1

            else:
                next_state = prev_state
                next_charge = prev_charge
                mode = 0

            value = _value_wrapper(
                state=next_state,
                charge=next_charge,
                action=action,
                max_state=max_state,
                min_state=min_state,
                discharge_gain=discharge_price,
                charge_costs=charge_price,
                t=t,
                states=states,
                max_charges=max_charges,
                max_charge=max_charge,
                value_matrix=value_matrix,
                mode=mode,
                penalty=penalty,
                tolerance=tolerance,
            )

            if best_value < value:
                best_value = value
                chosen_state = next_state
                chosen_action = action
                chosen_charge = next_charge

        dsptch_max_action = min(base_dispatch_action + max_action, max_action)
        dsptch_min_action = max(base_dispatch_action + min_action, min_action)

        # Check if continuous action could fill the storage
        if (max_state - prev_state) <= (dsptch_max_action * eff_in):
            next_charge = prev_charge + max_state - prev_state

            if next_charge <= max_charge:
                idx_charge = np.searchsorted(max_charges, next_charge)
                ceil = idx_charge
                floor = idx_charge - 1
                reward = (-1) * (max_state - prev_state) * charge_price_volume
                value = __linear_interpolate_charge(
                    value_matrix=value_matrix,
                    t=t,
                    ceil_idx=ceil,
                    floor_idx=floor,
                    state_idx=max_state_id,
                    charge=next_charge,
                    charges=max_charges,
                    reward=reward,
                    penalty=penalty,
                    tolerance=tolerance,
                )

                next_state = max_state
                action = (max_state - prev_state) * (1 / eff_in)
                # next_charge = prev_charge + 1
                if best_value < value:
                    best_value = value
                    chosen_state = next_state
                    chosen_action = action
                    chosen_charge = next_charge

        # Check if continuous action could empty the storage
        if np.abs((min_state - prev_state)) <= np.abs(dsptch_min_action):
            reward = np.abs((min_state - prev_state)) * discharge_price

            value = __linear_interpolate_charge(
                value_matrix=value_matrix,
                t=t,
                ceil_idx=prev_ceil,
                floor_idx=prev_floor,
                state_idx=0,
                charge=prev_charge,
                charges=max_charges,
                reward=reward,
                penalty=penalty,
                tolerance=tolerance,
            )

            next_state = min_state
            action = (-1) * np.abs((min_state - prev_state))
            next_charge = prev_charge
            if best_value < value:
                best_value = value
                chosen_state = next_state
                chosen_action = action
                chosen_charge = next_charge

        # Check if maxing out the max_charges is optimal (consider state of charge)
        if (max_charge - prev_charge <= dsptch_max_action * eff_in) and (prev_state + (max_charge - prev_charge) * eff_in <= max_state):
            reward = (-1) * (max_charge - prev_charge) * charge_price_volume
            next_state = prev_state + (max_charge - prev_charge) * eff_in

            idx_next_state = np.searchsorted(states, next_state)

            value = __linear_interpolate_state(
                value_matrix=value_matrix,
                t=t + 1,
                ceil_idx=idx_next_state,
                floor_idx=idx_next_state - 1,
                state=next_state,
                charge_idx=max_charge_id,
                states=states,
                reward=reward,
                penalty=penalty,
                tolerance=tolerance,
            )

            next_charge = max_charge
            action = (max_charge - prev_charge) / eff_in

            if best_value < value:
                best_value = value
                chosen_state = next_state
                chosen_action = action
                chosen_charge = next_charge

        # check if reaching the end state is possible and optimal
        if end_state is not None:
            if end_state > prev_state:
                if (end_state - prev_state) <= (dsptch_max_action * eff_in):
                    next_charge = prev_charge + end_state - prev_state

                    if next_charge <= max_charge:
                        idx_charge = np.searchsorted(max_charges, next_charge)
                        ceil = idx_charge
                        floor = idx_charge - 1

                        reward = (-1) * (end_state - prev_state) * charge_price_volume

                        value = __linear_interpolate_charge(
                            value_matrix=value_matrix,
                            t=t,
                            ceil_idx=ceil,
                            floor_idx=floor,
                            state_idx=end_state_id,
                            charge=next_charge,
                            charges=max_charges,
                            reward=reward,
                            penalty=penalty,
                            tolerance=tolerance,
                        )

                        next_state = end_state
                        action = (end_state - prev_state) * (1 / eff_in)
                        if best_value < value:
                            best_value = value
                            chosen_state = next_state
                            chosen_action = action
                            chosen_charge = next_charge

            elif end_state < prev_state:
                if np.abs((end_state - prev_state)) <= np.abs(dsptch_min_action):
                    reward = np.abs((end_state - prev_state)) * discharge_price

                    value = __linear_interpolate_charge(
                        value_matrix=value_matrix,
                        t=t,
                        ceil_idx=prev_ceil,
                        floor_idx=prev_floor,
                        state_idx=end_state_id,
                        charge=prev_charge,
                        charges=max_charges,
                        reward=reward,
                        penalty=penalty,
                        tolerance=tolerance,
                    )

                    next_state = end_state
                    action = end_state - prev_state
                    next_charge = prev_charge
                    if best_value < value:
                        best_value = value
                        chosen_state = next_state
                        chosen_action = action
                        chosen_charge = next_charge

        state_choices[t] = chosen_state
        charges_choices[t] = chosen_charge
        action_choices[t - 1] = chosen_action

        if chosen_action > 0:
            objective[t - 1] = chosen_action * (-1) * charge_price
        else:
            objective[t - 1] = chosen_action * (-1) * discharge_price

        # print(print_value)

    state_choices *= max_capacity / 100.0
    charges_choices *= max_capacity / 100.0
    action_choices *= max_capacity / 100.0

    return state_choices, charges_choices, action_choices, objective


class BatteryStorage:

    def __init__(
        self,
        eff_in: float,
        eff_out: float,
        max_capacity: float,
        bid_prices: np.ndarray,
        ask_prices: np.ndarray,
        # pfc: EnergyPriceForwardCurve,
        states: np.ndarray,
        actions: np.ndarray,
        max_charges: np.ndarray,
        base_dispatch: Optional[np.ndarray] = None,
        start_state: Optional[float] = None,
        start_charges: Optional[int] = None,
        end_state: Optional[float] = None,
        # start_date: Optional[dt.datetime] = None,
        # end_date: Optional[dt.datetime] = None,
        precompile: bool = True,
        precompile_timefraction: float = 0.1,
        penalty: float = -1e12,
        tolerance: float = 1e-8,
    ):
        self._eff_in = eff_in
        self._eff_out = eff_out
        self._max_capacity = max_capacity

        self._states = states
        self._actions = actions
        self._max_charges = max_charges

        self._bid_prices = bid_prices
        self._ask_prices = ask_prices

        if base_dispatch is None:
            self._base_dispatch = np.zeros(len(bid_prices) - 1, dtype=np.float32)
        else:
            self._base_dispatch = base_dispatch
        # self._pfc = pfc

        # self.pfc = self._pfc.get_pfc()

        # if (start_date is None) and (end_date is not None):
        #     filtered_df = self.pfc.iloc[self.pfc.index < end_date, 0]

        # elif (start_date is not None) and (end_date is None):
        #     filtered_df = self.pfc.iloc[start_date <= self.pfc.index, 0]

        # elif (start_date is not None) and (end_date is not None):
        #     filtered_df = self.pfc.iloc[(start_date <= self.pfc.index) & (self.pfc.index < end_date), 0]

        # else:
        #     filtered_df = self.pfc.iloc[:, 0]

        # self.prices = filtered_df.values
        # self.datetimes = list(filtered_df.index)

        self._start_state = start_state

        self._start_charges = start_charges

        self._end_state = end_state

        self._penalty = penalty
        self._tolerance = tolerance

        if precompile is True:
            self.__precompile(precompuile_timefraction=precompile_timefraction)

        self.__optimized: bool = False

    def __precompile(self, precompuile_timefraction: float):
        def state_check(state: Optional[Union[float, int]], value: float):
            if state is None:
                return None
            else:
                return value

        eff_in = self._eff_in
        eff_out = self._eff_out
        states = self._states
        actions = self._actions
        max_charges = self._max_charges
        max_capacity = self._max_capacity
        bid_prices = self._bid_prices[: max(int(round(len(self._bid_prices) * precompuile_timefraction)), 2)]
        ask_prices = self._ask_prices[: max(int(round(len(self._ask_prices) * precompuile_timefraction)), 2)]

        base_dispatch = np.zeros(len(ask_prices - 1), dtype=np.float32)

        start_state = state_check(self._start_state, value=0.0)
        start_charges = state_check(self._start_charges, value=0.0)
        end_state = state_check(self._end_state, 0.0)

        value_matrix = backward(
            eff_in,
            eff_out,
            max_capacity,
            states,
            actions,
            bid_prices,
            ask_prices,
            max_charges,
            base_dispatch,
            end_state=end_state,
            penalty=self._penalty,
            tolerance=self._tolerance,
        )
        _ = forward(
            eff_in,
            eff_out,
            max_capacity,
            value_matrix,
            states,
            actions,
            bid_prices,
            ask_prices,
            max_charges,
            base_dispatch,
            start_state=start_state,
            start_charges=start_charges,
            end_state=end_state,
            penalty=self._penalty,
            tolerance=self._tolerance,
        )

    def optimize(self):
        value_matrix = backward(
            eff_in=self._eff_in,
            eff_out=self._eff_out,
            max_capacity=self._max_capacity,
            states=self._states,
            actions=self._actions,
            bid_prices=self._bid_prices,
            ask_prices=self._ask_prices,
            max_charges=self._max_charges,
            base_dispatch=self._base_dispatch,
            end_state=self._end_state,
            penalty=self._penalty,
            tolerance=self._tolerance,
        )

        self._state_choices, self._charges_choices, self._action_choices, self._objective = forward(
            eff_in=self._eff_in,
            eff_out=self._eff_out,
            max_capacity=self._max_capacity,
            value_matrix=value_matrix,
            states=self._states,
            actions=self._actions,
            bid_prices=self._bid_prices,
            ask_prices=self._ask_prices,
            max_charges=self._max_charges,
            base_dispatch=self._base_dispatch,
            start_state=self._start_state,
            start_charges=self._start_charges,
            end_state=self._end_state,
            penalty=self._penalty,
            tolerance=self._tolerance,
        )

        self.__optimized = True

    def create_output(self) -> pd.DataFrame:
        if not self.__optimized:
            raise ValueError("Not able to create an output. Consider running the optimization routine first!")

        obj_array = np.zeros(len(self._ask_prices))
        action_array = np.zeros(len(self._ask_prices))

        obj_array[:-1] = self._objective
        action_array[:-1] = self._action_choices
        data_dict = {
            # "DateTime": self.datetimes,
            "Bid Price": self._bid_prices,
            "Ask Price": self._ask_prices,
            "SOC": self._state_choices,
            "ChargeCycle": self._charges_choices,
            "Charging": action_array,
            "Value": obj_array,
        }
        print(np.sum(obj_array))
        return pd.DataFrame(data_dict)

    def get_dispatch(self) -> np.ndarray:
        return self._action_choices


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

    timesteps = 1000

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
