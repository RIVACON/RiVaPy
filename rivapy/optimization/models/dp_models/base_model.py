import abc
import numpy as np
from typing import List, Literal


class DPBaseModel(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_state_transition_matrix(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def state_mapping(self, state_transition_matrix: np.ndarray):
        pass

    @abc.abstractmethod
    def get_states_number(self) -> int:
        pass

    @abc.abstractmethod
    def create_output(self, actions: List[int], states: List[int], values: np.ndarray):
        pass
