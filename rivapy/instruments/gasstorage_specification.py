import numpy as np
import rivapy.tools.interfaces as interfaces

class GasStorageSpecification(interfaces.FactoryObject):
    def __init__(self, 
                storage_capacity: float, 
                withdrawal_rate: float, 
                injection_rate: float, 
                withdrawal_cost: float = 0.0, 
                injection_cost: float = 0.0,
                min_level: float = 0.0,
                start_level: float = 0.0,
                end_level: float = 0.0 
                ):
        """Constructor for gas storage specification

        Args:
            storage_capacity (float): Maximum possible level for the gas storage.
            withdrawal_rate (float): Maximum withdrawal rate.
            injection_rate (float): Maximum injection rate.
            withdrawal_cost (float): Relative cost of withdrawal.
            injection_cost (float): Relative cost of injection.
            min_level (float, optional): Minimum level for the gas storage. Defaults to 0.0.
            start_level (float, optional): Start level for the gas storage. Defaults to 0.0.
            end_level (float, optional): End level for gas storage. Defaults to 0.0.
        """
        
        self.storage_capacity = storage_capacity
        self.withdrawal_rate = withdrawal_rate
        self.injection_rate = injection_rate
        self.withdrawal_cost = withdrawal_cost
        self.injection_cost = injection_cost
        self.min_level = min_level
        self.start_level = start_level
        self.end_level = end_level
        
    def _to_dict(self) -> dict:
        result = {'storage_capacity': self.storage_capacity, 
                  'withdrawal_rate': self.withdrawal_rate,
                  'injection_rate': self.injection_rate,
                  'withdrawal_cost': self.withdrawal_cost,
                  'injection_cost': self.injection_cost,
                  'min_level': self.min_level,
                  'start_level': self.start_level,
                  'end_level': self.end_level
                  }
        return super()._to_dict()