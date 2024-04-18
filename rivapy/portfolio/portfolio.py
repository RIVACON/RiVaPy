"""
Defines a portfolio class. 
"""

from typing import List, Union

import numpy as np


class BaseSpecification:

    def value(self, market_price: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError("A BaseSpecification cannot be priced")
        # this is a placeholder for a general parent class for all specifications

class StockSpecification(BaseSpecification):
    def __init__(self, symbol: str, quantity: float):
        self.__symbol = symbol
        self.__quantity = quantity

    def value(self, market_price: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.__quantity * market_price
    
    @property
    def symbol(self):
        return self.__symbol

class Portfolio:
    
    __supported_types = [StockSpecification]

    def __init__(
            self, 
            constituents: List[BaseSpecification],
        ):

        self.__constituents = constituents
        
        # checks, if any objects were inputted, that are not supported.
        unsupported_types = []
        for c in self.__constituents:
            if not isinstance(c, tuple(self.__supported_types)):
                unsupported_types.append(type(c))
        if unsupported_types:
            raise ValueError("The following types are not yet supported in the `Portfolio` class: {unsupported_types}")

    @property
    def value(self):
        return np.sum([ c.value for c in self.__constituents ])
    
    @property
    def nominal(self):
        return np.sum([ c.nominal for c in self.__constituents ])

    @property
    def size(self):
        return len(self.__constituents)

    @property
    def constituents(self):
        return self.__constituents
    
    def add_constituent(self, const):
        # if any(isinstance(...))
        pass