# 2025.07.24 Hans Nguyen
from datetime import datetime, date
from typing import List as _List, Union as _Union, Tuple, Dict, Any


#########################################################################
class CashFlow:
    # goal is to define a dynamically growing class that is still able to use
    # type validation and dot-access e.g. class.variable
    # the point for dynamically growing is to allow for flexibility of future development and use cases
    # In the end, it might be better to just define clearly the CashFlow class with
    # strict attributes ... #TODO

    # Define expected types here
    # Can be expanded when we know for sure which features we want to ensure typing for
    _schema = {
        "start_date": datetime,
        "end_date": datetime,
        "ccy": str,
        "amortization": bool,
        "prepayment_risk": bool,
    }

    def __init__(self, val: float = None):
        self.val = val
        self._attributes = {}

    def __getattr__(self, name: str) -> Any:
        """overwritting default getter for dynamically growing one

        Args:
            name (str): name of the the desired attribute

        Raises:
            AttributeError: attribute name not included

        Returns:
            Any: value of the desired attribute
        """
        try:
            return self._attributes[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        """overwriting default setter for dynamically growing one
        which also checks for expected type validation.

        Args:
            name (str): new name for desired attribute
            value (Any): value to be stored in desired attribute

        Raises:
            TypeError: For known attributes defined in schema, raise error if type mismatch for value
        """
        if name in {"val", "_attributes"}:  # avoid infinite recursion
            super().__setattr__(name, value)  # use the the normal attribute storage from base class
        else:  # logic for new attirbute storage
            expected_type = self._schema.get(name)  # if it doesnt exist, can attempt to set new attribute
            if expected_type is not None and not isinstance(value, expected_type):
                raise TypeError(f"Attribute '{name}' must be of type {expected_type}, got {type(value)}")
            self._attributes[name] = value

    def __delattr__(self, name: str):
        if name in self._attributes:
            del self._attributes[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def keys(self):
        return list(self._attributes.keys())

    def items(self):
        return self._attributes.items()

    def __dir__(self):
        """overwritten in order to show dynamically stored attributes as well.

        Returns:
            _type_: _description_
        """
        return super().__dir__() + list(self._attributes.keys())
