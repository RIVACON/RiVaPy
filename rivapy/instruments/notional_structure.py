# 2025.07.18 HN
# refactor of notional structurer from pyvacon
# main objective is that for swaps at least, notionals are not always constant

import abc
import rivapy.tools.interfaces as interfaces
from datetime import datetime
from typing import List, Optional, Dict, Union


class NotionalStructure(interfaces.FactoryObject):
    """Abstract base class for notional structures."""

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_amount(self, period: int) -> float:
        """here, period is the INDEX that maps to the notional amount"""
        pass

    def get_pay_date_start(self, period: int) -> Optional[datetime]:
        """get the notional exchange date at the beginning of the period

        Args:
            period (int): is the index of the period

        Returns:
          the date of notional exchange at the beginning of the period,
                or None if there is no notional exchange at the beginning
                of the period

        """

        return None

    def get_pay_date_end(self, period: int) -> Optional[datetime]:
        """get the notional exchange date at the end of the period

        Args:
            period (int): is the index of the period

        Returns:
          the date of notional exchange at the beginning of the period,
                or None if there is no notional exchange at the beginning
                of the period

        """
        return None

    @abc.abstractmethod
    def get_size(self) -> int:
        pass

    @abc.abstractmethod
    def _to_dict(self) -> Dict:
        return_dict = {}
        return return_dict


class ConstNotionalStructure(NotionalStructure):
    """Constant notional means that it does not change over the lifetime.
    Meaning that there are no notional cashflows as well, inflow or outflow.

    Args:
        NotionalStructure (_type_): _description_
    """

    def __init__(self, notional: float):
        self._notional = [notional]

    def get_amount(self, period: int) -> float:
        """since notional is constant, period is used, but expects it to be zero...
        If somethign else is passed, there is an inconsistency in the usage of ConstNotionalStructure... RETHINK

        Args:
            period (int): _description_

        Returns:
            float: _description_
        """
        return self._notional[period]

    def get_size(self) -> int:
        """If the notional structure is constant, we default the 'size' as 1

        Returns:
            int: _description_
        """
        return len(self._notional)

    def _to_dict(self) -> Dict:
        return_dict = {
            "notional": self._notional,
        }
        return return_dict


class VariableNotionalStructure(NotionalStructure):
    def __init__(self, notionals: list[float], pay_date_start: list[datetime], pay_date_end: list[datetime]):
        """_summary_

        Args:
            notionals (list[float]): _description_
            pay_date_start (list[datetime]): _description_
            pay_date_end (list[datetime]): _description_
        """
        self._notional = notionals
        self._pay_date_start = pay_date_start
        self._pay_date_end = pay_date_end

    def get_amount(self, period: int) -> float:
        return self._notional[period]

    def get_pay_date_start(self, period: int) -> datetime:
        return self._pay_date_start[period]

    def get_pay_date_end(self, period: int) -> datetime:
        return self._pay_date_end[period]

    def get_size(self) -> int:
        """Returns the number of notionals

        Returns:
            int: _description_
        """
        return len(self._notional)

    def _to_dict(self) -> Dict:
        # TODO fill out more
        return_dict = {
            "notional": self._notional,
        }
        return return_dict


class ResettingNotionalStructure(NotionalStructure):
    def __init__(
        self,
        ref_currency: str,
        fx_fixing_id: str,
        notionals: list[float],
        pay_date_start: list[datetime],
        pay_date_end: list[datetime],
        fixing_dates: list[datetime],
    ):
        """_summary_

        Args:
            ref_currency (str): _description_
            fx_fixing_id (str): _description_
            notionals (list[float]): _description_
            pay_date_start (list[datetime]): _description_
            pay_date_end (list[datetime]): _description_
            fixing_dates (list[datetime]): _description_
        """

        self._ref_currency = ref_currency
        self._fx_fixing_id = fx_fixing_id
        self._notional = notionals
        self._pay_date_start = pay_date_start
        self._pay_date_end = pay_date_end
        self._fixing_date = fixing_dates

    def get_amount(self, period: int) -> float:
        return self._notional[period]

    def get_pay_date_start(self, period: int) -> datetime:
        return self._pay_date_start[period]

    def get_pay_date_end(self, period: int) -> datetime:
        return self._pay_date_end[period]

    def get_fixing_date(self, period: int) -> datetime:
        return self._fixing_date[period]

    def get_reference_currency(self) -> str:
        return self._ref_currency

    def get_size(self) -> int:
        """Returns the number of notionals

        Returns:
            int: _description_
        """
        return len(self._notional)

    def _to_dict(self) -> Dict:
        # TODO fill out more
        return_dict = {
            "notional": self._notional,
        }
        return return_dict


# #TODO implment in the case we need to build a notional structure from a given database...
# def build_notional_structure(
#     notional_data: Dict[str, List[str]],
#     var_notional_data: Optional[Dict[str, List[str]]],
#     notional_id: int
# ) -> Union[ConstNotionalStructure, VariableNotionalStructure, ResettingNotionalStructure]:

#     """
#     Factory function to build the appropriate NotionalStructure object.

#     Parameters:
#         notional_data: Dictionary-like table of notional metadata.
#         var_notional_data: Dictionary-like table of variable/resetting notional values.
#         notional_id: ID of the notional structure to build.

#     Returns:
#         An instance of ConstNotionalStructure, VariableNotionalStructure, or ResettingNotionalStructure.
#     """


if __name__ == "__main__":
    pass
