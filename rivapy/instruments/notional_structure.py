# 2025.07.18 HN
# refactor of notional structurer from pyvacon
# main objective is that for swaps at least, notionals are not always constant


import rivapy.tools.interfaces as interfaces
from datetime import datetime
from typing import List, Optional, Dict, Union


class NotionalStructure(interfaces.FactoryObject):
    """Abstract base class for notional structures."""

    def __init__(self):
        pass

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

    @_abstract
    def

class ConstNotionalStructure(NotionalStructure):
    def __init__(self, notional: float):
        self._notional = notional

    def get_amount(self, period: int) -> float:
        """since notional is constant, period is unused...

        Args:
            period (int): _description_

        Returns:
            float: _description_
        """
        return self._notional


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
