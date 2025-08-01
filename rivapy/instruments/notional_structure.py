# 2025.07.18 HN
# refactor of notional structure from pyvacon
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
        """Constructor for a notional structure with a constant notional.

        Args:
            notional (float): _description_
        """
        self._notional = [notional]

    def get_amount(self, period: int) -> float:
        """Get the value of the notional.

        Note: Kept list structure to stay consistent with other notional structures.
        However, expectation is that of only one entry in this list.

        Args:
            period (int): index rerferencing to a specific period of rolled out notional

        Returns:
            float: notional value
        """
        return self._notional[period]

    def get_size(self) -> int:
        """If the notional structure is constant, we expect the size to be 1.
        Otherwise, return the amount of notional time stamps used.

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
        """Constructor for a variable notional structure

        Args:
            notionals (list[float]): values for each period, referenced by index and matched to the pay_date_start/end
            pay_date_start (list[datetime]): start date of the payment period
            pay_date_end (list[datetime]): end date of the payment period
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
            "pay_date_start": self._pay_date_start,
            "pay_date_end": self._pay_date_end,
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
        """Notional is recalculated/reset dynamically based on underlying referenced by fx_fixing_id at specific datets (fixing_dates)

        Args:
            ref_currency (str): Currency of the reference
            fx_fixing_id (str): Id of the fixing
            notionals (list[float]): notional values
            pay_date_start (list[datetime]): start of accrual period for that notional
            pay_date_end (list[datetime]): end of accrual period for that notional
            fixing_dates (list[datetime]): date at which notional is reset
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
            "pay_date_start": self._pay_date_start,
            "pay_date_end": self._pay_date_end,
            "ref_currency": self._ref_currency,
            "fx_fixing_id": self._fx_fixing_id,
            "fixing_date": self._fixing_date,
        }
        return return_dict


if __name__ == "__main__":
    pass
