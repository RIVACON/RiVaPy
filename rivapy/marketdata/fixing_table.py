# 2025.07.17 Hans Nguyen
# Fixing table class for Interest Rate Swap Pricer for IR bootstrapping implementation


# Modules
from typing import List as _List, Union as _Union, Tuple, Dict, Any
from datetime import datetime, date
import bisect


# Class
class FixingTable:
    """Contains all the past fixings of an instrument.
    refactored by Hans Nguyen from Christan Menn (c++)
    """

    def __init__(self, id: str = None, fixings: Dict[str, Tuple[_List[datetime], _List[float]]] = None):
        # id: Optional[str] = None,
        # fixings: Optional[Dict[str, Tuple[List[datetime], List[float]]]] = None):
        """Constructor for FixingTable. Creates 'empty' instance if no parameters passed.

        Args:
            id (str): identifier for the table. Defaults to None.
            fixings (Dict[str, Tuple[List[datetime], List[float]]]): A dictionary mapping underlying IDs to
                                                         pairs of fixing dates and values. Defaults to None.
        """

        self.id = id
        self.fixings: Dict[str, Tuple[_List[datetime], _List[float]]] = fixings if fixings else {}

        if fixings:
            expected_len = None
            for udl, (dates, values) in fixings.items():
                if expected_len is None:
                    expected_len = len(dates)
                if len(dates) != expected_len:
                    raise ValueError(f"Inconsistent number of dates for {udl}: {len(dates)} != {expected_len}")
                if len(values) != expected_len:
                    raise ValueError(f"Inconsistent number of fixings for {udl}: {len(values)} != {expected_len}")

    def get_object_type(self) -> str:
        return "FIXING_TABLE"

    def get_fixing(self, udl_id: str, fixing_date: datetime) -> float:
        """
        Return fixing for a given underlying and date. Raises error if not found.


        Args:
            udl_id (str):  The underlying ID.
            fixing_date (datetime): The date of the fixing

        Raises:
            ValueError: no fixings for underlying
            ValueError: no fixing for given date

        Returns:
            float:  The fixing value if found, otherwise raise error
        """

        if udl_id not in self.fixings:
            raise ValueError(f"No fixings for underlying '{udl_id}'")

        dates, values = self.fixings[udl_id]
        for i, date in enumerate(dates):
            if date == fixing_date:
                return values[i]

        raise ValueError(f"No fixing found for '{udl_id}' on {fixing_date.isoformat()}")

    def get_num_underlyings(self) -> int:
        """Return number of underlyings

        Returns:
            int: _description_
        """
        return len(self.fixings)

    def add(self, key: str, fixing_date: datetime, value: float) -> None:
        """
        Add a fixing and keep entries sorted by date.

        Args:
            key (str): The underlying ID key to map to fixing.
            fixing_date (datetime): The date of the fixing.
            value (float): The fixing value.
        """

        if key not in self.fixings:
            self.fixings[key] = ([], [])

        dates, values = self.fixings[key]

        # Insert while keeping both lists sorted to preserve date order
        index = bisect.bisect_left(dates, fixing_date)
        dates.insert(index, fixing_date)
        values.insert(index, value)

    def get(self, udl_id: str) -> Tuple[_List[datetime], _List[float]]:
        """
        Get Tuple of all fixings for a given underlying.

        Args:
            udl_id (str): key of underlying ID

        Returns:
            Tuple[List[datetime], List[float]]: Tuple of dates and fixings.
        """
        if udl_id not in self.fixings:
            return [], []  # TODO or riase an error?
        dates, values = self.fixings[udl_id]

        return dates[:], values[:]


# Functions


if __name__ == "__main__":
    pass
