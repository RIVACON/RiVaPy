import numpy as np
import datetime as dt
import rivapy.tools.interfaces as interfaces
from rivapy.tools import SimpleSchedule
from rivapy.tools.factory import create as _create
from typing import Tuple, List


class EnergyFutureSpecifications(interfaces.FactoryObject):
    def __init__(self, schedule: SimpleSchedule, price: float, name: str) -> None:
        """Specification for an energy future contract. The delivery period is defined by the schedule.

        Args:
            schedule (SimpleSchedule): Delivery period
            price (float): Price
            name (str): Name
        """
        if isinstance(schedule, dict):
            self.schedule = _create(schedule)
        else:
            self.schedule = schedule
        self.price = price
        self.name = name

    def get_schedule(self) -> np.ndarray:
        """Returns each delivery date time

        Returns:
            np.ndarray: Numpy array containing each delivery date time
        """
        return self.schedule.get_schedule()

    def get_price(self) -> float:
        """Returns the price

        Returns:
            float: Price
        """
        return self.price

    def get_start(self) -> dt.datetime:
        """Returns the delivery start. Note that this may not necessarily correspond to the first delivery.

        Returns:
            dt.datetime: Start date time of the delivery scheduler
        """
        return self.schedule.start

    def get_end(self) -> dt.datetime:
        """Returns the delivery end. Note that this may not necessarily correspond to the last delivery.

        Returns:
            dt.datetime: End date time of the delivery scheduler
        """
        return self.schedule.end

    def get_start_end(self) -> Tuple[dt.datetime, dt.datetime]:
        """Returns the start and end as a tuple, where (start, end)

        Returns:
            Tuple[dt.datetime, dt.datetime]: Tuple containing start and end
        """
        return (self.get_start(), self.get_end())

    def _to_dict(self) -> dict:
        return {"schedule": self.schedule.to_dict(), "price": self.price, "name": self.name}

    @staticmethod
    def _create_sample(n_samples: int, seed: int = None, ref_date=None) -> List["EnergyFutureSpecifications"]:
        """Creates a sample of random ``EnergyFutureSpecifiactions`` objects.

        Returns:
            ListEnergyFutureSpecifications]: List of sampled ``EnergyFutureSpecifiactions`` objects
        """
        if seed is not None:
            np.random.seed(seed)
        schedules = SimpleSchedule._create_sample(n_samples, seed, ref_date)
        result = []
        for i, schedule in enumerate(schedules):
            price = np.random.uniform(low=50.0, high=150.0)
            result.append(EnergyFutureSpecifications(schedule=schedule, price=price, name=f"Contract_{i+1}"))
        return result
