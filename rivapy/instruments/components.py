# -*- coding: utf-8 -*-
from typing import Union as _Union, List, Tuple, Dict, Any, Optional
import numpy as np
from datetime import datetime, date
import rivapy.tools.interfaces as interfaces
from rivapy.tools.datetools import _date_to_datetime
from rivapy.tools._validators import _check_positivity, _check_relation, _is_chronological
from rivapy.tools.enums import DayCounterType, Rating, Sector, Country, ESGRating
import abc


class Coupon:
    def __init__(
        self,
        accrual_start: _Union[date, datetime],
        accrual_end: _Union[date, datetime],
        payment_date: _Union[date, datetime],
        day_count_convention: _Union[DayCounterType, str],
        annualised_fixed_coupon: float,
        fixing_date: _Union[date, datetime],
        floating_period_start: _Union[date, datetime],
        floating_period_end: _Union[date, datetime],
        floating_spread: float = 0.0,
        floating_rate_cap: float = 1e10,
        floating_rate_floor: float = -1e10,
        floating_reference_index: str = "dummy_reference_index",
        amortisation_factor: float = 1.0,
    ):
        # accrual start and end date as well as payment date
        if _is_chronological(accrual_start, [accrual_end], payment_date):
            self.__accrual_start = accrual_start
            self.__accrual_end = accrual_end
            self.__payment_date = payment_date

        self.__day_count_convention = DayCounterType.to_string(day_count_convention)

        self.__annualised_fixed_coupon = _check_positivity(annualised_fixed_coupon)

        self.__fixing_date = _date_to_datetime(fixing_date)

        # spread on floating rate
        self.__spread = floating_spread

        # cap/floor on floating rate
        self.__floating_rate_floor, self.__floating_rate_cap = _check_relation(floating_rate_floor, floating_rate_cap)

        # reference index for fixing floating rates
        if floating_reference_index == "":
            # do not leave reference index empty as this causes pricer to ignore floating rate coupons!
            self.floating_reference_index = "dummy_reference_index"
        else:
            self.__floating_reference_index = floating_reference_index
        self.__amortisation_factor = _check_positivity(amortisation_factor)


class Issuer(interfaces.FactoryObject):
    def __init__(
        self, obj_id: str, name: str, rating: _Union[Rating, str], esg_rating: _Union[ESGRating, str], country: _Union[Country, str], sector: Sector
    ):
        self.__obj_id = obj_id
        self.__name = name
        self.__rating = Rating.to_string(rating)
        self.__esg_rating = ESGRating.to_string(esg_rating)
        self.__country = Country.to_string(country)
        self.__sector = Sector.to_string(sector)

    @staticmethod
    def _create_sample(
        n_samples: int,
        seed: int = None,
        issuer: List[str] = None,
        rating_probs: np.ndarray = None,
        country_probs: np.ndarray = None,
        sector_probs: np.ndarray = None,
        esg_rating_probs: np.ndarray = None,
    ) -> List:
        """Just sample some test data

        Args:
            n_samples (int): Number of samples.
            seed (int, optional): If set, the seed is set, if None, no seed is explicitely set. Defaults to None.
            issuer (List[str], optional): List of issuer names chosen from. If None, a unqiue name for each samples is generated. Defaults to None.
            rating_probs (np.ndarray): Numpy array defining the probability for each rating (ratings ordererd from AAA (first) to D (last array element)). If None, all ratings are chosen with equal probabilities.
        Raises:
            Exception: _description_

        Returns:
            List: List of sampled issuers.
        """
        if seed is not None:
            np.random.seed(seed)
        result = []
        ratings = list(Rating)
        if rating_probs is not None:
            if len(ratings) != rating_probs.shape[0]:
                raise Exception("Number of rating probabilities must equal number of ratings")
        else:
            rating_probs = np.ones(
                (
                    len(
                        ratings,
                    )
                )
            ) / len(ratings)

        if country_probs is not None:
            if len(Country) != country_probs.shape[0]:
                raise Exception("Number of country probabilities must equal number of countries")
        else:
            country_probs = np.ones(
                (
                    len(
                        Country,
                    )
                )
            ) / len(Country)

        if sector_probs is not None:
            if len(Sector) != sector_probs.shape[0]:
                raise Exception("Number of sector probabilities must equal number of sectors")
        else:
            sector_probs = np.ones(
                (
                    len(
                        Sector,
                    )
                )
            ) / len(Sector)

        if esg_rating_probs is not None:
            if len(ESGRating) != esg_rating_probs.shape[0]:
                raise Exception("Number of ESG rating probabilities must equal number of ESG ratings")
        else:
            esg_rating_probs = np.ones(
                (
                    len(
                        ESGRating,
                    )
                )
            ) / len(ESGRating)

        esg_ratings = list(ESGRating)
        sectors = list(Sector)
        country = list(Country)
        if issuer is None:
            issuer = ["Issuer_" + str(i) for i in range(n_samples)]
        elif (n_samples is not None) and (n_samples != len(issuer)):
            raise Exception("Cannot create data since length of issuer list does not equal number of samples. Set n_namples to None.")
        for i in range(n_samples):
            result.append(
                Issuer(
                    "Issuer_" + str(i),
                    issuer[i],
                    np.random.choice(ratings, p=rating_probs),
                    np.random.choice(esg_ratings, p=esg_rating_probs),
                    np.random.choice(country, p=country_probs),
                    np.random.choice(sectors, p=sector_probs),
                )
            )
        return result

    def _to_dict(self) -> dict:
        return {
            "obj_id": self.obj_id,
            "name": self.name,
            "rating": self.rating,
            "esg_rating": self.esg_rating,
            "country": self.country,
            "sector": self.sector,
        }

    @property
    def obj_id(self) -> str:
        """
        Getter for issuer id.

        Returns:
            str: Issuer id.
        """
        return self.__obj_id

    @property
    def name(self) -> str:
        """
        Getter for issuer name.

        Returns:
            str: Issuer name.
        """
        return self.__name

    @property
    def rating(self) -> str:
        """
        Getter for issuer's rating.

        Returns:
            Rating: Issuer's rating.
        """
        return self.__rating

    @rating.setter
    def rating(self, rating: _Union[Rating, str]):
        """
        Setter for issuer's rating.

        Args:
            rating: Rating of issuer.
        """
        self.__rating = Rating.to_string(rating)

    @property
    def esg_rating(self) -> str:
        """
        Getter for issuer's rating.

        Returns:
            Rating: Issuer's rating.
        """
        return self.__esg_rating

    @esg_rating.setter
    def esg_rating(self, esg_rating: _Union[ESGRating, str]):
        """
        Setter for issuer's rating.

        Args:
            rating: Rating of issuer.
        """
        self.__esg_rating = ESGRating.to_string(esg_rating)

    @property
    def country(self) -> str:
        """
        Getter for issuer's country.

        Returns:
            Country: Issuer's country.
        """
        return self.__country

    @property
    def sector(self) -> str:
        """
        Getter for issuer's sector.

        Returns:
            Sector: Issuer's sector.
        """
        return self.__sector

    @sector.setter
    def sector(self, sector: _Union[Sector, str]) -> str:
        """
        Setter for issuer's sector.

        Returns:
            Sector: Issuer's sector.
        """
        self.__sector = Sector.to_string(sector)


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

    @staticmethod
    def _create_sample(n_samples: int, seed: int = None):
        """Creates a sample of random ``CashFlow`` objects.

        Returns:
            List[CashFlow]: List of sampled ``CashFlow`` objects
        """
        result = []
        if seed is not None:
            np.random.seed(seed)

        for i in range(n_samples):
            cashflow_val = np.random.choice(np.arange(1000, 10000, 100), 1)[0]
            result.append(
                {
                    "val": cashflow_val,
                }
            )

    def _to_dict(self) -> dict:
        result = {
            "val": self.val,
            "attributes": self._attributes,
        }
        return result


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
