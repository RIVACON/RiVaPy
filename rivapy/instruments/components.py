# -*- coding: utf-8 -*-
from typing import Union as _Union, List
import numpy as np
from datetime import datetime, date
import rivapy.tools.interfaces as interfaces
from rivapy.tools.datetools import _date_to_datetime
from rivapy.tools._validators import _check_positivity, _check_relation, _is_chronological
from rivapy.tools.enums import DayCounterType, Rating, Sector, Country, ESGRating


class Coupon:
    def __init__(self,
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
                 floating_reference_index: str = 'dummy_reference_index',
                 amortisation_factor: float = 1.0):
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
        if floating_reference_index == '':
            # do not leave reference index empty as this causes pricer to ignore floating rate coupons!
            self.floating_reference_index = 'dummy_reference_index'
        else:
            self.__floating_reference_index = floating_reference_index
        self.__amortisation_factor = _check_positivity(amortisation_factor)


class Issuer(interfaces.FactoryObject):
    def __init__(self,
                 obj_id: str,
                 name: str,
                 rating: _Union[Rating, str],
                 esg_rating: _Union[ESGRating, str],
                 country: _Union[Country, str],
                 sector: Sector):
        self.__obj_id = obj_id
        self.__name = name
        self.__rating = Rating.to_string(rating)
        self.__esg_rating = ESGRating.to_string(esg_rating)
        self.__country = Country.to_string(country)
        self.__sector = Sector.to_string(sector)

    @staticmethod
    def _create_sample(n_samples: int, seed: int = None, issuer: List[str] = None, 
                    rating_probs: np.ndarray = None, 
                    country_probs:np.ndarray = None,
                    sector_probs:np.ndarray = None,
                    esg_rating_probs:np.ndarray=None )->List:
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
                raise Exception('Number of rating probabilities must equal number of ratings')
        else:
            rating_probs = np.ones((len(ratings,)))/len(ratings)

        if country_probs is not None:
            if len(Country) != country_probs.shape[0]:
                raise Exception('Number of country probabilities must equal number of countries')
        else:
            country_probs = np.ones((len(Country,)))/len(Country)

        if sector_probs is not None:
            if len(Sector) != sector_probs.shape[0]:
                raise Exception('Number of sector probabilities must equal number of sectors')
        else:
            sector_probs = np.ones((len(Sector,)))/len(Sector)

        if esg_rating_probs is not None:
            if len(ESGRating) != esg_rating_probs.shape[0]:
                raise Exception('Number of ESG rating probabilities must equal number of ESG ratings')
        else:
            esg_rating_probs = np.ones((len(ESGRating,)))/len(ESGRating)

        
            
        esg_ratings = list(ESGRating)
        sectors = list(Sector)
        country = list(Country)
        if issuer is None:
            issuer = ['Issuer_'+str(i) for i in range(n_samples)]
        elif (n_samples is not None) and (n_samples !=  len(issuer)):
            raise Exception('Cannot create data since length of issuer list does not equal number of samples. Set n_namples to None.')
        for i in range(n_samples):
            result.append(Issuer('Issuer_'+str(i), issuer[i],
                        np.random.choice(ratings, p=rating_probs), 
                        np.random.choice(esg_ratings, p=esg_rating_probs), 
                        np.random.choice(country, p=country_probs),
                        np.random.choice(sectors, p=sector_probs)))
        return result

    def _to_dict(self) -> dict:
        return {'obj_id': self.obj_id, 
                'name': self.name, 'rating': self.rating,
                'esg_rating': self.esg_rating, 
                'country': self.country, 'sector': self.sector}

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
        self.__rating =Rating.to_string(rating)

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
    def sector(self, sector:_Union[Sector, str]) -> str:
        """
        Setter for issuer's sector.

        Returns:
            Sector: Issuer's sector.
        """
        self.__sector = Sector.to_string(sector)

