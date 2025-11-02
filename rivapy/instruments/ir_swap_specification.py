from abc import abstractmethod as _abstractmethod
from locale import currency
from typing import List as _List, Union as _Union, Tuple, Dict
import numpy as np
from datetime import datetime, date, timedelta
from rivapy.tools.holidays_compat import HolidayBase as _HolidayBase, ECB as _ECB
from rivapy.instruments._logger import logger

from rivapy.tools.datetools import Period, Schedule, _date_to_datetime, _datetime_to_date_list, _term_to_period, serialize_date
from rivapy.tools.enums import DayCounterType, RollConvention, SecuritizationLevel, Currency, Rating, Instrument
from rivapy.tools._validators import _check_positivity, _check_start_before_end, _string_to_calendar, _is_ascending_date_list
import rivapy.tools.interfaces as interfaces
from rivapy.tools.datetools import Period, Schedule

from rivapy.instruments.bond_specifications import BondBaseSpecification
from rivapy.instruments.components import NotionalStructure, ConstNotionalStructure, VariableNotionalStructure  # , ResettingNotionalStructure
from rivapy.tools.enums import IrLegType

# Base each swap leg, off of the IRSwapBaseSpecification
# This IRSwapBaseSpecification is in turn, based off of the BondBaseSpecification
# Can think about basing the float/fixed leg off of the BondFlaoting/Fixed Note class...

# WIP


class IrSwapLegSpecification(interfaces.FactoryObject):
    """Base interest rate swap leg specification used to define both fixed and floating legs."""

    def __init__(
        self,
        obj_id: str,
        notional: _Union[float, NotionalStructure],
        start_dates: _List[datetime],
        end_dates: _List[datetime],
        pay_dates: _List[datetime],
        currency: _Union[Currency, str],
        day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
    ):
        """Constructor for the IrSwapLegSpecification class used to define both fixed and floating legs.

        Args:
            obj_id (str): obj ID for the instrument.
            notional (_Union[float, NotionalStructure]): If given a singular float, will convert to ConstNotinalStructure. Contains the notional information.
            start_dates (_List[datetime]): start date of interest accrual period.
            end_dates (_List[datetime]): end date of interest accrual period.
            pay_dates (_List[datetime]): Dates when both legs of the swap exchange cash flows.
            currency (_Union[Currency, str]): The currency of the swap
            day_count_convention (_Union[DayCounterType, str], optional): The day count convention used. Defaults to DayCounterType.ThirtyU360.
        """
        self.obj_id = obj_id
        self.notional_structure = notional
        self.start_dates = start_dates
        self.end_dates = end_dates
        self.pay_dates = pay_dates
        self.currency = Currency.to_string(currency)
        self.day_count_convention = day_count_convention

    @property
    def currency(self) -> str:
        """The swap leg's currency as a string."""
        return self._currency

    @currency.setter
    def currency(self, value: _Union[Currency, str]):
        self._currency = Currency.to_string(value)

    @property
    def start_dates(self) -> _List[datetime]:
        """start dates for the interest periods

        Returns:
            _List[datetime]: _description_
        """
        return self._start_dates

    @start_dates.setter
    def start_dates(self, value: _List[datetime]):
        self._start_dates = value

    @property
    def end_dates(self) -> _List[datetime]:
        """end dates for the interest periods

        Returns:
            _List[datetime]: _description_
        """
        return self._end_dates

    @end_dates.setter
    def end_dates(self, value: _List[datetime]):
        self._end_dates = value

    @property
    def pay_dates(self) -> _List[datetime]:
        """pay dates for the interest periods

        Returns:
            _List[datetime]:
        """
        return self._pay_dates

    @pay_dates.setter
    def pay_dates(self, value: _List[datetime]):
        self._pay_dates = value

    @property
    def notional_structure(self) -> NotionalStructure:
        """Return the notionals

        Returns:
            NotionalStructure: class object detailing the notionals, start dates, ...
        """
        return self._notional_structure

    @notional_structure.setter
    def notional_structure(self, value: _Union[float, NotionalStructure]):
        """If only a float is given, assume a constant notional and create a ConstNotionalStructure.

        Args:
            value (_Union[float, NotionalStructure]):
        """
        if isinstance(value, (int, float)):
            self._notional_structure = ConstNotionalStructure(float(value))
        else:
            self._notional_structure = value

    # @abstractmethod
    # def reset_dates(self) -> _List[datetime]:
    #    """ #TODO brought over from pyvacon, for float leg?
    #    """
    #    pass

    def _to_dict(self) -> Dict:
        return_dict = {
            "obj_id": self.obj_id,
            "notional": self.notional_structure,
            "start_dates": self.start_dates,
            "end_dates": self.end_dates,
            "pay_dates": self.pay_dates,
            "currency": self.currency,
            "day_count_convention": self.day_count_convention,
        }
        return return_dict

    @staticmethod
    def _create_sample(n_samples: int, seed: int = None):
        pass  # TODO


class IrFixedLegSpecification(IrSwapLegSpecification):
    """Specification for a fixed leg for an interest rate swap."""

    def __init__(
        self,
        fixed_rate: float,
        obj_id: str,
        notional: _Union[float, NotionalStructure],
        start_dates: _List[datetime],
        end_dates: _List[datetime],
        pay_dates: _List[datetime],
        currency: _Union[Currency, str],
        day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
    ):
        """Constructor for a fixed leg of an interest rate swap.

        Args:
            fixed_rate (float): The fixed interest rate defining this leg of the swap.
            obj_id (str): obj ID for the instrument.
            notional (_Union[float, NotionalStructure]): If given a singular float, will convert to ConstNotinalStructure. Contains the notional information.
            start_dates (_List[datetime]): start date of interest accrual period.
            end_dates (_List[datetime]): end date of interest accrual period.
            pay_dates (_List[datetime]): Dates when both legs of the swap exchange cash flows.
            currency (_Union[Currency, str]): The currency of the swap
            day_count_convention (_Union[DayCounterType, str], optional): The day count convention used. Defaults to DayCounterType.ThirtyU360.
        """
        super().__init__(obj_id, notional, start_dates, end_dates, pay_dates, currency, day_count_convention)
        self.fixed_rate = fixed_rate  # _check_positivity(fixed_rate) # TODO is there a need for it to be always positive?

    # region properties
    @property
    def leg_type(self) -> str:
        return IrLegType.FIXED

    @property
    def fixed_rate(self) -> float:
        return self._fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, value: float):
        self._fixed_rate = value

    # @property
    # def reset_dates(self) -> _List[datetime]:
    #    return self.start_dates

    @property
    def udl_id(self) -> str:
        return ""  # fixed leg has no underlying

    def _to_dict(self):

        return_dict = super()._to_dict()
        return_dict["fixed_rate"] = self.fixed_rate
        return return_dict

    @staticmethod
    def _create_sample(n_samples: int, seed: int = None):
        pass  # TODO

    # endregion

    def get_NotionalStructure(self):

        return self.notional_structure


class IrFloatLegSpecification(IrSwapLegSpecification):
    def __init__(
        self,
        obj_id: str,
        notional: _Union[float, NotionalStructure],
        reset_dates: _List[datetime],
        start_dates: _List[datetime],
        end_dates: _List[datetime],
        rate_start_dates: _List[datetime],  # are these needed here? or are they obtained from the underlying
        rate_end_dates: _List[datetime],
        pay_dates: _List[datetime],
        currency: _Union[Currency, str],
        udl_id: str,
        fixing_id: str,
        day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
        rate_day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
        spread: float = 0.0,
    ):
        """Constructor for a floating leg of an interest rate swap.

        Args:
            obj_id (str): obj ID for the instrument.
            notional (_Union[float, NotionalStructure]): If given a singular float, will convert to ConstNotinalStructure. Contains the notional information.
            reset_dates (_List[datetime]): Date on which the floating rate (e.g., SOFR, LIBOR) is determined
            start_dates (_List[datetime]): Date the entire swap begins (effective date)
            end_dates (_List[datetime]): Date the swap matures
            rate_start_dates (_List[datetime]): start dates for the determination of the underlying rate
            rate_end_dates (_List[datetime]): end dates for the determination of the underlying rate
            pay_dates (_List[datetime]): Dates when both legs of the swap exchange cash flows.
            currency (_Union[Currency, str]): The currency of the swap
            udl_id (str): ID of the underlying rate
            fixing_id (str): fixing id
            day_count_convention (_Union[DayCounterType, str], optional): The day count convention used.. Defaults to DayCounterType.ThirtyU360.
            rate_day_count_convention (_Union[DayCounterType, str], optional): The day count convention used for the underlying
                                                . Defaults to DayCounterType.ThirtyU360.
            spread (float, optional): Defaults to 0.0.
        """
        super().__init__(obj_id, notional, start_dates, end_dates, pay_dates, currency, day_count_convention)
        self.reset_dates = reset_dates  # TODO: ADD setters to get rid of error notification?
        self.rate_start_dates = rate_start_dates
        self.rate_end_dates = rate_end_dates
        self._spread = spread
        self.udl_id = udl_id
        self.fixing_id = fixing_id
        self.rate_day_count_convention = DayCounterType.to_string(rate_day_count_convention)

    # TODO to_dict method

    # region properties
    @property
    def leg_type(self) -> str:
        return IrLegType.FLOAT

    @property
    def reset_dates(self) -> _List[datetime]:
        return self._reset_dates

    @reset_dates.setter
    def reset_dates(self, value):
        self._reset_dates = value

    @property
    def udl_id(self) -> str:
        return self._udl_id

    @udl_id.setter
    def udl_id(self, value):
        self._udl_id = value

    @property
    def fixing_id(self) -> str:
        return self._fixing_id

    @fixing_id.setter
    def fixing_id(self, value):
        self._fixing_id = value

    @property
    def spread(self) -> float:
        return self._spread

    @spread.setter
    def spread(self, value):
        self._spread = value

    @property
    def rate_day_count(self) -> str:
        return self.rate_day_count

    @property
    def rate_start_dates(self) -> _List[datetime]:
        return self._rate_start_dates

    @rate_start_dates.setter
    def rate_start_dates(self, value):
        self._rate_start_dates = value

    @property
    def rate_end_dates(self) -> _List[datetime]:
        return self._rate_end_dates

    @rate_end_dates.setter
    def rate_end_dates(self, value):
        self._rate_end_dates = value

    def get_underlyings(self) -> Dict[str, str]:
        return {self.udl_id: self.fixing_id}

    # endregion

    def get_NotionalStructure(self):

        return self.notional_structure


class IrOISLegSpecification(IrSwapLegSpecification):
    def __init__(
        self,
        obj_id: str,
        notional: _Union[float, NotionalStructure],
        rate_reset_dates: _List[datetime],
        start_dates: _List[datetime],
        end_dates: _List[datetime],
        rate_start_dates: _List[datetime],  # are these needed here? or are they obtained from the underlying
        rate_end_dates: _List[datetime],
        pay_dates: _List[datetime],
        currency: _Union[Currency, str],
        udl_id: str,
        fixing_id: str,
        day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
        rate_day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
        spread: float = 0.0,
    ):

        # fixingLag?
        # payLag?
        # freq?
        # rate currency vs notional currency?
        # leg holidays, and rate holidays - again to be dealt with in the scheduler?
        # FYI scheduler needs, start date, end date, freq, roll convention, holiday...
        # for the reset dates, will need fixing lag
        # for the pay dates generation, will need payLag

        """Constructor for a floating leg of an overnight index swap.

        Args:
            obj_id (str): obj ID for the instrument.
            notional (_Union[float, NotionalStructure]): If given a singular float, will convert to ConstNotinalStructure. Contains the notional information.
            reset_dates (_List[datetime]): Date on which the floating rate (e.g., SOFR, LIBOR) is determined
            start_dates (_List[datetime]): Date the entire swap begins (effective date)
            end_dates (_List[datetime]): Date the swap matures
            rate_start_dates (_List[datetime]): start dates for the determination of the underlying rate
            rate_end_dates (_List[datetime]): end dates for the determination of the underlying rate
            pay_dates (_List[datetime]): Dates when both legs of the swap exchange cash flows.
            currency (_Union[Currency, str]): The currency of the swap
            udl_id (str): ID of the underlying rate
            fixing_id (str): fixing id
            day_count_convention (_Union[DayCounterType, str], optional): The day count convention used.. Defaults to DayCounterType.ThirtyU360.
            rate_day_count_convention (_Union[DayCounterType, str], optional): The day count convention used for the underlying
                                                . Defaults to DayCounterType.ThirtyU360.
            spread (float, optional): . Defaults to 0.0.
        """
        super().__init__(obj_id, notional, start_dates, end_dates, pay_dates, currency, day_count_convention)
        self.rate_reset_dates = rate_reset_dates
        self.rate_start_dates = rate_start_dates
        self.rate_end_dates = rate_end_dates
        self._spread = spread
        self.udl_id = udl_id
        self.fixing_id = fixing_id
        self.rate_day_count_convention = DayCounterType.to_string(rate_day_count_convention)

    # TODO to_dict method

    # region properties
    @property
    def leg_type(self) -> str:
        return IrLegType.OIS

    @property
    def reset_dates(self) -> _List[datetime]:
        return self._rate_reset_dates

    @reset_dates.setter
    def rate_reset_dates(self, value):
        self._rate_reset_dates = value

    @property
    def udl_id(self) -> str:
        return self._udl_id

    @udl_id.setter
    def udl_id(self, value):
        self._udl_id = value

    @property
    def fixing_id(self) -> str:
        return self._fixing_id

    @fixing_id.setter
    def fixing_id(self, value):
        self._fixing_id = value

    @property
    def spread(self) -> float:
        return self._spread

    @spread.setter
    def spread(self, value):
        self._spread = value

    @property
    def rate_day_count(self) -> str:
        return self.rate_day_count

    @property
    def rate_start_dates(self) -> _List[datetime]:
        return self._rate_start_dates

    @rate_start_dates.setter
    def rate_start_dates(self, value):
        self._rate_start_dates = value

    @property
    def rate_end_dates(self) -> _List[datetime]:
        return self._rate_end_dates

    @rate_end_dates.setter
    def rate_end_dates(self, value):
        self._rate_end_dates = value

    def get_underlyings(self) -> Dict[str, str]:
        return {self.udl_id: self.fixing_id}

    # endregion

    def get_NotionalStructure(self):

        return self.notional_structure

    @staticmethod
    def ois_scheduler_2D(start_dates: _List[datetime], end_dates: _List[datetime]):
        """The OIS makeshift scheduler to account for expected 2D array structure of
            >>> rivapy.instruments.interest_rate_swap_pricer.populate_cashflows_ois
            Alternatively, the rivapy Scheduler class is used.
            This is a simplified version for testing purposes.
        Args:
            start_dates (_List[datetime]): start dates for a given accrual period, containing the daily start rates inside that accrual period
            end_dates (_List[datetime]): _description_

        Returns:
            _type_: Multi-dimensional lists of datetimes
        """

        daily_rate_start_dates = []  # 2D list: coupon i -> list of daily starts
        daily_rate_end_dates = []  # 2D list: coupon i -> list of daily ends
        daily_rate_reset_dates = []  # 2D list: coupon i -> list of reset dates
        pay_dates = []  # 1D list: one pay date per coupon

        for i in range(len(start_dates)):

            # for this test we keep it simple and ignore conventions e.g. business day or so. i.e just take every day
            num_days = (end_dates[i] - start_dates[i]).days
            daily_schedule = [start_dates[i] + timedelta(days=j) for j in range(num_days)]

            # Build start/end date pairs for accrual periods
            starts = daily_schedule[:-1]  # all except last
            ends = daily_schedule[1:]  # all except first

            daily_rate_start_dates.append(starts)
            daily_rate_end_dates.append(ends)

            # 4. Compute reset dates (fixing lag applied to each start)
            # resets = [add_business_days(start, fixingLag, rateHolidays)
            #           for start in starts]
            # assume simple case reset date is the same as start date
            resets = starts  # reset dates are equal to start dates if spot lag is 0.
            daily_rate_reset_dates.append(resets)

            # Compute payment date for the coupon
            # pay_date = add_business_days(end_dates[i], payLag, holidays)
            # assume simple case, pay date is end date
            pay_date = end_dates[i]
            pay_dates.append(pay_date)

        return [daily_rate_start_dates, daily_rate_end_dates, daily_rate_reset_dates, pay_dates]


class InterestRateSwapSpecification(interfaces.FactoryObject):

    def __init__(
        self,
        obj_id: str,
        notional: _Union[float, NotionalStructure],
        issue_date: _Union[date, datetime],
        maturity_date: _Union[date, datetime],
        pay_leg: _Union[IrFixedLegSpecification, IrFloatLegSpecification, IrOISLegSpecification],
        receive_leg: _Union[IrFixedLegSpecification, IrFloatLegSpecification, IrOISLegSpecification],
        currency: _Union[Currency, str] = "EUR",
        calendar: _Union[_HolidayBase, str] = None,
        day_count_convention: _Union[DayCounterType, str] = DayCounterType.ThirtyU360,
        business_day_convention: _Union[RollConvention, str] = RollConvention.FOLLOWING,
        issuer: str = None,
        securitization_level: _Union[SecuritizationLevel, str] = SecuritizationLevel.NONE,
        rating: _Union[Rating, str] = Rating.NONE,
    ):
        """Specification of the entire swap, encapsulating both the pay leg and the receive leg.

        Args:
            obj_id (str): _description_
            notional (_Union[float, NotionalStructure]): _description_
            issue_date (_Union[date, datetime]): _description_
            maturity_date (_Union[date, datetime]): _description_
            pay_leg (_Union[IrFixedLegSpecification, IrFloatLegSpecification]): _description_
            receive_leg (_Union[IrFixedLegSpecification, IrFloatLegSpecification]): _description_
            currency (_Union[Currency, str], optional): _description_. Defaults to "EUR".
            calendar (_Union[_HolidayBase, str], optional): _description_. Defaults to None.
            day_count_convention (_Union[DayCounterType, str], optional): _description_. Defaults to DayCounterType.ThirtyU360.
            business_day_convention (_Union[RollConvention, str], optional): _description_. Defaults to RollConvention.FOLLOWING.
            issuer (str, optional): _description_. Defaults to None.
            securitization_level (_Union[SecuritizationLevel, str], optional): _description_. Defaults to SecuritizationLevel.NONE.
            rating (_Union[Rating, str], optional): _description_. Defaults to Rating.NONE.
        """
        self.obj_id = obj_id
        if issuer is not None:
            self.issuer = issuer
        if securitization_level is not None:
            self.securitization_level = securitization_level
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.currency = currency
        self.notional_structure = notional
        self.rating = Rating.to_string(rating)
        # validate dates
        self._validate_derived_issued_instrument()
        self.pay_leg = pay_leg
        self.receive_leg = receive_leg
        self.day_count_convention = day_count_convention
        self.business_day_convention = RollConvention.to_string(business_day_convention)
        if calendar is None:
            self.calendar = _ECB(years=range(issue_date.year, maturity_date.year + 1))
        else:
            self.calendar = _string_to_calendar(calendar)

    @staticmethod  # TODO
    def _create_sample(
        n_samples: int, seed: int = None, ref_date=None, issuers: _List[str] = None, sec_levels: _List[str] = None, currencies: _List[str] = None
    ) -> _List["InterestRateSwapSpecification"]:
        if seed is not None:
            np.random.seed(seed)
        if ref_date is None:
            ref_date = datetime.now()
        else:
            ref_date = _date_to_datetime(ref_date)
        if issuers is None:
            issuers = ["Issuer_" + str(i) for i in range(int(n_samples / 2))]
        result = []
        if currencies is None:
            currencies = list(Currency)
        if sec_levels is None:
            sec_levels = list(SecuritizationLevel)
        for i in range(n_samples):
            days = int(15.0 * 365.0 * np.random.beta(2.0, 2.0)) + 1
            issue_date = ref_date + timedelta(days=np.random.randint(low=-365, high=0))
            result.append(
                InterestRateSwapSpecification(
                    obj_id=f"IRS_{i}",
                    issue_date=issue_date,
                    maturity_date=ref_date + timedelta(days=days),
                    currency=np.random.choice(currencies),
                    notional=np.random.choice([100.0, 1000.0, 10_000.0, 100_0000.0]),
                    issuer=np.random.choice(issuers),
                    securitization_level=np.random.choice(sec_levels),
                )
            )
        return result

    def _validate_derived_issued_instrument(self):
        self.__issue_date, self.__maturity_date = _check_start_before_end(self.__issue_date, self.__maturity_date)

    def _to_dict(self) -> dict:
        result = {
            "obj_id": self.obj_id,
            "issuer": self.issuer,
            "securitization_level": self.securitization_level,
            "issue_date": serialize_date(self.issue_date),
            "maturity_date": serialize_date(self.maturity_date),
            "currency": self.currency,
            "notional": self.notional_structure,
            "rating": self.rating,
            "receive_leg": self.receive_leg,
            "pay_leg": self.pay_leg,
            "calendar": getattr(self.calendar, "name", self.calendar.__class__.__name__),
            "day_count_convention": self.day_count_convention,
            "business_day_convention": self.business_day_convention,
        }
        return result

    # region properties
    @property
    def issuer(self) -> str:
        """
        Getter for instrument's issuer.

        Returns:
            str: Instrument's issuer.
        """
        return self.__issuer

    @issuer.setter
    def issuer(self, issuer: str):
        """
        Setter for instrument's issuer.

        Args:
            issuer(str): Issuer of the instrument.
        """
        self.__issuer = issuer

    @property
    def rating(self) -> str:
        return self.__rating

    @rating.setter
    def rating(self, rating: _Union[Rating, str]) -> str:
        self.__rating = Rating.to_string(rating)

    @property
    def securitization_level(self) -> str:
        """
        Getter for instrument's securitisation level.

        Returns:
            str: Instrument's securitisation level.
        """
        return self.__securitization_level

    @securitization_level.setter
    def securitization_level(self, securitisation_level: _Union[SecuritizationLevel, str]):
        self.__securitization_level = SecuritizationLevel.to_string(securitisation_level)

    @property
    def issue_date(self) -> date:
        """
        Getter for IR swap's issue date.

        Returns:
            date: IR swap's issue date.
        """
        return self.__issue_date

    @issue_date.setter
    def issue_date(self, issue_date: _Union[datetime, date]):
        """
        Setter for IR swap's issue date.

        Args:
            issue_date (Union[datetime, date]): IR swap's issue date.
        """
        self.__issue_date = _date_to_datetime(issue_date)

    @property
    def maturity_date(self) -> date:
        """
        Getter for IR swap's maturity date.

        Returns:
            date: IR swap's maturity date.
        """
        return self.__maturity_date

    @maturity_date.setter
    def maturity_date(self, maturity_date: _Union[datetime, date]):
        """
        Setter for IR swap's maturity date.

        Args:
            maturity_date (Union[datetime, date]): IR swap's maturity date.
        """
        self.__maturity_date = _date_to_datetime(maturity_date)

    @property
    def currency(self) -> str:
        """
        Getter for IR swap's currency.

        Returns:
            str: IR swap's ISO 4217 currency code
        """
        return self.__currency

    @currency.setter
    def currency(self, currency: str):
        self.__currency = Currency.to_string(currency)

    @property
    def notional_structure(self) -> NotionalStructure:
        """Return the notionals

        Returns:
            NotionalStructure: class object detailing the notionals, start dates, ...
        """
        return self._notional_structure

    @notional_structure.setter
    def notional_structure(self, value: _Union[float, NotionalStructure]):
        """If only a float is given, assume a constant notional and create a ConstNotionalStructure.

        Args:
            value (_Union[float, NotionalStructure]): _description_
        """
        if isinstance(value, (int, float)):
            self._notional_structure = ConstNotionalStructure(float(value))
        else:
            self._notional_structure = value

    def get_pay_leg(self):
        return self.pay_leg

    def get_receive_leg(self):
        return self.receive_leg

    def get_fixed_leg(self):
        """get the fixed leg (only possible for fixed vs. floating swaps -> throws otherwise)"""

        pay_leg_is_fixed = self.get_pay_leg().leg_type == IrLegType.FIXED
        receive_leg_is_fixed = self.get_receive_leg().leg_type == IrLegType.FIXED

        if pay_leg_is_fixed and not receive_leg_is_fixed:
            leg = self.get_pay_leg()
        elif receive_leg_is_fixed and not pay_leg_is_fixed:
            leg = self.get_receive_leg()
        else:
            raise ValueError("Swap is not comprised of one fixed leg and one float/OIS leg!")

        return leg

    def get_float_leg(self):
        """get the float leg (only possible for fixed vs. floating swaps -> throws otherwise)"""

        pay_leg_is_float = self.get_pay_leg().leg_type in [IrLegType.FLOAT, IrLegType.OIS]
        receive_leg_is_float = self.get_receive_leg().leg_type in [IrLegType.FLOAT, IrLegType.OIS]

        if pay_leg_is_float and not receive_leg_is_float:
            leg = self.get_pay_leg()
        elif receive_leg_is_float and not pay_leg_is_float:
            leg = self.get_receive_leg()
        else:
            raise ValueError("Swap is not comprised of one fixed leg and one float/OIS leg!")

        return leg

    def ins_type(self):
        """Return instrument type

        Returns:
            Instrument: Interest Rate Swap
        """
        return Instrument.IRS

    # temp placeholder
    def get_end_date(self):
        return self.maturity_date

    # endregion
