from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Union
from datetime import date, datetime
from rivapy.tools.enums import Currency, DayCounterType, RollConvention
from rivapy.tools._validators import _check_positivity
from rivapy.tools.datetools import _date_to_datetime


class IrSwapLegSpecification(ABC):
    def __init__(
        self,
        notional_structure,
        start_dates: List[datetime],
        end_dates: List[datetime],
        pay_dates: List[datetime],
        currency: Union[Currency, str],
        day_count: Union[DayCounterType, str],
    ):
        self._notional_structure = notional_structure
        self._start_dates = start_dates
        self._end_dates = end_dates
        self._pay_dates = pay_dates
        self._currency = Currency.to_string(currency)
        self._day_counter = DayCounterType.to_string(day_count)

    @property
    @abstractmethod
    def leg_type(self) -> str:
        pass

    @property
    def notional_structure(self):
        return self._notional_structure

    @property
    def start_dates(self) -> List[datetime]:
        return self._start_dates

    @property
    def end_dates(self) -> List[datetime]:
        return self._end_dates

    @property
    def pay_dates(self) -> List[datetime]:
        return self._pay_dates

    @abstractmethod
    def reset_dates(self) -> List[datetime]:
        pass

    @abstractmethod
    def udl_id(self) -> str:
        pass


class IrFixedLegSpecification(IrSwapLegSpecification):
    def __init__(
        self,
        fixed_rate: float,
        notional_structure,
        start_dates: List[datetime],
        end_dates: List[datetime],
        pay_dates: List[datetime],
        currency: Union[Currency, str],
        day_count: Union[DayCounterType, str],
    ):
        super().__init__(notional_structure, start_dates, end_dates, pay_dates, currency, day_count)
        self._fixed_rate = _check_positivity(fixed_rate)

    @property
    def leg_type(self) -> str:
        return IrLegType.FIXED

    @property
    def fixed_rate(self) -> float:
        return self._fixed_rate

    @property
    def reset_dates(self) -> List[datetime]:
        return self._start_dates

    @property
    def udl_id(self) -> str:
        return ""  # fixed leg has no underlying


class IrFloatLegSpecification(IrSwapLegSpecification):
    def __init__(
        self,
        notional_structure,
        reset_dates: List[datetime],
        start_dates: List[datetime],
        end_dates: List[datetime],
        rate_start_dates: List[datetime],
        rate_end_dates: List[datetime],
        pay_dates: List[datetime],
        currency: Union[Currency, str],
        udl_id: str,
        fixing_id: str,
        day_count: Union[DayCounterType, str],
        rate_day_count: Union[DayCounterType, str],
        spread: float = 0.0,
    ):
        super().__init__(notional_structure, start_dates, end_dates, pay_dates, currency, day_count)
        self._reset_dates = reset_dates
        self._rate_start_dates = rate_start_dates
        self._rate_end_dates = rate_end_dates
        self._spread = spread
        self._udl_id = udl_id
        self._fixing_id = fixing_id
        self._rate_day_count = DayCounterType.to_string(rate_day_count)

    @property
    def leg_type(self) -> str:
        return IrLegType.FLOAT

    @property
    def reset_dates(self) -> List[datetime]:
        return self._reset_dates

    @property
    def udl_id(self) -> str:
        return self._udl_id

    @property
    def fixing_id(self) -> str:
        return self._fixing_id

    @property
    def spread(self) -> float:
        return self._spread

    @property
    def rate_day_count(self) -> str:
        return self._rate_day_count

    @property
    def rate_start_dates(self) -> List[datetime]:
        return self._rate_start_dates

    @property
    def rate_end_dates(self) -> List[datetime]:
        return self._rate_end_dates

    def get_underlyings(self) -> Dict[str, str]:
        return {self._udl_id: self._fixing_id}


class InterestRateBasisSwapSpecification:
    def __init__(
        self,
        id: str,
        issuer: str,
        sec_lvl,
        currency,
        expiry: datetime,
        receive_leg,
        pay_leg,
        spread_leg,
        holidays: str = "",
        ex_settle: int = 0,
        trade_settle: int = 0,
    ):
        """
        General basis swap specification: 2 floating legs + 1 fixed spread leg.
        """
        self.id = id
        self.issuer = issuer
        self.securitization_level = sec_lvl
        self.currency = currency
        self.expiry = expiry
        self.receive_leg = receive_leg
        self.pay_leg = pay_leg
        self.spread_leg = spread_leg
        self.holidays = holidays
        self.ex_settle = ex_settle
        self.trade_settle = trade_settle

    @staticmethod
    def make_specification(
        id: str,
        issuer: str,
        sec_lvl,
        currency,
        trade_date: date,
        spot_days: int,
        maturity,
        notional: float,
        spread_rate: float,
        pay_udl_id: str,
        pay_fixing_id: str,
        receive_udl_id: str,
        receive_fixing_id: str,
        holiday_calendar,
        spread_freq,
        spread_dcc,
        spread_roll,
        pay_freq,
        pay_dcc,
        pay_roll,
        pay_rate_freq,
        pay_rate_dcc,
        pay_rate_roll,
        receive_freq,
        receive_dcc,
        receive_roll,
        receive_rate_freq,
        receive_rate_dcc,
        receive_rate_roll,
    ):
        """
        Factory method to construct a basis swap specification.
        """

        start_date = holiday_calendar.add_business_days(trade_date, spot_days)

        spread_leg = IrFixedLegSpecification.make_specification(
            start_date, maturity, notional, currency, spread_rate, holiday_calendar, spread_freq, spread_dcc, spread_roll
        )

        pay_leg = IrFloatLegSpecification.make_specification(
            start_date,
            maturity,
            notional,
            currency,
            pay_udl_id,
            pay_fixing_id,
            holiday_calendar,
            holiday_calendar,
            pay_freq,
            pay_dcc,
            pay_roll,
            pay_rate_freq,
            pay_rate_dcc,
            pay_rate_roll,
        )

        receive_leg = IrFloatLegSpecification.make_specification(
            start_date,
            maturity,
            notional,
            currency,
            receive_udl_id,
            receive_fixing_id,
            holiday_calendar,
            holiday_calendar,
            receive_freq,
            receive_dcc,
            receive_roll,
            receive_rate_freq,
            receive_rate_dcc,
            receive_rate_roll,
        )

        # Determine the latest end date among the legs
        expiry = max(max(pay_leg.get_end_dates()), max(receive_leg.get_end_dates()), max(spread_leg.get_end_dates()))

        return InterestRateBasisSwapSpecification(id, issuer, sec_lvl, currency, expiry, receive_leg, pay_leg, spread_leg)

    def get_pay_leg(self):
        return self.pay_leg

    def get_receive_leg(self):
        return self.receive_leg

    def get_spread_leg(self):
        return self.spread_leg
