import abc
from typing import List, Tuple, TYPE_CHECKING
from rivapy.tools.interfaces import FactoryObject
import datetime as dt
from dateutil.relativedelta import relativedelta

# import rivapy.tools.interfaces as interfaces
from rivapy.tools.enums import SecuritizationLevel, Currency, DayCounterType, RollConvention, RollRule
from typing import List, Tuple, Optional as _Optional, Union as _Union
from rivapy.tools.datetools import Period, _date_to_datetime, _term_to_period, _string_to_calendar, DayCounter, Schedule, roll_day
from holidays import HolidayBase as _HolidayBase
from holidays import EuropeanCentralBank as _ECB
from rivapy.tools._validators import (
    _check_positivity,
    _check_start_before_end,
    _check_start_at_or_before_end,
    _string_to_calendar,
    _is_ascending_date_list,
)

# if TYPE_CHECKING:
# from rivapy.marketdata.curves import DiscountCurve

from rivapy import _pyvacon_available

if _pyvacon_available:
    import pyvacon.finance.specification as _spec

    ComboSpecification = _spec.ComboSpecification
    # Equity/FX
    PayoffStructure = _spec.PayoffStructure
    ExerciseSchedule = _spec.ExerciseSchedule
    BarrierDefinition = _spec.BarrierDefinition
    BarrierSchedule = _spec.BarrierSchedule
    BarrierPayoff = _spec.BarrierPayoff
    BarrierSpecification = _spec.BarrierSpecification
    # EuropeanVanillaSpecification = _spec.EuropeanVanillaSpecification
    # AmericanVanillaSpecification = _spec.AmericanVanillaSpecification
    # RainbowUnderlyingSpec = _spec.RainbowUnderlyingSpec
    # RainbowBarrierSpec = _spec.RainbowBarrierSpec
    LocalVolMonteCarloSpecification = _spec.LocalVolMonteCarloSpecification
    RainbowSpecification = _spec.RainbowSpecification
    # MultiMemoryExpressSpecification = _spec.MultiMemoryExpressSpecification
    # MemoryExpressSpecification = _spec.MemoryExpressSpecification
    ExpressPlusSpecification = _spec.ExpressPlusSpecification
    AsianVanillaSpecification = _spec.AsianVanillaSpecification
    RiskControlStrategy = _spec.RiskControlStrategy
    AsianRiskControlSpecification = _spec.AsianRiskControlSpecification

    # Interest Rates
    IrSwapLegSpecification = _spec.IrSwapLegSpecification
    IrFixedLegSpecification = _spec.IrFixedLegSpecification
    IrFloatLegSpecification = _spec.IrFloatLegSpecification
    InterestRateSwapSpecification = _spec.InterestRateSwapSpecification
    InterestRateBasisSwapSpecification = _spec.InterestRateBasisSwapSpecification
    DepositSpecification = _spec.DepositSpecification
    # 2025.06.30 HN test to run notebook discount_curves
    # ForwardRateAgreementSpecification = _spec.ForwardRateAgreementSpecification
    InterestRateFutureSpecification = _spec.InterestRateFutureSpecification
    # 2025.06.30 HN test to run notebook discount_curves
    # CapSpecification = _spec.CapSpecification
    # 2025.06.30 HN test to run notebook discount_curves
    # SwaptionSpecification = _spec.SwaptionSpecification

    # 2025.06.30 HN test to run notebook discount_curves
    # InflationLinkedBondSpecification = _spec.InflationLinkedBondSpecification
    CallableBondSpecification = _spec.CallableBondSpecification

    # GasStorageSpecification = _spec.GasStorageSpecification

    # ScheduleSpecification = _spec.ScheduleSpecification

    # SpecificationManager = _spec.SpecificationManager

    # Bonds/Credit
    CouponDescription = _spec.CouponDescription
    BondSpecification = _spec.BondSpecification
else:
    # empty placeholder...
    class BondSpecification:
        pass

    class ComboSpecification:
        pass

    class BarrierSpecification:
        pass

    class RainbowSpecification:
        pass

    class MemoryExpressSpecification:
        pass


class EuropeanVanillaSpecification:
    def __init__(
        self,
        id: str,
        type: str,
        expiry: dt,
        strike: float,
        issuer: str = "",
        sec_lvl: str = SecuritizationLevel.COLLATERALIZED,
        curr: str = Currency.EUR,
        udl_id: str = "",
        share_ratio: float = 1.0,
        #  holidays: str = '',
        #  ex_settle: int = 0, not implemented
        #  trade_settle: int = 0 not implemented
    ):
        """Constructor for european vanilla option

        Args:
            id (str): Identifier (name) of the european vanilla specification.
            type (str): Type of the european vanilla option ('PUT','CALL').
            expiry (dt): Expiration date.
            strike (float): Strike price.
            issuer (str, optional): Issuer Id. Must not be set if pricing data is manually defined. Defaults to ''.
            sec_lvl (str, optional): Securitization level. Can be selected from rivapy.enums.SecuritizationLevel. Defaults to SecuritizationLevel.COLLATERALIZED.
            curr (str, optional): Currency (ISO-4217 Code). Must not be set if pricing data is manually defined. Can be selected from rivapy.enums.Currency. Defaults to Currency.EUR.
            udl_id (str, optional): Underlying Id. Must not be set if pricing data is manually defined. Defaults to ''.
            share_ratio (float, optional): Ratio of covered shares of the underlying by a single option contract. Defaults to 1.0.
        """

        self.id = id
        self.issuer = issuer
        self.sec_lvl = sec_lvl
        self.curr = curr
        self.udl_id = udl_id
        self.type = type
        self.expiry = expiry
        self.strike = strike
        self.share_ratio = share_ratio
        # self.holidays = holidays
        # self.ex_settle = ex_settle
        # self.trade_settle = trade_settle

        self._pyvacon_obj = None

    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _spec.EuropeanVanillaSpecification(
                self.id, self.issuer, self.sec_lvl, self.curr, self.udl_id, self.type, self.expiry, self.strike, self.share_ratio, "", 0, 0
            )

        return self._pyvacon_obj


class AmericanVanillaSpecification:
    def __init__(
        self,
        id: str,
        type: str,
        expiry: dt,
        strike: float,
        issuer: str = "",
        sec_lvl: str = SecuritizationLevel.COLLATERALIZED,
        curr: str = Currency.EUR,
        udl_id: str = "",
        share_ratio: float = 1.0,
        exercise_before_ex_date: bool = False,
        #  ,holidays: str
        #  ,ex_settle: str
        #  ,trade_settle: str
    ):
        """Constructor for american vanilla option

        Args:
            id (str): Identifier (name) of the american vanilla specification.
            type (str): Type of the american vanilla option ('PUT','CALL').
            expiry (dt): Expiration date.
            strike (float): Strike price.
            issuer (str, optional): Issuer Id. Must not be set if pricing data is manually defined. Defaults to ''.
            sec_lvl (str, optional): Securitization level. Can be selected from rivapy.enums.SecuritizationLevel. Defaults to SecuritizationLevel.COLLATERALIZED.
            curr (str, optional): Currency (ISO-4217 Code). Must not be set if pricing data is manually defined. Can be selected from rivapy.enums.Currency. Defaults to Currency.EUR.
            udl_id (str, optional): Underlying Id. Must not be set if pricing data is manually defined. Defaults to ''.
            share_ratio (float, optional): Ratio of covered shares of the underlying by a single option contract. Defaults to 1.0.
            exercise_before_ex_date (bool, optional): Indicates if option can be exercised within two days before dividend ex-date. Defaults to False.
        """

        self.id = id
        self.type = type
        self.expiry = expiry
        self.strike = strike
        self.issuer = issuer
        self.sec_lvl = sec_lvl
        self.curr = curr
        self.udl_id = udl_id
        self.share_ratio = share_ratio
        self.exercise_before_ex_date = exercise_before_ex_date
        # self.holidays = holidays
        # self.ex_settle = ex_settle
        # self.trade_settle = trade_settle

        self._pyvacon_obj = None

    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _spec.AmericanVanillaSpecification(
                self.id,
                self.issuer,
                self.sec_lvl,
                self.curr,
                self.udl_id,
                self.type,
                self.expiry,
                self.strike,
                self.share_ratio,
                self.exercise_before_ex_date,
                "",
                0,
                0,
            )

        return self._pyvacon_obj


class HasExpectedCashflows(FactoryObject):
    def __init__(
        self,
        obj_id: str,
        first_fixing_date: _Union[dt.date, dt.datetime],
        start_date: _Union[dt.date, dt.datetime],
        end_date: _Union[dt.date, dt.datetime],
        maturity_date: _Union[dt.date, dt.datetime],
        frequency: _Union[Period, str],
        notional: float = 100.0,
        currency: _Union[Currency, str] = Currency.EUR,
        notional_exchange: bool = True,
        coupon: float = 0.0,
        day_count_convention: _Union[DayCounterType, str] = DayCounterType.ACT360,
        business_day_convention: _Union[RollConvention, str] = RollConvention.MODIFIED_FOLLOWING,
        roll_convention: _Union[RollRule, str] = RollRule.EOM,
        calendar: _Union[_HolidayBase, str] = _ECB(),
        coupon_type: str = "fix",
        settlement_days: int = 0,
        spot_lag: int = 2,
        pays_in_arrears: bool = True,
        issuer: _Optional[str] = None,
        securitization_level: _Union[SecuritizationLevel, str] = SecuritizationLevel.NONE,
        backwards=True,
        stub_type_is_Long=True,
        last_fixing: _Optional[float] = None,
    ):
        """Initializes the HasExpectedCashflows object.

        Args:
            obj_id (str): Unique identifier for the object.
            first_fixing_date (_Union[date, datetime]): Date of the first fixing.
            start_date (_Union[date, datetime]): Start date of the first accrual period.
            end_date (_Union[date, datetime]): End of the last accrual period. Not necessarily a good business day.
            maturity_date (_Union[date, datetime]): Adjusted end date of the last accrual period. Is a good business day.
            notional (float): Notional amount of the instrument.
            coupon (float): Fixed coupon rate .
            frequency (_Union[Period, str]): frequency of fixings.
            day_count_convention (_Union[DayCounterType, str], optional): Day count convention. Defaults to DayCounterType.ACT360.
            business_day_convention (_Union[RollConvention, str], optional): Business day convention. Defaults to RollConvention.MODIFIED_FOLLOWING.
            roll_convention (_Union[RollRule, str], optional): Roll convention. Defaults to RollRule.EOM.
            calendar (_Union[_HolidayBase, str], optional): Holiday calendar. Defaults to _ECB().
            settlement_days (int, optional): Number of settlement days. Defaults to 0.
            notional_exchange (bool, optional): Indicates if notional is exchanged at maturity. Defaults
            pays_in_arrears (bool, optional): Indicates if the instrument pays in arrears. Defaults to True.
            fwd_curve (_Optional[DiscountCurve], optional): Forward curve used for pricing. Defaults to None.
        """
        self._obj_id = obj_id
        self._first_fixing_date = _date_to_datetime(first_fixing_date)
        self._start_date = _date_to_datetime(start_date)
        self._end_date = _date_to_datetime(end_date)
        self._maturity_date = _date_to_datetime(maturity_date)
        self._notional = notional
        self._coupon = coupon
        self._frequency = frequency
        self._day_count_convention = day_count_convention
        self._business_day_convention = business_day_convention
        self._roll_convention = roll_convention
        self._calendar = calendar
        self._coupon_type = coupon_type
        self._notional_exchange = notional_exchange
        self._settlement_days = settlement_days
        self._spot_days = spot_lag
        self._pays_in_arrears = pays_in_arrears
        self._currency = Currency.to_string(currency)
        if issuer is not None:
            self._issuer = issuer
        if securitization_level is not None:
            self._securitization_level = SecuritizationLevel.to_string(securitization_level)
        self._backwards = backwards
        self._stub_type_is_Long = stub_type_is_Long
        self._validate()
        self._last_fixing = last_fixing

    @property
    def coupon(self) -> float:
        """
        Getter for instrument's coupon.

        Returns:
            float: Instrument's coupon.
        """
        return self._coupon

    @coupon.setter
    def coupon(self, rate: float):
        """
        Setter for instrument's rate.

        Args:
            rate(float): interest rate of the instrument.
        """
        self._rate = rate

    @property
    def start_date(self) -> dt.date:
        """
        Getter for deposit's start date.

        Returns:
            date: deposit's start date.
        """
        return self._start_date

    @start_date.setter
    def start_date(self, start_date: _Union[dt.datetime, dt.date]):
        """
        Setter for deposit's start date.

        Args:
            start_date (Union[datetime, date]): deposit's start date.
        """
        self._start_date = _date_to_datetime(start_date)

    @property
    def maturity_date(self) -> dt.datetime:
        """
        Getter for deposit's maturity date.

        Returns:
            date: deposit's maturity date.
        """
        return self._maturity_date

    @maturity_date.setter
    def maturity_date(self, maturity_date: _Union[dt.datetime, dt.date]):
        """
        Setter for deposit's maturity date.

        Args:
            maturity_date (Union[datetime, date]): deposit's maturity date.
        """
        self._maturity_date = _date_to_datetime(maturity_date)

    @property
    def end_date(self) -> dt.datetime:
        """
        Getter for deposit's end date.

        Returns:
            date: deposit's end date.
        """
        return self._end_date

    @end_date.setter
    def end_date(self, end_date: _Union[dt.datetime, dt.date]):
        """
        Setter for deposit's end date.

        Args:
            end_date (Union[datetime, date]): deposit's end date.
        """
        if not isinstance(end_date, (dt.datetime, dt.date)):
            raise TypeError("end_date must be a datetime or date object.")
        self._end_date = _date_to_datetime(end_date)

    @property
    def frequency(self) -> Period:
        """
        Getter for instrument's fixing frequency.

        Returns:
            Period: instrument's fixing frequency.
        """
        return self._frequency

    @frequency.setter
    def frequency(self, frequency: _Union[Period, str]):
        """
        Setter for instrument's frequency.

        Args:
            frequency (Union[Period, str]): instrument's fixing frequency.
        """
        self._frequency = _term_to_period(frequency)

    @property
    def notional(self) -> float:
        """
        Getter for deposit's face value.

        Returns:
            float: deposit's face value.
        """
        return self._notional

    @notional.setter
    def notional(self, notional):
        self._notional = _check_positivity(notional)

    @property
    def day_count_convention(self) -> DayCounterType:
        """
        Getter for FRA's day count convention.

        Returns:
            str: FRA's day count convention.
        """
        return self._day_count_convention

    @day_count_convention.setter
    def day_count_convention(self, day_count_convention: _Union[DayCounterType, str]):
        self._day_count_convention = DayCounterType.to_string(day_count_convention)

    @property
    def business_day_convention(self) -> str:
        """
        Getter for FRA's day count convention.

        Returns:
            str: FRA's day count convention.
        """
        return self._business_day_convention

    @business_day_convention.setter
    def business_day_convention(self, business_day_convention: _Union[RollConvention, str]):
        self._business_day_convention = DayCounterType.to_string(business_day_convention)

    @property
    def roll_convention(self) -> str:
        """
        Getter for the roll convention used for business day adjustment.

        Returns:
            str: The roll convention used for business day adjustment.
        """
        return self._roll_convention

    @roll_convention.setter
    def roll_convention(self, roll_convention: _Union[RollRule, str]):
        """
        Setter for the roll convention used for business day adjustment.

        Args:
            roll_convention (_Union[RollRule, str]): The roll convention used for business day adjustment.
        """
        self._roll_convention = RollRule.to_string(roll_convention)

    @property
    def calendar(self):
        """
        Getter for the calendar used for business day adjustment.

        Returns:
            The calendar used for business day adjustment.
        """
        return self._calendar

    @calendar.setter
    def calendar(self, calendar: _Union[_HolidayBase, str]):
        """
        Setter for the calendar used for business day adjustment.

        Args:
            calendar (_Union[_HolidayBase, str]): The calendar used for business day adjustment.
        """
        if isinstance(calendar, str) and calendar.upper() == "TARGET":
            self._calendar = _ECB()
        else:
            self._calendar = _string_to_calendar(calendar)

    @property
    def notional_exchange(self):
        return self._notional_exchange

    @notional_exchange.setter
    def notional_exchange(self, notional_exchange: bool):
        self._notional_exchange = notional_exchange

    @property
    def settlement_days(self) -> int:
        """
        Getter for the number of settlement days.

        Returns:
            int: Number of settlement days.
        """
        return self._settlement_days

    @settlement_days.setter
    def settlement_days(self, settlement_days: int):
        """
        Setter for the number of settlement days.

        Args:
            settlement_days (int): Number of settlement days.
        """
        if not isinstance(settlement_days, int) or settlement_days < 0:
            raise ValueError("Settlement days must be a non-negative integer.")
        self._settlement_days = settlement_days

    @property
    def pays_in_arrears(self) -> bool:
        """
        Getter for the pays_in_arrears flag.

        Returns:
            bool: True if the instrument pays in arrears, False otherwise.
        """
        return self._pays_in_arrears

    @pays_in_arrears.setter
    def pays_in_arrears(self, pays_in_arrears: bool):
        """
        Setter for the pays_in_arrears flag.

        Args:
            pays_in_arrears (bool): True if the instrument pays in arrears, False otherwise.
        """
        if not isinstance(pays_in_arrears, bool):
            raise ValueError("pays_in_arrears must be a boolean value.")
        self._pays_in_arrears = pays_in_arrears

    @property
    def currency(self) -> str:
        """
        Getter for deposit's currency.

        Returns:
            str: deposit's  currency code
        """
        return self._currency

    @currency.setter
    def currency(self, currency: str):
        self._currency = Currency.to_string(currency)

    @property
    def securitization_level(self) -> str:
        """The bond's securitization level as a string."""
        return self._securitization_level

    @securitization_level.setter
    def securitization_level(self, value: _Union[SecuritizationLevel, str]):
        self._securitization_level = SecuritizationLevel.to_string(value)

    @property
    def issuer(self) -> str:
        """
        Getter for instrument's issuer.

        Returns:
            str: Instrument's issuer.
        """
        return self._issuer

    @issuer.setter
    def issuer(self, issuer: str):
        """
        Setter for instrument's issuer.

        Args:
            issuer (str): Instrument's issuer.
        """
        if not isinstance(issuer, str):
            raise ValueError("Issuer must be a string.")
        self._issuer = issuer

    @property
    def first_fixing_date(self) -> dt.datetime:
        """
        Getter for the first fixing date of the instrument.

        Returns:
            dt.datetime: The first fixing date.
        """
        return self._first_fixing_date

    @first_fixing_date.setter
    def first_fixing_date(self, first_fixing_date: _Union[dt.datetime, dt.date]):
        """
        Setter for the first fixing date of the instrument.

        Args:
            first_fixing_date (_Union[dt.datetime, dt.date]): The first fixing date.
        """
        self._first_fixing_date = _date_to_datetime(first_fixing_date)

    @property
    def obj_id(self) -> str:
        """
        Getter for the unique identifier of the object.

        Returns:
            str: Unique identifier of the object.
        """
        return self._obj_id

    @obj_id.setter
    def obj_id(self, obj_id: str):
        """
        Setter for the unique identifier of the object.

        Args:
            obj_id (str): Unique identifier of the object.
        """
        if not isinstance(obj_id, str):
            raise ValueError("Object ID must be a string.")
        self._obj_id = obj_id

    @property
    def backwards(self) -> bool:
        """
        Getter for the backwards flag.

        Returns:
            bool: True if the schedule is generated backwards, False otherwise.
        """
        return self._backwards

    @backwards.setter
    def backwards(self, backwards: bool):
        """
        Setter for the backwards flag.

        Args:
            backwards (bool): True if the schedule is generated backwards, False otherwise.
        """
        if not isinstance(backwards, bool):
            raise ValueError("Backwards must be a boolean value.")
        self._backwards = backwards

    @property
    def stub_type_is_Long(self) -> bool:
        """
        Getter for the stub type flag.

        Returns:
            bool: True if the stub type is long, False otherwise.
        """
        return self._stub_type_is_Long

    @stub_type_is_Long.setter
    def stub_type_is_Long(self, stub_type_is_Long: bool):
        """
        Setter for the stub type flag.

        Args:
            stub_type_is_Long (bool): True if the stub type is long, False otherwise.
        """
        if not isinstance(stub_type_is_Long, bool):
            raise ValueError("Stub type must be a boolean value.")
        self._stub_type_is_Long = stub_type_is_Long

    @property
    def spot_days(self) -> int:
        """
        Getter for the number of spot days.

        Returns:
            int: Number of spot days.
        """
        return self._spot_days

    @spot_days.setter
    def spot_days(self, spot_days: int):
        """
        Setter for the number of spot days.

        Args:
            spot_days (int): Number of spot days.
        """
        if not isinstance(spot_days, int) or spot_days < 0:
            raise ValueError("Spot days must be a non-negative integer.")
        self._spot_days = spot_days

    @property
    def last_fixing(self) -> _Optional[float]:
        """
        Getter for the last fixing value.

        Returns:
            _Optional[float]: The last fixing value, or None if not set.
        """
        return self._last_fixing

    @last_fixing.setter
    def last_fixing(self, last_fixing: _Optional[float]):
        """
        Setter for the last fixing value.

        Args:
            last_fixing (_Optional[float]): The last fixing value, or None if not set.
        """
        if last_fixing is not None and not isinstance(last_fixing, (float, int)):
            raise ValueError("Last fixing must be a float or None.")
        self._last_fixing = float(last_fixing) if last_fixing is not None else None

    def _validate(self):
        """Validates the parameters of the instrument."""
        _check_positivity(self._notional)
        _check_start_at_or_before_end(self._first_fixing_date, self._start_date)
        _check_start_before_end(self._start_date, self._end_date)
        _check_start_at_or_before_end(self._end_date, self._maturity_date)
        _check_positivity(self._settlement_days)
        if not isinstance(self._frequency, (Period, str)):
            raise ValueError("Frequency must be a Period object or string.")
        if not isinstance(self._calendar, (_HolidayBase, str)):
            raise ValueError("Calendar must be a HolidayBase or string.")

    # def _adjust_to_payment_date(self, accrual_end_date: dt.datetime) -> dt.datetime:
    #     """Adjusts the payment date by applying business day conventions and settlement days.

    #     Args:
    #         accrual_end_date: End date of the accrual period

    #     Returns:
    #         dt.datetime: Adjusted payment date that is guaranteed to be >= accrual_end_date
    #     """
    #     try:
    #         # First business day adjustment
    #         adjusted_date = roll_day(accrual_end_date, self._calendar, self._business_day_convention)

    #         # Add settlement days
    #         from dateutil.relativedelta import relativedelta

    #         with_settlement = adjusted_date + relativedelta(days=self._settlement_days)

    #         # Final business day adjustment
    #         final_date = roll_day(with_settlement, self._calendar, self._business_day_convention)

    #         # Ensure the payment date is not before the accrual end date
    #         if final_date < accrual_end_date:
    #             raise ValueError(f"Adjusted payment date {final_date} is before accrual end date {accrual_end_date}")

    #         return final_date
    #     except Exception as e:
    #         raise ValueError(f"Failed to adjust payment date: {e}")

    # def expected_cashflows(self) -> List[Tuple[dt.datetime, float]]:
    #     schedule = self.get_schedule()
    #     dates = schedule._roll_out(
    #         from_=self._start_date,
    #         to_=self._end_date,
    #         term=_term_to_period(self._frequency),
    #     )
    #     dcc = DayCounter(self.day_count_convention)
    #     if self._coupon_type == "float":
    #         cashflows = [(self._adjust_to_payment_date(d1), self._notional * self._coupon * dcc.yf(d1, d2)) for d1, d2 in zip(dates[:-1], dates[1:])]
    #     else:
    #         cashflows = [(self._adjust_to_payment_date(d1), self._notional * self._coupon * dcc.yf(d1, d2)) for d1, d2 in zip(dates[:-1], dates[1:])]
    #     if self._notional_exchange:
    #         cashflows.append((self._maturity_date, self._notional))
    #     return cashflows

    def get_schedule(self) -> Schedule:
        """Returns the schedule of the accrual periods of the instrument."""
        return Schedule(
            start_day=self._start_date,
            end_day=self._end_date,
            time_period=self._frequency,
            backwards=self._backwards,
            stub_type_is_Long=self._stub_type_is_Long,
            business_day_convention=self._business_day_convention,
            roll_convention=self._roll_convention,
            calendar=self._calendar,
        )

    @abc.abstractmethod
    def _to_dict(self) -> dict:
        pass
