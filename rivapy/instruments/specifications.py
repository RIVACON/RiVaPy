import abc
from typing import List, Tuple
from rivapy.tools.interfaces import FactoryObject
import datetime as dt

# import rivapy.tools.interfaces as interfaces
from rivapy.tools.enums import SecuritizationLevel, Currency, DayCounterType, RollConvention, RollRule
from typing import List, Tuple, Optional as _Optional, Union as _Union
from rivapy.tools.datetools import Period, _date_to_datetime, _term_to_period, _string_to_calendar, DayCounter, Schedule
from holidays import HolidayBase as _HolidayBase
from holidays import EuropeanCentralBank as _ECB

# from rivapy.marketdata.curves import DiscountCurve

# from rivapy.enums import Currency
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
        start_date: _Union[dt.date, dt.datetime],
        maturity_date: _Union[dt.date, dt.datetime],
        notional: float,
        notional_exchange: bool = True,
        coupon: float = 0.0,
        frequency: _Optional[_Union[Period, str]] = None,
        day_count_convention: _Union[DayCounterType, str] = DayCounterType.ACT360,
        business_day_convention: _Union[RollConvention, str] = RollConvention.MODIFIED_FOLLOWING,
        roll_convention: _Union[RollRule, str] = RollRule.EOM,
        calendar: _Union[_HolidayBase, str] = _ECB(),
        coupon_type: str = "fix",
        # fwd_curve: _Optional[DiscountCurve] = None,
    ):
        """Initializes the HasExpectedCashflows object.

        Args:
            obj_id (str): Unique identifier for the object.
            start_date (_Union[date, datetime]): Start date of the first accrual period.
            maturity_date (_Union[date, datetime]): End of the last accrual period.
            notional (float): Notional amount of the instrument.
            coupon (float): Fixed coupon rate .
            frequency (_Union[Period, str]): frequency of fixings.
            day_count_convention (_Union[DayCounterType, str], optional): Day count convention. Defaults to DayCounterType.ACT360.
            business_day_convention (_Union[RollConvention, str], optional): Business day convention. Defaults to RollConvention.MODIFIED_FOLLOWING.
            roll_convention (_Union[RollRule, str], optional): Roll convention. Defaults to RollRule.EOM.
            calendar (_Union[_HolidayBase, str], optional): Holiday calendar. Defaults to _ECB().
        """
        self._obj_id = obj_id
        self._start_date = _date_to_datetime(start_date)
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
        # self._fwd_curve = fwd_curve

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
    def day_count_convention(self) -> RollConvention:
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

    def expected_cashflows(self) -> List[Tuple[dt.datetime, float]]:
        schedule = self.get_schedule()
        dates = schedule._roll_out(
            from_=self._start_date,
            to_=self._maturity_date,
            term=_term_to_period(self._frequency),
        )
        dcc = DayCounter(self.day_count_convention)
        if self._coupon_type == "float":
            cashflows = [(d1, self._notional * self._coupon * dcc.yf(d1, d2)) for d1, d2 in zip(dates[:-1], dates[1:])]
        else:
            cashflows = [(d1, self._notional * self._coupon * dcc.yf(d1, d2)) for d1, d2 in zip(dates[:-1], dates[1:])]
        if self._notional_exchange:
            cashflows.append((self._maturity_date, self._notional))
        return cashflows

    def get_schedule(self) -> Schedule:
        """Returns the schedule of the cashflows."""
        return Schedule(
            start_day=self._start_date,
            end_day=self._maturity_date,
            time_period=self._frequency,
            backwards=True,
            stub_type_is_Long=True,
            business_day_convention=self._business_day_convention,
            roll_convention=self._roll_convention,
            calendar=self._calendar,
        )

    @abc.abstractmethod
    def _to_dict(self) -> dict:
        pass
