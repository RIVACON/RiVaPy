import abc
from datetime import datetime, date
from typing import List, Any, Optional, Union, Tuple
from holidays import HolidayBase
from holidays.financial import ECB
import QuantLib as ql
from dateutil.relativedelta import relativedelta 
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from rivapy.tools.enums import Currency, Rating, SecuritizationLevel, RollConvention
from rivapy.tools.datetools import _date_to_datetime, Schedule, Period, DayCounterType, DayCounter
from rivapy.tools._validators import _check_positivity, _check_start_before_end, _string_to_calendar
from rivapy.tools.interfaces import BaseDatedCurve 
from scipy.optimize import brentq


class BondSpec(abc.ABC):
    """Abstract base class for bond specifications."""
    def __init__(self,
                 obj_id: str,
                 schedule: 'Schedule', 
                 notional: float,
                 currency: Union[Currency, str],
                 issue_date: Union[date, datetime],
                 issuer: Optional[str] = None,
                 securitization_level: Optional[Union[SecuritizationLevel, str]] = SecuritizationLevel.NONE,
                 rating: Optional[Union[Rating, str]] = Rating.NONE):
        """
        Initializes the base bond specification.

        Args:
            obj_id (str): A unique identifier for the bond, e.g., ISIN.
            schedule (Schedule): The payment schedule of the bond.
            notional (float): The face value of the bond.
            currency (Union[Currency, str]): The currency of the bond.
            issue_date (Union[date, datetime]): The date the bond was issued.
            issuer (Optional[str], optional): The issuer of the bond. Defaults to None.
            securitization_level (Optional[Union[SecuritizationLevel, str]], optional): The securitization level. Defaults to SecuritizationLevel.NONE.
            rating (Optional[Union[Rating, str]], optional): The credit rating of the bond. Defaults to Rating.NONE.
        """

        self.obj_id = obj_id 

        if not isinstance(schedule, Schedule):
            raise TypeError("schedule must be an instance of rivapy.tools.datetools.Schedule.") 
        
        self._schedule = schedule 
        self.notional = notional 
        self.currency = currency 
        self.issue_date = issue_date 
        self.issuer = issuer
        self.securitization_level = securitization_level
        self.rating = rating

        _check_start_before_end(self.issue_date, self.maturity_date)

    @property
    def schedule(self) -> 'Schedule':
        """The bond's schedule of payments."""
        return self._schedule

    @property
    def issue_date(self) -> datetime: 
        """The bond's issue date as a datetime object."""
        return self._issue_date

    @issue_date.setter
    def issue_date(self, value: Union[date, datetime]):
        self._issue_date = _date_to_datetime(value)

    @property
    def maturity_date(self) -> datetime: 
        """The bond's maturity date, derived from the schedule's end date."""
        return _date_to_datetime(self._schedule.end_day) 

    @property
    def notional(self) -> float:
        """The bond's notional amount (face value)."""
        return self._notional

    @notional.setter
    def notional(self, value: float):
        self._notional = _check_positivity(value)

    @property
    def currency(self) -> str: 
        """The bond's currency as a string."""
        return self._currency

    @currency.setter
    def currency(self, value: Union[Currency, str]):
        self._currency = Currency.to_string(value)

    @property
    def issuer(self) -> Optional[str]:
        """The bond's issuer."""
        return self._issuer

    @issuer.setter
    def issuer(self, value: Optional[str]):
        self._issuer = value

    @property
    def securitization_level(self) -> str: 
        """The bond's securitization level as a string."""
        return self._securitization_level

    @securitization_level.setter
    def securitization_level(self, value: Union[SecuritizationLevel, str]):
        self._securitization_level = SecuritizationLevel.to_string(value)

    @property
    def rating(self) -> str: 
        """The bond's credit rating as a string."""
        return self._rating

    @rating.setter
    def rating(self, value: Union[Rating, str]):
        self._rating = Rating.to_string(value)

    @abc.abstractmethod
    def expected_cashflows(self) -> List[Tuple[datetime, float]]:
        """
        Computes all expected cashflows of the bond.
        
        Returns:
            List[Tuple[datetime, float]]: A list of tuples, where each tuple contains
                                          the payment date and the cashflow amount.
        """
        pass

    @abc.abstractmethod
    def compute_price(self, discount_curve: 'DiscountCurve') -> float:
        """
        Computes the dirty price of the bond.
        The dirty price is the price of a bond including any accrued interest.

        Args:
            discount_curve (DiscountCurve): The curve used to discount future cashflows.

        Returns:
            float: The dirty price of the bond.
        """
        pass
    @abc.abstractmethod
    def compute_yield(self, price: float, val_date: datetime) -> float: 
        """
        Computes the yield-to-maturity (YTM) of the bond.

        Args:
            price (float): The dirty price of the bond.
            val_date (datetime): The valuation date.

        Returns:
            float: The computed yield-to-maturity.
        """
        pass

class FixedRateBond(BondSpec):
    """
    Represents a fixed-rate bond with regular coupon payments.
    """
    def __init__(self,
                 obj_id: str,
                 schedule: 'Schedule',
                 notional: float,
                 currency: Union[Currency, str],
                 issue_date: Union[date, datetime],
                 coupon_rate: float, 
                 issuer: Optional[str] = None,
                 securitization_level: Optional[Union[SecuritizationLevel, str]] = SecuritizationLevel.NONE, 
                 rating: Optional[Union[Rating, str]] = Rating.NONE,
                 accrual_day_counter_type: DayCounterType = DayCounterType.ActActICMA): 
        """
        Initializes a fixed-rate bond.

        Args:
            obj_id (str): A unique identifier for the bond.
            schedule (Schedule): The payment schedule of the bond.
            notional (float): The face value of the bond.
            currency (Union[Currency, str]): The currency of the bond.
            issue_date (Union[date, datetime]): The date the bond was issued.
            coupon_rate (float): The annual coupon rate (e.g., 0.05 for 5%).
            issuer (Optional[str], optional): The issuer of the bond. Defaults to None.
            securitization_level (Optional[Union[SecuritizationLevel, str]], optional): The securitization level. Defaults to SecuritizationLevel.NONE.
            rating (Optional[Union[Rating, str]], optional): The credit rating of the bond. Defaults to Rating.NONE.
            accrual_day_counter_type (DayCounterType, optional): The day count convention for accrual calculations. Defaults to DayCounterType.ActActICMA.
        """
        super().__init__(obj_id, schedule, notional, currency, issue_date, issuer, securitization_level, rating)
        if coupon_rate < 0: 
            raise ValueError("Coupon rate must be non-negative.")
        self.coupon_rate = coupon_rate 
        self._accrual_day_counter = DayCounter(accrual_day_counter_type)
        self.accrual_day_counter_type = accrual_day_counter_type
    
    def _get_coupon_frequency_and_reference_period_days(self, payment_date: datetime) -> Tuple[float, int]:
        """
        Helper to get coupon frequency and days in reference period for a payment date.
        This is primarily used for the Act/Act ICMA day count convention.

        Args:
            payment_date (datetime): The payment date of a coupon period.

        Returns:
            Tuple[float, int]: A tuple containing the numeric coupon frequency (e.g., 1.0 for annual, 2.0 for semi-annual)
                               and the number of days in a regular coupon period."""
                
        coupon_frequency = 0.0
        if self._schedule.time_period.years > 0:
            coupon_frequency = 1.0 / self._schedule.time_period.years
        elif self._schedule.time_period.months > 0:
            coupon_frequency = 12.0 / self._schedule.time_period.months
        elif self._schedule.time_period.days > 0: 
            coupon_frequency = 365.25 / self._schedule.time_period.days 
        else:
            coupon_frequency = 1.0 

        
        ref_period_start_dt = payment_date - relativedelta(
            years=self._schedule.time_period.years,
            months=self._schedule.time_period.months,
            days=self._schedule.time_period.days
        )
        regular_coupon_period_in_days = (payment_date - ref_period_start_dt).days
        return coupon_frequency, regular_coupon_period_in_days

    def expected_cashflows(self) -> List[Tuple[datetime, float]]:
        """
        Computes all expected cashflows (coupons and notional) of the bond.

        Returns:
            List[Tuple[datetime, float]]: A sorted list of (date, amount) tuples for each cashflow.
        """
        # Generate all schedule dates, which are already business-day adjusted
        all_schedule_dates = self._schedule.generate_dates(ends_only=False)  

        if len(all_schedule_dates) < 2:
            # Only notional at maturity if it's after issue date
            if self.maturity_date >= self.issue_date:
                return [(self.maturity_date, self.notional)]
            return []

        cashflows = []
        # Get frequency information needed for Act/Act ICMA calculation
        coupon_freq_numeric, _ = self._get_coupon_frequency_and_reference_period_days(all_schedule_dates[1])
        coupon_frequency_int = int(coupon_freq_numeric)

        # --- Iterate over all coupon periods ---
        for i in range(len(all_schedule_dates) - 1):
            period_start_dt = _date_to_datetime(all_schedule_dates[i])
            payment_date_dt = _date_to_datetime(all_schedule_dates[i+1])

            # Skip cashflows that are paid out before or on the issue date
            if payment_date_dt <= self.issue_date:
                continue 

            # The accrual for the coupon calculation always uses the full period from the schedule
            # to correctly handle stub periods (short/long first or last coupons).
            year_fraction_for_coupon = self._accrual_day_counter.yf(
                period_start_dt, payment_date_dt,
                coupon_schedule=all_schedule_dates,
                coupon_frequency=coupon_frequency_int
            )

            coupon_amount = self.notional * self.coupon_rate * year_fraction_for_coupon
            if coupon_amount > 0.0: 
                cashflows.append((payment_date_dt, coupon_amount))
            logger.debug(f"  RiVaPy YF Detail: Accrual {period_start_dt.strftime('%Y-%m-%d')} to {payment_date_dt.strftime('%Y-%m-%d')}, DayCount: {self.accrual_day_counter_type.name}, YF: {year_fraction_for_coupon:.8f}")

        # Add notional at maturity date (which is the last date in the schedule)
        maturity_payment_date = _date_to_datetime(all_schedule_dates[-1])
        if maturity_payment_date >= self.issue_date:
            cashflows.append((maturity_payment_date, self.notional))

        # Use a dictionary to sum amounts for cashflows on the same date (e.g., last coupon + notional)
        combined_cashflows = {}
        for cf_date, amount in cashflows:
            # Normalize datetime to date to ensure correct grouping if time components differ
            normalized_date = cf_date.replace(hour=0, minute=0, second=0, microsecond=0)
            combined_cashflows[normalized_date] = combined_cashflows.get(normalized_date, 0.0) + amount
        
        # Convert back to list of tuples and sort by date
        result_cashflows = sorted([(dt, amt) for dt, amt in combined_cashflows.items()], key=lambda x: x[0])
    
        return result_cashflows

    def _compute_dirty_price(self, discount_curve: 'DiscountCurve') -> float:
        """
        Internal method to compute the dirty price by discounting all future cashflows.
        
        Args:
            discount_curve (DiscountCurve): The curve used for discounting.

        Returns:
            float: The calculated dirty price.
        """
        val_date_dt = _date_to_datetime(discount_curve.valuation_date)
        cashflows = self.expected_cashflows()
        pv_cashflows = 0.0
        for c in cashflows:
            if c[0] > val_date_dt:
                df = discount_curve.value(val_date_dt, c[0])
                pv_cashflows += df * c[1]
        return pv_cashflows

    def compute_accrued_interest(self, valuation_date: Union[date, datetime]) -> float:
        """
        Computes the accrued interest of the bond on a given valuation date.

        Args:
            valuation_date (Union[date, datetime]): The date for which to calculate the accrued interest.

        Returns:
            float: The amount of accrued interest. Returns 0.0 if the valuation date is
                   outside the bond's life (before issue or on/after maturity).
        """
        val_date_dt = _date_to_datetime(valuation_date)
        # No accrued interest if valuation is outside the bond's life
        if val_date_dt >= self.maturity_date or val_date_dt < self.issue_date:
            return 0.0

        all_coupon_schedule_dates = self._schedule.generate_dates(ends_only=False) 
        
        current_accrual_start = None
        current_accrual_end = None

        if len(all_coupon_schedule_dates) < 2: 
            return 0.0

        # Find the coupon period that contains the valuation date
        for i in range(len(all_coupon_schedule_dates) - 1):
            p_start_dt = _date_to_datetime(all_coupon_schedule_dates[i])
            p_end_dt = _date_to_datetime(all_coupon_schedule_dates[i+1])
            
            
            if p_start_dt <= val_date_dt < p_end_dt:
                current_accrual_start = p_start_dt
                current_accrual_end = p_end_dt
                break
        
        if current_accrual_start is None or current_accrual_end is None:
            return 0.0

        # Get frequency information needed for Act/Act ICMA
        coupon_frequency, regular_coupon_period_in_days = \
            self._get_coupon_frequency_and_reference_period_days(current_accrual_end)

        # Calculate accrued interest year fraction for the period [current_accrual_start, val_date_dt]
        if self.accrual_day_counter_type == DayCounterType.ActActICMA:
            accrued_year_fraction = self._accrual_day_counter.yf(
                current_accrual_start, val_date_dt, 
                all_coupon_schedule_dates, int(coupon_frequency)
            )
        else: 
            accrued_year_fraction = self._accrual_day_counter.yf(
                current_accrual_start, val_date_dt
            )

        
        return self.notional * self.coupon_rate * accrued_year_fraction

    def compute_price(self, discount_curve: 'DiscountCurve') -> float:
        """
        Computes the dirty price of the bond.
        The dirty price is the price of a bond including any accrued interest.

        Args:
            discount_curve (DiscountCurve): The curve used to discount future cashflows.

        Returns:
            float: The dirty price of the bond.
        """
        return self._compute_dirty_price(discount_curve)

    def compute_clean_price(self, discount_curve: 'DiscountCurve') -> float:
        """
        Computes the clean price of the bond.
        Clean Price = Dirty Price - Accrued Interest.

        Args:
            discount_curve (DiscountCurve): The curve used to discount future cashflows.

        Returns:
            float: The clean price of the bond.
        """
        dirty_price = self._compute_dirty_price(discount_curve)
        accrued = self.compute_accrued_interest(discount_curve.valuation_date)
        return dirty_price - accrued

    def compute_yield(self, dirty_price: float, val_date: datetime,
                        yield_search_lower_bound: float = -0.2, 
                        yield_search_upper_bound: float = 1.5) -> float: 
        """
        Computes the yield-to-maturity (YTM) for a given dirty price.
        This method uses the brentq root-finding algorithm to find the yield that
        equates the present value of future cashflows to the given dirty price.

        Args:
            dirty_price (float): The dirty price of the bond.
            val_date (datetime): The valuation date.
            yield_search_lower_bound (float, optional): The lower bound for the yield search. Defaults to -0.2.
            yield_search_upper_bound (float, optional): The upper bound for the yield search. Defaults to 1.5.

        Returns:
            float: The calculated yield-to-maturity (annually compounded).
        """
        valuation_datetime = _date_to_datetime(val_date)
        cashflows = self.expected_cashflows()
        
        # For ActActICMA, we need the schedule and frequency for the day counter
        all_schedule_dates = None
        coupon_frequency_int = None
        if self.accrual_day_counter_type == DayCounterType.ActActICMA:
            all_schedule_dates = self._schedule.generate_dates(ends_only=False)
            if len(all_schedule_dates) > 1:
                coupon_freq_numeric, _ = self._get_coupon_frequency_and_reference_period_days(all_schedule_dates[1])
                coupon_frequency_int = int(coupon_freq_numeric)

        

        def target_function(r: float)->float:
            """
            Calculates the difference between the PV of cashflows (for a given yield r) and the dirty price.
            The root of this function is the desired YTM.
            """
            # Calculate the dirty price for a given yield 'r' without creating a full DiscountCurve object. This ensures we use the bond's specific day counter for the yield calculation, matching QuantLib.
            pv_cashflows = 0.0
            for cf_date, amount in cashflows:
                if cf_date > valuation_datetime:
                    # Calculate year fraction for discounting using the bond's accrual day counter
                    yf = self._accrual_day_counter.yf(valuation_datetime, cf_date, 
                                                      coupon_schedule=all_schedule_dates, 
                                                      coupon_frequency=coupon_frequency_int)
                    
                    # Discount the cashflow (annually compounded, matching QL's default)
                    df = 1.0 / ((1.0 + r) ** yf)
                    pv_cashflows += df * amount
            
            return pv_cashflows - dirty_price
        
        # Use brentq to find the root of the target function (i.e., the yield)
        result =  brentq(target_function, yield_search_lower_bound, yield_search_upper_bound, 
                         full_output = False)
        return result

class DiscountCurve(BaseDatedCurve):
    """
    A simple discount curve implementation based on a single flat interest rate.
    """
    def __init__(self, 
                 valuation_date: Union[date, datetime], 
                 flat_rate: Optional[float] = 0.05,
                 curve_data: Any = None,
                 day_counter_type: DayCounterType = DayCounterType.Act365Fixed): 
        """
        Initializes the flat discount curve.

        Args:
            valuation_date (Union[date, datetime]): The valuation date of the curve.
            flat_rate (Optional[float], optional): The flat interest rate used for discounting. Defaults to 0.05.
            curve_data (Any, optional): Placeholder for more complex curve data (not used in this implementation). Defaults to None.
            day_counter_type (DayCounterType, optional): The day count convention for calculating year fractions. Defaults to DayCounterType.Act365Fixed.
        """
        self.valuation_date = valuation_date
        self._flat_rate = flat_rate
        self._curve_data = curve_data # Placeholder for more complex curve data
        self._day_counter = DayCounter(day_counter_type)

    @property
    def valuation_date(self) -> datetime:
        """The valuation date of the curve as a datetime object."""
        return self._valuation_date

    @valuation_date.setter
    def valuation_date(self, value: Union[date, datetime]):
        self._valuation_date = _date_to_datetime(value)

    def get_discount_factor(self, target_date: Union[date, datetime]) -> float:
        """
        Calculates the discount factor from the valuation date to a target date.

        Args:
            target_date (Union[date, datetime]): The date to which to discount.

        Returns:
            float: The discount factor. Returns 0.0 if the target date is before the valuation date.
        """
        val_date_dt = _date_to_datetime(self.valuation_date)
        target_date_dt = _date_to_datetime(target_date)
        
        if target_date_dt < val_date_dt:
            return 0.0 
        time_to_maturity_years = self._day_counter.yf(val_date_dt, target_date_dt)
        rate_to_use = self._flat_rate if self._flat_rate is not None else 0.02 # Fallback if flat_rate is None
        return 1 / ((1 + rate_to_use) ** time_to_maturity_years)
    
    def value(self, ref_date: datetime, target_date: datetime) -> float:
        """
        Returns the discount factor from a reference date to a target date.
        For this simple implementation, the reference date must be the curve's valuation date.

        Args:
            ref_date (datetime): The reference date (must match the curve's valuation date).
            target_date (datetime): The date to which to discount.

        Raises:
            ValueError: If the reference date does not match the curve's valuation date.

        Returns:
            float: The discount factor.
        """
        # Ensure ref_date matches the curve's valuation_date for this simple implementation
        if _date_to_datetime(ref_date).date() != self.valuation_date.date():
            raise ValueError(f"Reference date {ref_date} does not match DiscountCurve valuation date {self.valuation_date}")
        return self.get_discount_factor(target_date)

def run_comparison_example():
    """
    Runs a comparison example between the RiVaPy bond implementation and QuantLib.
    This function sets up a specific bond and market scenario, calculates various
    metrics (prices, accrued interest, yield) using both libraries, and prints
    a summary of the differences.
    """
    # --- Bond and Market Parameters ---
    obj_id = 'DE000CZ40NT7'
    issue_date_dt = date(2019, 2, 1)
    maturity_date_dt = date(2021, 2, 1)
    coupon_rate_val = 0.0  
    tenor_period = Period(years=1)  
    notional_val = 100000.0
    currency_val = 'EUR'
    issuer_val = 'Commerzbank'
    securitisation_level_val = SecuritizationLevel.NON_PREFERRED_SENIOR
    valuation_date_dt = date(2020, 2, 1)
    flat_rate_val = 0.02 
    
    # --- RiVaPy Setup ---
    # Create a payment schedule for the bond
    rivapy_schedule = Schedule(start_day=issue_date_dt,           
                               end_day=maturity_date_dt,           
                               time_period=tenor_period,       
                               backwards=True,
                               stub=True, 
                               business_day_convention=RollConvention.FOLLOWING,
                               calendar=ECB(years=range(issue_date_dt.year, maturity_date_dt.year + 2)))

    # Create the RiVaPy FixedRateBond instance
    rivapy_bond = FixedRateBond(obj_id=obj_id,
                                schedule=rivapy_schedule, 
                                notional=notional_val,
                                currency=currency_val,
                                issue_date=issue_date_dt,
                                coupon_rate=coupon_rate_val,
                                issuer=issuer_val, 
                                securitization_level=securitisation_level_val, 
                                accrual_day_counter_type=DayCounterType.ACT_ACT) 
    
    
    # Create a simple flat discount curve
    rivapy_discount_curve = DiscountCurve(valuation_date=valuation_date_dt, 
                                          flat_rate=flat_rate_val,
                                          day_counter_type=DayCounterType.ACT_ACT)  

    # --- RiVaPy Calculations ---
    rivapy_dirty_price = rivapy_bond.compute_price(rivapy_discount_curve)
    rivapy_accrued_interest = rivapy_bond.compute_accrued_interest(valuation_date_dt)
    rivapy_clean_price = rivapy_dirty_price - rivapy_accrued_interest 
    rivapy_ytm = rivapy_bond.compute_yield(dirty_price=rivapy_dirty_price, val_date=valuation_date_dt)

    # --- Debug Logging for RiVaPy ---
    logger.debug("RiVaPy Cashflows:")
    for cf_date, amount in rivapy_bond.expected_cashflows():
        logger.debug(f"Date: {cf_date.strftime('%Y-%m-%d')}, Amount: {amount:.4f}")
    logger.debug(f"RiVaPy Accrued Interest: {rivapy_accrued_interest:.4f}")
    logger.debug(f"RiVaPy Dirty Price: {rivapy_dirty_price}")
    logger.debug(f"RiVaPy Discount Factors:")
    val_date_dt = _date_to_datetime(valuation_date_dt)
    for cf_date, amount in rivapy_bond.expected_cashflows():
        if cf_date >= val_date_dt:
            df = rivapy_discount_curve.value(val_date_dt, cf_date)
            logger.debug(f"Date: {cf_date.strftime('%Y-%m-%d')}, DF: {df:.8f}")

    # --- QuantLib Setup ---
    ql_valuation_date = ql.Date(valuation_date_dt.day, valuation_date_dt.month, valuation_date_dt.year)
    ql.Settings.instance().evaluationDate = ql_valuation_date

    # Define QuantLib calendar and conventions
    ql_calendar = ql.TARGET() 
    ql_convention = ql.Unadjusted 
    
    
    # Create QuantLib schedule
    ql_issue_date = ql.Date(issue_date_dt.day, issue_date_dt.month, issue_date_dt.year)
    ql_maturity_date = ql.Date(maturity_date_dt.day, maturity_date_dt.month, maturity_date_dt.year)
    ql_schedule = ql.Schedule(ql_issue_date, 
                              ql_maturity_date, 
                              ql.Period(ql.Annual), 
                              ql_calendar,
                              ql_convention,
                              ql_convention, 
                              ql.DateGeneration.Backward, 
                              False) 

    # Create QuantLib FixedRateBond instance
    ql_accrual_day_counter = ql.ActualActual(ql.ActualActual.ISDA) 
    ql_bond = ql.FixedRateBond(0, # SettlementTage
                               notional_val,
                               ql_schedule,
                               [coupon_rate_val], 
                               ql_accrual_day_counter,
                               ql_convention, 
                               100.0,
                               ql_issue_date)
                                

    # Create QuantLib discount curve (YieldTermStructure)
    ql_discount_day_counter = ql.ActualActual(ql.ActualActual.ISDA)
    ql_discount_curve = ql.FlatForward(ql_valuation_date,
                                       ql.QuoteHandle(ql.SimpleQuote(flat_rate_val)),
                                       ql_discount_day_counter,
                                       ql.Compounded, 
                                       ql.Annual) 

    # Set the pricing engine for the QuantLib bond
    ql_bond_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(ql_discount_curve))
    ql_bond.setPricingEngine(ql_bond_engine)

    # --- QuantLib Calculations ---
    # Prices are returned per 100 notional by default
    ql_clean_price_per_100 = ql_bond.cleanPrice()
    ql_dirty_price_per_100 = ql_bond.dirtyPrice()
    ql_accrued_amount_per_100 = ql_bond.accruedAmount()

    # Scale results to the bond's actual notional
    scale_factor = notional_val / 100.0
    ql_absolute_clean_price = ql_clean_price_per_100 * scale_factor
    ql_absolute_dirty_price = ql_dirty_price_per_100 * scale_factor
    ql_absolute_accrued_amount = ql_accrued_amount_per_100 * scale_factor

    # Calculate Yield-to-Maturity in QuantLib
    ql_dirty_price_object = ql.BondPrice(ql_dirty_price_per_100, ql.BondPrice.Dirty)
    ql_ytm = ql_bond.bondYield(ql_dirty_price_object, 
                               ql_accrual_day_counter,    
                               ql.Compounded,             
                               ql.Annual,                 
                               ql_valuation_date,         
                               1.0e-10, # Accuracy
                               200) # Max iterations

    # --- Debug Logging for QuantLib ---
    logger.debug("\nQuantLib Cashflows and Year Fractions:")
    for cf_idx, cf in enumerate(ql_bond.cashflows()):
        coupon = ql.as_coupon(cf)
        if coupon:
            # For QuantLib coupons, the accrualPeriod() method directly gives the year fraction
            # as calculated by its day counter for its specific accrual period.
            yf_ql_coupon = coupon.accrualPeriod()
            accrual_start_dt_ql = coupon.accrualStartDate()
            accrual_end_dt_ql = coupon.accrualEndDate()
            day_counter_name_ql = coupon.dayCounter().name()
            
            logger.debug(f"  QL CF {cf_idx+1}: Date: {cf.date().ISO()}, Amount: {cf.amount():.4f}, "
                         f"Accrual: {accrual_start_dt_ql.ISO()} to {accrual_end_dt_ql.ISO()}, "
                         f"DayCount: {day_counter_name_ql}, YF: {yf_ql_coupon:.8f}")
        elif cf.amount() != 0: # e.g. Notional payment
            logger.debug(f"  QL CF {cf_idx+1}: Date: {cf.date().ISO()}, Amount: {cf.amount():.4f} (Non-coupon, e.g., Notional)")
            
    logger.debug(f"QuantLib Accrued Interest: {ql_absolute_accrued_amount:.4f}")
    logger.debug(f"QuantLib Clean Price: {ql_absolute_clean_price:.4f}")
    logger.debug(f"QuantLib Discount Factors:")
    for cf in ql_bond.cashflows():
        if cf.date() >= ql_valuation_date:
            t = ql_discount_day_counter.yearFraction(ql_valuation_date, cf.date())
            df = ql_discount_curve.discount(cf.date())
            logger.debug(f"Date: {cf.date().ISO()}, DF: {df:.8f}")

    # --- Compare Differences ---
    logger.debug("\nDetailed Differences:")
    logger.debug(f"Accrued Interest Diff: {rivapy_accrued_interest - ql_absolute_accrued_amount:.4f}")
    logger.debug(f"Clean Price Diff: {rivapy_clean_price - ql_absolute_clean_price:.4f}")
    logger.debug(f"Dirty Price Diff: {rivapy_dirty_price - ql_absolute_dirty_price:.4f}")
    logger.debug(f"YTM Diff: {(rivapy_ytm - ql_ytm)*100:.4f}%")

    # --- Comparison Summary ---
    print("\n--- Comparison Summary ---")
    print(f"Clean Price Difference (RiVaPy - QuantLib): {currency_val} {rivapy_clean_price - ql_absolute_clean_price:,.4f}")
    print(f"Dirty Price Difference (RiVaPy - QuantLib): {currency_val} {rivapy_dirty_price - ql_absolute_dirty_price:,.4f}")
    print(f"Yield Difference (RiVaPy - QuantLib): {(rivapy_ytm - ql_ytm)*100:.4f}%")
    print(f"RiVaPy Clean Price: {rivapy_clean_price:,.4f}, QuantLib Absolute Clean Price: {ql_absolute_clean_price:,.4f}")


if __name__ == '__main__':
    run_comparison_example()