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
from rivapy.tools.interfaces import BaseDatedCurve #, HasExpectedCashflows # HasExpectedCashflows not directly used by BondSpec
from scipy.optimize import brentq


class BondSpec(abc.ABC):
    def __init__(self,
                 obj_id: str,
                 schedule: 'Schedule', 
                 notional: float,
                 currency: Union[Currency, str],
                 issue_date: Union[date, datetime],
                 issuer: Optional[str] = None,
                 securitization_level: Optional[Union[SecuritizationLevel, str]] = SecuritizationLevel.NONE,
                 rating: Optional[Union[Rating, str]] = Rating.NONE):

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
        return self._notional

    @notional.setter
    def notional(self, value: float):
        self._notional = _check_positivity(value)

    @property
    def currency(self) -> str: 
        return self._currency

    @currency.setter
    def currency(self, value: Union[Currency, str]):
        self._currency = Currency.to_string(value)

    @property
    def issuer(self) -> Optional[str]:
        return self._issuer

    @issuer.setter
    def issuer(self, value: Optional[str]):
        self._issuer = value

    @property
    def securitization_level(self) -> str: 
        return self._securitization_level

    @securitization_level.setter
    def securitization_level(self, value: Union[SecuritizationLevel, str]):
        self._securitization_level = SecuritizationLevel.to_string(value)

    @property
    def rating(self) -> str: 
        return self._rating

    @rating.setter
    def rating(self, value: Union[Rating, str]):
        self._rating = Rating.to_string(value)

    @abc.abstractmethod
    def expected_cashflows(self) -> List[Tuple[datetime, float]]:
        """Computes all expected cashflows of the bond."""
        pass

    @abc.abstractmethod
    def compute_price(self, discount_curve: 'DiscountCurve') -> float:
        """
        Computes the dirty price of the bond.
        The dirty price is the price of a bond including any accrued interest.
        """
        pass
    @abc.abstractmethod
    def compute_yield(self, price: float, val_date: datetime) -> float: 
        pass

class FixedRateBond(BondSpec):
    
    def __init__(self,
                 obj_id: str,
                 schedule: 'Schedule',
                 notional: float,
                 currency: Union[Currency, str],
                 issue_date: Union[date, datetime],
                 coupon_rate: float, 
                 issuer: Optional[str] = None,
                 securitization_level: Optional[Union[SecuritizationLevel, str]] = SecuritizationLevel.NONE, # Corrected typo
                 rating: Optional[Union[Rating, str]] = Rating.NONE,
                 accrual_day_counter_type: DayCounterType = DayCounterType.ActActICMA): # Added for accrual
        super().__init__(obj_id, schedule, notional, currency, issue_date, issuer, securitization_level, rating)
        if coupon_rate < 0: 
            raise ValueError("Coupon rate must be non-negative.")
        self.coupon_rate = coupon_rate 
        self._accrual_day_counter = DayCounter(accrual_day_counter_type)
        self.accrual_day_counter_type = accrual_day_counter_type
    
    def _get_coupon_frequency_and_reference_period_days(self, payment_date: datetime) -> Tuple[float, int]:
        """Helper to get coupon frequency and days in reference period for a payment date."""
                
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
        all_schedule_dates = self._schedule.generate_dates(ends_only=False)  

        if len(all_schedule_dates) < 2:
            return []

        cashflows = []

        if not all_schedule_dates or len(all_schedule_dates) < 2:
            
            if all_schedule_dates and len(all_schedule_dates) > 0:
                
                final_payment_date_dt = _date_to_datetime(all_schedule_dates[-1])
                if final_payment_date_dt >= self.issue_date:
                     cashflows.append((final_payment_date_dt, self.notional))
            return sorted(cashflows, key=lambda x: x[0])

        for i in range(len(all_schedule_dates) - 1):
            period_start_dt = _date_to_datetime(all_schedule_dates[i])
            payment_date_dt = _date_to_datetime(all_schedule_dates[i+1])

            coupon_freq_numeric, regular_ref_period_days = self._get_coupon_frequency_and_reference_period_days(payment_date_dt)

            if i == 0 or self._schedule.stub: # First period or a stub period
                if self.accrual_day_counter_type == DayCounterType.ActActICMA:
                    year_fraction_for_coupon = self._accrual_day_counter._yf(
                        period_start_dt, payment_date_dt,
                        regular_ref_period_days, coupon_freq_numeric
                    )
                else: # Assumes other relevant day counters (like ActAct ISDA) take 2 args
                    year_fraction_for_coupon = self._accrual_day_counter._yf(
                        period_start_dt, payment_date_dt
                    )
            else: # Regular full period after the first/stub
                year_fraction_for_coupon = 1.0 / coupon_freq_numeric
            
            coupon_amount = self.notional * self.coupon_rate * year_fraction_for_coupon if self.coupon_rate > 0.0 else 0.0
            if coupon_amount > 0.0 or payment_date_dt == _date_to_datetime(all_schedule_dates[-1]): # Add if coupon or if it's the maturity payment date
                cashflows.append((payment_date_dt, coupon_amount))

        
        final_payment_date_from_schedule_gen = _date_to_datetime(all_schedule_dates[-1])
        if cashflows and cashflows[-1][0] == final_payment_date_from_schedule_gen:
            # Notional is added to the last coupon payment
            last_pmt_date, last_amount = cashflows.pop()
            cashflows.append((last_pmt_date, last_amount + self.notional))
        else:
            
            already_has_notional = any(cf_date == final_payment_date_from_schedule_gen and amount == self.notional for cf_date, amount in cashflows)
            if not already_has_notional:
                 cashflows.append((final_payment_date_from_schedule_gen, self.notional))

        return sorted(cashflows, key=lambda x: x[0])

    def _compute_dirty_price(self, discount_curve: 'DiscountCurve') -> float:
        val_date_dt = _date_to_datetime(discount_curve.valuation_date)
        cashflows = self.expected_cashflows()
        pv_cashflows = 0.0
        for c in cashflows:
            if c[0] >= val_date_dt:
                df = discount_curve.value(val_date_dt, c[0])
                pv_cashflows += df * c[1]
        return pv_cashflows

    def compute_accrued_interest(self, valuation_date: Union[date, datetime]) -> float:
        val_date_dt = _date_to_datetime(valuation_date)
        
        if val_date_dt >= self.maturity_date or val_date_dt < self.issue_date:
            return 0.0

        all_coupon_schedule_dates = self._schedule.generate_dates(ends_only=False) 
        
        current_accrual_start = None
        current_accrual_end = None

        if len(all_coupon_schedule_dates) < 2: 
            return 0.0

        
        for i in range(len(all_coupon_schedule_dates) - 1):
            p_start_dt = _date_to_datetime(all_coupon_schedule_dates[i])
            p_end_dt = _date_to_datetime(all_coupon_schedule_dates[i+1])
            
            
            if p_start_dt <= val_date_dt < p_end_dt:
                current_accrual_start = p_start_dt
                current_accrual_end = p_end_dt
                break
        
        if current_accrual_start is None or current_accrual_end is None:
            return 0.0

        
        coupon_frequency, regular_coupon_period_in_days = \
            self._get_coupon_frequency_and_reference_period_days(current_accrual_end)

        if self.accrual_day_counter_type == DayCounterType.ActActICMA:
            accrued_year_fraction = self._accrual_day_counter._yf(
                current_accrual_start, val_date_dt, 
                regular_coupon_period_in_days, coupon_frequency
            )
        else: # Assumes other relevant day counters (like ActAct ISDA) take 2 args
            accrued_year_fraction = self._accrual_day_counter._yf(
                current_accrual_start, val_date_dt
            )

        
        return self.notional * self.coupon_rate * accrued_year_fraction

    def compute_price(self, discount_curve: 'DiscountCurve') -> float:
        """Computes the dirty price of the bond."""
        return self._compute_dirty_price(discount_curve)

    def compute_clean_price(self, discount_curve: 'DiscountCurve') -> float:
        """Computes the clean price of the bond."""
        dirty_price = self._compute_dirty_price(discount_curve)
        accrued = self.compute_accrued_interest(discount_curve.valuation_date)
        return dirty_price - accrued

    def compute_yield(self, dirty_price: float, val_date: datetime,
                        yield_search_lower_bound: float = -0.2, 
                        yield_search_upper_bound: float = 1.5) -> float: 
        valuation_datetime = _date_to_datetime(val_date)
        

        def target_function(r: float)->float:
            dc = DiscountCurve(valuation_date=valuation_datetime, flat_rate=r)
            calculated_dirty_price = self._compute_dirty_price(discount_curve=dc) 
            return calculated_dirty_price - dirty_price 
        
        result =  brentq(target_function, yield_search_lower_bound, yield_search_upper_bound, 
                         full_output = False)
        return result

class DiscountCurve(BaseDatedCurve):
    def __init__(self, 
                 valuation_date: Union[date, datetime], 
                 flat_rate: Optional[float] = 0.05,
                 curve_data: Any = None,
                 day_counter_type: DayCounterType = DayCounterType.Act365Fixed): 
        self.valuation_date = valuation_date
        self._flat_rate = flat_rate
        self._curve_data = curve_data # Placeholder for more complex curve data
        self._day_counter = DayCounter(day_counter_type)

    @property
    def valuation_date(self) -> datetime:
        return self._valuation_date

    @valuation_date.setter
    def valuation_date(self, value: Union[date, datetime]):
        self._valuation_date = _date_to_datetime(value)

    def get_discount_factor(self, target_date: Union[date, datetime]) -> float:
        val_date_dt = _date_to_datetime(self.valuation_date)
        target_date_dt = _date_to_datetime(target_date)
        
        if target_date_dt < val_date_dt:
            return 0.0 
        time_to_maturity_years = self._day_counter.yf(val_date_dt, target_date_dt)
        rate_to_use = self._flat_rate if self._flat_rate is not None else 0.02 # Fallback if flat_rate is None
        return 1 / ((1 + rate_to_use) ** time_to_maturity_years)
    
    def value(self, ref_date: datetime, target_date: datetime) -> float:
        # Ensure ref_date matches the curve's valuation_date for this simple implementation
        if _date_to_datetime(ref_date).date() != self.valuation_date.date():
            raise ValueError(f"Reference date {ref_date} does not match DiscountCurve valuation date {self.valuation_date}")
        return self.get_discount_factor(target_date)

def run_comparison_example():
    # Bond Parameters
    obj_id = 'DE000CZ40NT7'
    issue_date_dt = date(2019, 1, 1)
    maturity_date_dt = date(2022, 5, 26 )
    coupon_rate_val = 0.025  
    tenor_period = Period(years=1)  
    notional_val = 100000.0
    currency_val = 'EUR'
    issuer_val = 'Commerzbank'
    securitisation_level_val = SecuritizationLevel.NON_PREFERRED_SENIOR
    valuation_date_dt = date(2021, 1, 15)
    flat_rate_val = 0.02 
    
    # RiVaPy Schedule
    rivapy_schedule = Schedule(start_day=issue_date_dt,           
                               end_day=maturity_date_dt,           
                               time_period=tenor_period,       
                               backwards=True,
                               stub=True, 
                               business_day_convention=RollConvention.FOLLOWING,
                               calendar=ECB(years=range(issue_date_dt.year, maturity_date_dt.year + 2)))

    # RiVaPy Bond
    rivapy_bond = FixedRateBond(obj_id=obj_id,
                                schedule=rivapy_schedule, 
                                notional=notional_val,
                                currency=currency_val,
                                issue_date=issue_date_dt,
                                coupon_rate=coupon_rate_val,
                                issuer=issuer_val,
                                securitization_level=securitisation_level_val, 
                                accrual_day_counter_type=DayCounterType.ACT_ACT) # Changed to ACT_ACT for ActActISDA
    
    
       
    rivapy_discount_curve = DiscountCurve(valuation_date=valuation_date_dt, 
                                          flat_rate=flat_rate_val,
                                          day_counter_type=DayCounterType.Act365Fixed)  

    # compute_price now returns the DIRTY price
    rivapy_dirty_price = rivapy_bond.compute_price(rivapy_discount_curve)
    rivapy_accrued_interest = rivapy_bond.compute_accrued_interest(valuation_date_dt)
    rivapy_clean_price = rivapy_dirty_price - rivapy_accrued_interest #rivapy_bond.compute_clean_price(rivapy_discount_curve)
    rivapy_ytm = rivapy_bond.compute_yield(dirty_price=rivapy_dirty_price, val_date=valuation_date_dt)

    # Debug logging for RiVaPy calculations
    logger.debug("RiVaPy Cashflows:")
    for cf_date, amount in rivapy_bond.expected_cashflows():
        logger.debug(f"Date: {cf_date.strftime('%Y-%m-%d')}, Amount: {amount:.4f}")
    logger.debug(f"RiVaPy Accrued Interest: {rivapy_accrued_interest:.4f}")
    logger.debug(f"RiVaPy Discount Factors:")
    val_date_dt = _date_to_datetime(valuation_date_dt)
    for cf_date, amount in rivapy_bond.expected_cashflows():
        if cf_date >= val_date_dt:
            df = rivapy_discount_curve.value(val_date_dt, cf_date)
            logger.debug(f"Date: {cf_date.strftime('%Y-%m-%d')}, DF: {df:.8f}")

    # QuantLib calculations
    ql_valuation_date = ql.Date(valuation_date_dt.day, valuation_date_dt.month, valuation_date_dt.year)
    ql.Settings.instance().evaluationDate = ql_valuation_date

    
    ql_calendar = ql.TARGET() 
    ql_convention = ql.Following 
    
    
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

    
    ql_accrual_day_counter = ql.ActualActual(ql.ActualActual.ISDA) # Changed to ISDA
    ql_bond = ql.FixedRateBond(0, # SettlementTage
                               notional_val,
                               ql_schedule,
                               [coupon_rate_val], 
                               ql_accrual_day_counter,
                               ql_convention, 
                               100.0,
                               ql_issue_date)
                                

    
    ql_discount_day_counter = ql.Actual365Fixed()
    ql_discount_curve = ql.FlatForward(ql_valuation_date,
                                       ql.QuoteHandle(ql.SimpleQuote(flat_rate_val)),
                                       ql_discount_day_counter,
                                       ql.Compounded, 
                                       ql.Annual) 

    
    ql_bond_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(ql_discount_curve))
    ql_bond.setPricingEngine(ql_bond_engine)

    
    ql_clean_price_per_100 = ql_bond.cleanPrice()
    ql_dirty_price_per_100 = ql_bond.dirtyPrice()
    ql_accrued_amount_per_100 = ql_bond.accruedAmount()

    
    scale_factor = notional_val / 100.0
    ql_absolute_clean_price = ql_clean_price_per_100 * scale_factor
    ql_absolute_dirty_price = ql_dirty_price_per_100 * scale_factor
    ql_absolute_accrued_amount = ql_accrued_amount_per_100 * scale_factor

    ql_dirty_price_object = ql.BondPrice(ql_dirty_price_per_100, ql.BondPrice.Dirty)

    ql_ytm = ql_bond.bondYield(ql_dirty_price_object, 
                               ql_accrual_day_counter,    
                               ql.Compounded,             
                               ql.Annual,                 
                               ql_valuation_date,         
                               1.0e-10,                  
                               200)                       

    
    logger.debug("\nQuantLib Cashflows:")
    for cf in ql_bond.cashflows():
        logger.debug(f"Date: {cf.date().ISO()}, Amount: {cf.amount():.4f}")
    logger.debug(f"QuantLib Accrued Interest: {ql_absolute_accrued_amount:.4f}")
    logger.debug(f"QuantLib Discount Factors:")
    for cf in ql_bond.cashflows():
        if cf.date() >= ql_valuation_date:
            t = ql_discount_day_counter.yearFraction(ql_valuation_date, cf.date())
            df = ql_discount_curve.discount(cf.date())
            logger.debug(f"Date: {cf.date().ISO()}, DF: {df:.8f}")

    # Compare differences
    logger.debug("\nDetailed Differences:")
    logger.debug(f"Accrued Interest Diff: {rivapy_accrued_interest - ql_absolute_accrued_amount:.4f}")
    logger.debug(f"Clean Price Diff: {rivapy_clean_price - ql_absolute_clean_price:.4f}")
    logger.debug(f"Dirty Price Diff: {rivapy_dirty_price - ql_absolute_dirty_price:.4f}")
    logger.debug(f"YTM Diff: {(rivapy_ytm - ql_ytm)*100:.4f}%")

    # --- Vergleich RiVaPy vs QuantLib ---
    print("\n--- Comparison Summary ---")
    print(f"Clean Price Difference (RiVaPy Clean - QuantLib Absolute Clean): {currency_val} {rivapy_clean_price - ql_absolute_clean_price:,.4f}")
    print(f"Dirty Price Difference (RiVaPy Dirty - QuantLib Absolute Dirty): {currency_val} {rivapy_dirty_price - ql_absolute_dirty_price:,.4f}")
    print(f"Yield Difference (RiVaPy - QuantLib): {(rivapy_ytm - ql_ytm)*100:.4f}%")
    print(f"RiVaPy Clean Price: {rivapy_clean_price:,.4f}, QuantLib Absolute Clean Price: {ql_absolute_clean_price:,.4f}")


if __name__ == '__main__':
    run_comparison_example()