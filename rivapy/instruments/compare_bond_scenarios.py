import logging
from datetime import date, datetime
import QuantLib as ql

from rivapy.instruments.rewrite import FixedRateBond, DiscountCurve
from rivapy.tools.datetools import Schedule, Period, DayCounterType
from rivapy.tools.enums import RollConvention, SecuritizationLevel
from holidays.financial import ECB

# --- Logging Konfiguration ---
# Standardmäßig wird der Logger auf INFO gesetzt, um nur Zusammenfassungen zu sehen.
# Wenn ein Test fehlschlägt, wird der Logger temporär auf DEBUG gesetzt, um Details zu sehen.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Default level

# Konfigurieren Sie den Handler, um Ausgaben auf der Konsole zu sehen
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Stellen Sie sicher, dass der Logger in rivapy.instruments.rewrite auch auf DEBUG gesetzt ist,
# wenn wir detaillierte Ausgaben wünschen.
rivapy_rewrite_logger = logging.getLogger('rivapy.instruments.rewrite')


def run_single_comparison(
    scenario_name: str,
    issue_date: date,
    maturity_date: date,
    coupon_rate: float,
    tenor_period: Period,
    valuation_date: date,
    roll_convention: RollConvention,
    stub_type: bool,
    flat_rate: float = 0.02,
    notional: float = 100000.0,
    currency: str = 'EUR',
    tolerance: float = 1e-6 # Toleranz für Gleitkommavergleiche
) -> bool:
    """
    Führt einen Vergleich für ein einzelnes Anleiheszenario durch.
    Gibt True zurück, wenn alle Werte übereinstimmen, False sonst.
    """
    logger.info(f"\n--- Running Scenario: {scenario_name} ---")
    
    # Temporär den Detaillierungsgrad des Loggers erhöhen, falls Abweichungen auftreten
    original_level = logger.level
    original_rewrite_level = rivapy_rewrite_logger.level
    
    # --- RiVaPy Setup ---
    rivapy_schedule = Schedule(start_day=issue_date,           
                               end_day=maturity_date,           
                               time_period=tenor_period,       
                               backwards=True, # QL's Backward generation is common for bonds
                               stub=stub_type, 
                               business_day_convention=roll_convention,
                               calendar=ECB(years=range(issue_date.year -1, maturity_date.year + 2))) # Extended year range for calendar

    rivapy_bond = FixedRateBond(obj_id=scenario_name,
                                schedule=rivapy_schedule, 
                                notional=notional,
                                currency=currency,
                                coupon_rate=coupon_rate,
                                accrual_day_counter_type=DayCounterType.ActActICMA) 
    
    rivapy_discount_curve = DiscountCurve(valuation_date=valuation_date, 
                                          flat_rate=flat_rate,
                                          day_counter_type=DayCounterType.Act365Fixed)  

    rivapy_dirty_price = rivapy_bond.compute_price(rivapy_discount_curve)
    rivapy_accrued_interest = rivapy_bond.compute_accrued_interest(valuation_date)
    rivapy_clean_price = rivapy_dirty_price - rivapy_accrued_interest 
    rivapy_ytm = rivapy_bond.compute_yield(dirty_price=rivapy_dirty_price, val_date=valuation_date)

    # --- QuantLib Setup ---
    ql_valuation_date = ql.Date(valuation_date.day, valuation_date.month, valuation_date.year)
    ql.Settings.instance().evaluationDate = ql_valuation_date

    ql_calendar = ql.TARGET() # ECB calendar in QuantLib
    
    # Map RiVaPy RollConvention to QuantLib BusinessDayConvention
    ql_roll_convention_map = {
        RollConvention.FOLLOWING: ql.Following,
        RollConvention.MODIFIED_FOLLOWING: ql.ModifiedFollowing,
        RollConvention.PRECEDING: ql.Preceding,
        RollConvention.MODIFIED_PRECEDING: ql.ModifiedPreceding,
        RollConvention.NEAREST: ql.Nearest,
        RollConvention.UNADJUSTED: ql.Unadjusted
    }
    ql_convention = ql_roll_convention_map.get(roll_convention, ql.ModifiedFollowing) # Fallback

    ql_issue_date = ql.Date(issue_date.day, issue_date.month, issue_date.year)
    ql_maturity_date = ql.Date(maturity_date.day, maturity_date.month, maturity_date.year)
    
    # QL Schedule: DateGeneration.Backward is common for bonds, aligns with RiVaPy's backwards=True
    # is_long_stub=False for short stub, True for long stub. RiVaPy's stub=True allows short/long.
    # For simplicity, we'll use False for is_long_stub to match common bond practices.
    ql_schedule = ql.Schedule(ql_issue_date, 
                              ql_maturity_date, 
                              ql.Period(ql.Annual), # Assuming annual for simplicity, adjust if tenor is different
                              ql_calendar,
                              ql_convention,
                              ql_convention, 
                              ql.DateGeneration.Backward, 
                              False) # is_long_stub = False

    ql_accrual_day_counter = ql.ActualActual(ql.ActualActual.ISMA) 
    ql_bond = ql.FixedRateBond(0, # Settlement days
                               notional,
                               ql_schedule,
                               [coupon_rate], 
                               ql_accrual_day_counter,
                               ql_convention, # Payment convention
                               100.0, # Redemption
                               ql_issue_date)
                                
    ql_discount_day_counter = ql.Actual365Fixed()
    ql_discount_curve = ql.FlatForward(ql_valuation_date,
                                       ql.QuoteHandle(ql.SimpleQuote(flat_rate)),
                                       ql_discount_day_counter,
                                       ql.Compounded, 
                                       ql.Annual) 

    ql_bond_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(ql_discount_curve))
    ql_bond.setPricingEngine(ql_bond_engine)

    ql_clean_price_per_100 = ql_bond.cleanPrice()
    ql_dirty_price_per_100 = ql_bond.dirtyPrice()
    ql_accrued_amount_per_100 = ql_bond.accruedAmount()

    scale_factor = notional / 100.0
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

    # --- Vergleich ---
    all_match = True
    
    if abs(rivapy_accrued_interest - ql_absolute_accrued_amount) > tolerance:
        all_match = False
    if abs(rivapy_clean_price - ql_absolute_clean_price) > tolerance:
        all_match = False
    if abs(rivapy_dirty_price - ql_absolute_dirty_price) > tolerance:
        all_match = False
    if abs(rivapy_ytm - ql_ytm) > tolerance * 10: # Erhöhte Toleranz für YTM
        all_match = False

    if not all_match:
        logger.setLevel(logging.DEBUG) # Detaillierte Logs bei Abweichung
        rivapy_rewrite_logger.setLevel(logging.DEBUG) # Auch interne RiVaPy Logs aktivieren
        
        logger.debug(f"Scenario: {scenario_name} - DISCREPANCY FOUND!")
        logger.debug(f"  RiVaPy Clean Price: {rivapy_clean_price:.4f}")
        logger.debug(f"  QL Clean Price:     {ql_absolute_clean_price:.4f}")
        logger.debug(f"  Diff Clean Price:   {rivapy_clean_price - ql_absolute_clean_price:.4f}")

        logger.debug(f"  RiVaPy Dirty Price: {rivapy_dirty_price:.4f}")
        logger.debug(f"  QL Dirty Price:     {ql_absolute_dirty_price:.4f}")
        logger.debug(f"  Diff Dirty Price:   {rivapy_dirty_price - ql_absolute_dirty_price:.4f}")

        logger.debug(f"  RiVaPy Accrued:     {rivapy_accrued_interest:.4f}")
        logger.debug(f"  QL Accrued:         {ql_absolute_accrued_amount:.4f}")
        logger.debug(f"  Diff Accrued:       {rivapy_accrued_interest - ql_absolute_accrued_amount:.4f}")

        logger.debug(f"  RiVaPy YTM:         {rivapy_ytm:.6f}")
        logger.debug(f"  QL YTM:             {ql_ytm:.6f}")
        logger.debug(f"  Diff YTM:           {(rivapy_ytm - ql_ytm)*100:.4f}%")
        
        logger.setLevel(original_level) # Logging Level zurücksetzen
        rivapy_rewrite_logger.setLevel(original_rewrite_level)
        return False
    else:
        logger.info(f"Scenario: {scenario_name} - All values match within tolerance.")
        return True

# --- Test Szenarien ---
test_scenarios = [
    {
        "scenario_name": "Scenario 1: Short First Stub (Issue on Holiday, Val on First Coupon)",
        "issue_date": date(2019, 1, 1), # New Year's Day
        "maturity_date": date(2023, 1, 1), # New Year's Day
        "coupon_rate": 0.025,
        "tenor_period": Period(years=1),
        "valuation_date": date(2020, 1, 2), # First adjusted coupon date
        "roll_convention": RollConvention.FOLLOWING,
        "stub_type": True,
    },
    {
        "scenario_name": "Scenario 2: Long Last Stub (Maturity after last full period)",
        "issue_date": date(2019, 1, 1),
        "maturity_date": date(2023, 3, 15), # Long stub after 2023-01-01
        "coupon_rate": 0.03,
        "tenor_period": Period(years=1),
        "valuation_date": date(2022, 6, 1),
        "roll_convention": RollConvention.FOLLOWING,
        "stub_type": True,
    },
    {
        "scenario_name": "Scenario 3: Leap Year Period (Valuation after leap day)",
        "issue_date": date(2019, 2, 1),
        "maturity_date": date(2021, 2, 1),
        "coupon_rate": 0.04,
        "tenor_period": Period(years=1),
        "valuation_date": date(2020, 3, 1), # After Feb 29, 2020
        "roll_convention": RollConvention.FOLLOWING,
        "stub_type": True,
    },
    {
        "scenario_name": "Scenario 4: Valuation Date just before Coupon",
        "issue_date": date(2020, 1, 15),
        "maturity_date": date(2024, 1, 15),
        "coupon_rate": 0.05,
        "tenor_period": Period(years=1),
        "valuation_date": date(2021, 1, 14), # Day before coupon
        "roll_convention": RollConvention.FOLLOWING,
        "stub_type": True,
    },
    {
        "scenario_name": "Scenario 5: Valuation Date just after Coupon",
        "issue_date": date(2020, 1, 15),
        "maturity_date": date(2024, 1, 15),
        "coupon_rate": 0.05,
        "tenor_period": Period(years=1),
        "valuation_date": date(2021, 1, 16), # Day after coupon
        "roll_convention": RollConvention.FOLLOWING,
        "stub_type": True,
    },
    {
        "scenario_name": "Scenario 6: Issue Date on Weekend, Maturity on Holiday",
        "issue_date": date(2019, 10, 26), # Saturday
        "maturity_date": date(2023, 12, 25), # Christmas Day
        "coupon_rate": 0.035,
        "tenor_period": Period(years=1),
        "valuation_date": date(2021, 5, 1),
        "roll_convention": RollConvention.FOLLOWING,
        "stub_type": True,
    },
]

# --- Hauptausführung ---
if __name__ == "__main__":
    all_tests_passed = True
    for scenario in test_scenarios:
        passed = run_single_comparison(**scenario)
        if not passed:
            all_tests_passed = False

    if all_tests_passed:
        logger.info("\nAll bond comparison scenarios passed successfully. No discrepancies found.")
    else:
        logger.error("\nSome bond comparison scenarios failed. Check logs for details.")