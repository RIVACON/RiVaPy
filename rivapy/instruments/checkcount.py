from datetime import date
import QuantLib as ql
from rivapy.tools.datetools import DayCounter
from dateutil.relativedelta import relativedelta
from rivapy.tools.enums import DayCounterType

test_periods = [
    {"id": "Standard", "d1": date(2023, 1, 15), "d2": date(2024, 7, 20),
     "ref_start": date(2024, 1, 15), "ref_end": date(2024, 7, 15), "freq": 2},
    {"id": "LeapYearSpan", "d1": date(2023, 11, 1), "d2": date(2024, 5, 1), # d2 is in a leap year
     "ref_start": date(2023, 11, 1), "ref_end": date(2024, 5, 1), "freq": 2},
    {"id": "SameYearNonLeap", "d1": date(2023, 2, 1), "d2": date(2023, 8, 1),
     "ref_start": date(2023, 2, 1), "ref_end": date(2023, 8, 1), "freq": 2},
    {"id": "SameYearLeap", "d1": date(2024, 2, 1), "d2": date(2024, 8, 1), # Both in leap year
     "ref_start": date(2024, 2, 1), "ref_end": date(2024, 8, 1), "freq": 2},
    {"id": "ShortPeriod", "d1": date(2023, 3, 1), "d2": date(2023, 3, 15),
     "ref_start": date(2023, 3, 1), "ref_end": date(2023, 9, 1), "freq": 2},
    {"id": "30_360_Feb_End_NonLeap", "d1": date(2023, 1, 30), "d2": date(2023, 2, 28),
     "ref_start": date(2023, 1, 15), "ref_end": date(2023, 7, 15), "freq": 2},
    {"id": "30_360_Feb_End_Leap", "d1": date(2024, 1, 30), "d2": date(2024, 2, 29),
     "ref_start": date(2024, 1, 15), "ref_end": date(2024, 7, 15), "freq": 2},
    {"id": "AcrossYearBoundary", "d1": date(2023, 12, 15), "d2": date(2024, 1, 15),
     "ref_start": date(2023, 7, 15), "ref_end": date(2024, 1, 15), "freq": 2},
]

def run_comparison(dc_name: str, riva_dc_type: DayCounterType, ql_dc_instance: ql.DayCounter, 
                   period_details: dict, is_icma: bool = False, epsilon: float = 1e-9) -> str:
    d1 = period_details['d1']
    d2 = period_details['d2']
    ql_d1 = ql.Date(d1.day, d1.month, d1.year)
    ql_d2 = ql.Date(d2.day, d2.month, d2.year)

    status_to_return = 'error'  

    try:
        dc_riva = DayCounter(riva_dc_type)
        if is_icma:
            coupon_freq = period_details['freq']
            
            # Determine period_step based on frequency
            if coupon_freq == 1: # Annual
                period_step = relativedelta(years=1)
            elif coupon_freq == 2: # Semi-annual
                period_step = relativedelta(months=6)
            elif coupon_freq == 4: # Quarterly
                period_step = relativedelta(months=3)
            else: # Default or error for unsupported frequencies in test
                period_step = relativedelta(years=1) 
                # Consider raising an error if freq is unexpected for ICMA tests

            # Generate a comprehensive coupon_schedule_list for RiVaPy
            # This schedule should bracket the d1 and d2 of the test period.
            # Anchor the schedule generation around period_details['ref_start']
            anchor_date = period_details['ref_start']
            
            current_schedule_nodes = {anchor_date}
            
            # Go backwards from anchor to cover d1
            temp_date = anchor_date
            while temp_date > d1:
                temp_date -= period_step
                current_schedule_nodes.add(temp_date)
            
            # Go forwards from anchor to cover d2
            temp_date = anchor_date
            while temp_date < d2:
                temp_date += period_step
                current_schedule_nodes.add(temp_date)

            # Add one more period at each end for safety with stubs
            current_schedule_nodes.add(min(current_schedule_nodes) - period_step)
            current_schedule_nodes.add(max(current_schedule_nodes) + period_step)
            
            coupon_schedule_list = sorted(list(current_schedule_nodes))
            yf_riva = dc_riva._yf(d1, d2, coupon_schedule_list, coupon_freq) 
        else:
            yf_riva = dc_riva.yf(d1, d2)

        if is_icma:
            # Parameters for QuantLib call should use the original ref_start and ref_end
            # from test_periods to define the coupon structure.
            ql_ref_s_param = ql.Date(period_details['ref_start'].day, period_details['ref_start'].month, period_details['ref_start'].year)
            ql_ref_e_param = ql.Date(period_details['ref_end'].day, period_details['ref_end'].month, period_details['ref_end'].year)
            
            yf_ql = ql_dc_instance.yearFraction(ql_d1, ql_d2, ql_ref_s_param, ql_ref_e_param)
        else:
            yf_ql = ql_dc_instance.yearFraction(ql_d1, ql_d2)
        
        diff = yf_riva - yf_ql
        abs_diff = abs(diff)

        if abs_diff <= epsilon:
            status_to_return = 'match'
        else:
            status_to_return = 'deviation'
            print(f"  Test Case ID: {period_details['id']} ({period_details['d1']} to {period_details['d2']})")
            print(f"    Rivapy ({dc_name}): {yf_riva:.12f}")
            print(f"    QuantLib ({ql_dc_instance.name()}): {yf_ql:.12f}")
            print(f"    Difference: {diff:.12f}")
            print("-" * 10)

    except Exception as e:
        # status_to_return is already 'error'
        print(f"  Test Case ID: {period_details['id']} ({period_details['d1']} to {period_details['d2']})")
        print(f"    Error comparing {dc_name} for period {period_details['id']}: {e}")
        print("-" * 10)
        
    return status_to_return

def compare_day_counters():
    """
    Compares day counting methods from rivapy.tools.datetools with QuantLib.
    """
    print("Starting Day Counter Comparison with QuantLib\n" + "="*50)
    epsilon = 1e-9  # Tolerance for float comparison

    conventions_to_test = [
        {"name": "Act/365 Fixed", "riva_type": DayCounterType.Act365Fixed, "ql_instance": ql.Actual365Fixed(), "is_icma": False},
        {"name": "Act/Act ISDA", "riva_type": DayCounterType.ACT_ACT, "ql_instance": ql.ActualActual(ql.ActualActual.ISDA), "is_icma": False},
        {"name": "Act/360", "riva_type": DayCounterType.ACT360, "ql_instance": ql.Actual360(), "is_icma": False},
        {"name": "ThirtyU/360 (Bond Basis)", "riva_type": DayCounterType.ThirtyU360, "ql_instance": ql.Thirty360(ql.Thirty360.BondBasis), "is_icma": False},
        {"name": "ThirtyE/360 (Eurobond Basis)", "riva_type": DayCounterType.ThirtyE360, "ql_instance": ql.Thirty360(ql.Thirty360.EurobondBasis), "is_icma": False},
        {"name": "Thirty/360 ISDA", "riva_type": DayCounterType.Thirty360ISDA, "ql_instance": ql.Thirty360(ql.Thirty360.ISDA), "is_icma": False},
        {"name": "Act/Act ICMA", "riva_type": DayCounterType.ActActICMA, "ql_instance": ql.ActualActual(ql.ActualActual.ISMA), "is_icma": True},
        # ISMA (ActualActual.ISMA) is often used for bonds and is similar to ICMA.
        # Rivapy's ActActICMA is (d2-d1).days / (coupon_period_in_days * coupon_frequency)
        # QL's ISMA is more complex and uses reference periods.
    ]

    total_tests = 0
    successful_matches = 0
    reported_deviations = 0
    error_cases = 0

    for conv_details in conventions_to_test:
        print(f"\n--- Comparing: {conv_details['name']} ---")
        
        convention_total_tests = 0
        convention_matches = 0
        convention_deviations = 0
        convention_errors = 0

        for period in test_periods:
            total_tests += 1
            convention_total_tests +=1
            
            status = run_comparison(
                dc_name=conv_details['name'],
                riva_dc_type=conv_details['riva_type'],
                ql_dc_instance=conv_details['ql_instance'],
                period_details=period,
                is_icma=conv_details['is_icma'],
                epsilon=epsilon
            )

            if status == 'match':
                successful_matches += 1
                convention_matches += 1
            elif status == 'deviation':
                reported_deviations += 1
                convention_deviations += 1
            elif status == 'error':
                error_cases += 1
                convention_errors += 1
        
        if convention_total_tests > convention_matches: # If any issue in this convention
            print(f"--- Summary for {conv_details['name']}: Matches: {convention_matches}/{convention_total_tests}, Deviations: {convention_deviations}, Errors: {convention_errors} ---")
        else:
            print(f"--- All tests passed for {conv_details['name']} ({convention_matches}/{convention_total_tests}) ---")
        print("=" * 30)

    print("\n" + "="*50)
    print("Overall Day Counter Comparison Summary:")
    print("="*50)
    print(f"Total Test Cases Executed: {total_tests}")
    print(f"Successful Matches (Difference <= {epsilon:.0e}): {successful_matches}")
    print(f"Reported Deviations (Difference > {epsilon:.0e}): {reported_deviations}")
    print(f"Errors during comparison: {error_cases}")
    print("="*50 + "\nComparison Finished.")

if __name__ == "__main__":
    compare_day_counters()