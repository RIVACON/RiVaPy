from datetime import date
from typing import Dict, Any, List, Tuple
from holidays import HolidayBase, country_holidays
from holidays.financial import ECB
import QuantLib as ql

from rivapy.tools.datetools import roll_day, _date_to_datetime
from rivapy.tools.enums import RollConvention

# Helper function to get calendar objects for RiVaPy (using 'holidays' library)
def get_calendar_object(calendar_str: str, years: Any) -> HolidayBase:
    """
    Retrieves a RiVaPy holiday calendar object based on a string identifier.
    'years' can be a range, list, or int.
    """
    # Ensure years is a valid format for the holidays library
    if isinstance(years, range):
        years_list = list(years)
    elif isinstance(years, int):
        years_list = [years]
    elif isinstance(years, list):
        years_list = years
    else:
        raise TypeError("Years argument must be a range, list of years, or a single year int.")


    if calendar_str.upper() == "ECB":
        return ECB(years=years_list)
    elif calendar_str.upper() == "DE":
        return country_holidays("DE", years=years_list)
    elif calendar_str.upper() == "US":
        return country_holidays("US", years=years_list)
    elif calendar_str.upper() == "GB":
        return country_holidays("GB", years=years_list)
    
    else:
        try:
            return country_holidays(calendar_str, years=years_list)
        except KeyError:
            raise ValueError(f"Calendar string '{calendar_str}' is not recognized as a country code or special calendar for RiVaPy.")

def get_ql_calendar_object(calendar_str: str) -> ql.Calendar:
    """
    Retrieves a QuantLib calendar object based on a string identifier.
    """
    cal_str_upper = calendar_str.upper()
    if cal_str_upper == "ECB" or cal_str_upper == "TARGET": # TARGET is the QL equivalent for ECB
        return ql.TARGET()
    elif cal_str_upper == "DE":
        return ql.Germany(ql.Germany.FrankfurtStockExchange) # Or Eurex, Xetra, Settlement
    elif cal_str_upper == "US":
        # More specific US calendars exist, e.g., ql.UnitedStates.NYSE, .FederalReserve, .GovernmentBond
        return ql.UnitedStates(ql.UnitedStates.Settlement) 
    elif cal_str_upper == "GB" or cal_str_upper == "UK":
        return ql.UnitedKingdom(ql.UnitedKingdom.Exchange) # Or Metals, Settlement
    # Add more specific QL calendars if needed
    else:
        # Attempt to get a generic calendar if QL supports the country code directly
        try:
            # QuantLib might have calendars like ql.Sweden(), ql.Japan() etc.
            # This requires the calendar_str to match a QL calendar class name.
            if hasattr(ql, calendar_str):
                 return getattr(ql, calendar_str)()
            # For two-letter country codes, some might be implicitly available
            # but it's less reliable than explicit mapping.
        except Exception:
            pass # Fall through to raise error
        raise ValueError(f"Calendar string '{calendar_str}' is not recognized for QuantLib or needs specific mapping.")


# Define test cases for business day conventions.
# 'expected_results' is removed as QL will be the benchmark.
test_business_days = [
    {
        "id": "RegularBizDay_NoChange",
        "date_to_roll": date(2023, 10, 26),  # Thursday
        "calendar_str": "ECB",
        "start_date_for_eom": None,
    },
    {
        "id": "Saturday_Rolls",
        "date_to_roll": date(2023, 10, 28),  # Saturday
        "calendar_str": "ECB",
        "start_date_for_eom": None,
    },
    {
        "id": "Sunday_Rolls", 
        "date_to_roll": date(2023, 10, 29),  # Sunday
        "calendar_str": "ECB",
        "start_date_for_eom": None,
    },
    {
        "id": "Holiday_Christmas_Rolls",
        "date_to_roll": date(2023, 12, 25),  # Monday, Christmas Day
        "calendar_str": "ECB", # ECB has 25th and 26th as holidays
        "start_date_for_eom": None,
    },
    {
        "id": "Holiday_NewYear_Rolls",
        "date_to_roll": date(2024, 1, 1),  # Monday, New Year's Day
        "calendar_str": "ECB", 
        "start_date_for_eom": None,
    },
    {
        "id": "ModFollowing_CrossMonth",
        "date_to_roll": date(2023, 4, 29),  # Saturday. May 1st (Mon) is ECB holiday.
        "calendar_str": "ECB",
        "start_date_for_eom": None,
    },
    {
        "id": "ModPreceding_CrossMonth",
        "date_to_roll": date(2023, 5, 1),  # Monday, Labour Day (ECB holiday)
        "calendar_str": "ECB",
        "start_date_for_eom": None,
    },
    # Test cases specifically for RiVaPy's EOM and BiMonthly logic,
    # as QL doesn't have direct equivalents for these with a simple 'adjust' call.
    {
        "id": "ModFollowingEOM_StartIsEOM_RiVaPy", 
        "date_to_roll": date(2023, 4, 29),  # Saturday. April 28th is last biz day.
        "calendar_str": "ECB",
        "start_date_for_eom": date(2023, 3, 31), # Friday, last biz day of March for ECB
    },
    {
        "id": "ModFollowingEOM_StartNotEOM_RiVaPy", 
        "date_to_roll": date(2023, 4, 29),  # Saturday. May 1st (Mon) is ECB holiday.
        "calendar_str": "ECB",
        "start_date_for_eom": date(2023, 3, 30), # Thursday, NOT last biz day of March
    },
    {
        "id": "ModFollowingBimonthly_CrossMidMonth_RiVaPy", 
        "date_to_roll": date(2023, 10, 14),  # Saturday. Following would be Monday 16th.
        "calendar_str": "ECB", 
        "start_date_for_eom": None,
    },
    {
        "id": "ModFollowingBimonthly_NoCross_RiVaPy", 
        "date_to_roll": date(2023, 10, 7),  # Saturday. Following would be Monday 9th.
        "calendar_str": "ECB",
        "start_date_for_eom": None,
    },
    {
        "id": "Nearest_MidWeekHoliday_DE", # Ascension Day in Germany is often a Thursday
        "date_to_roll": date(2023, 5, 18), # Thursday, Ascension Day (DE holiday)
        "calendar_str": "ECB", # German calendar
        "start_date_for_eom": None,
        # Expected QL: Following (2023-05-19, Fri) if 17th and 19th are biz days
        # RiVaPy's nearest_business_day default is following_first=True, so should match.
    },
    {
        "id": "Nearest_Saturday_BetweenHolidays_ECB", # Example: Sat between Good Friday and Easter Monday (if they were biz days around it)
        "date_to_roll": date(2023, 4, 8), # Saturday. Good Friday (7th) is holiday, Easter Monday (10th) is holiday for ECB.
        "calendar_str": "ECB",          # So, nearest biz days are Thurs 6th and Tues 11th.
        "start_date_for_eom": None,      # QL's Nearest should pick Tuesday 11th.
    },
    {
        "id": "ModFollowingEOM_WeekendAtMonthEnd_StartIsEOM_RiVaPy",
        "date_to_roll": date(2023, 9, 30),  # Saturday. September 29th (Fri) is last biz day.
        "calendar_str": "ECB", # No holiday on Sep 29/30 or Oct 1/2 for ECB
        "start_date_for_eom": date(2023, 8, 31), # Thursday, last biz day of August
        # Expected RiVaPy: 2023-09-29
    },
    {
        "id": "ModFollowingEOM_WeekendAtMonthEnd_StartNotEOM_RiVaPy",
        "date_to_roll": date(2023, 9, 30),  # Saturday.
        "calendar_str": "ECB",
        "start_date_for_eom": date(2023, 8, 30), # Wednesday, NOT last biz day of August
        # Expected RiVaPy: Should behave like ModifiedFollowing -> 2023-10-02 (Monday)
        # (as Sep 30 is Sat, Oct 1 is Sun)
    }
]

def run_business_day_comparison(
        conv_details: Dict[str, Any], # Contains RiVaPy and QL convention info
        test_case: Dict[str, Any]
    ) -> str:
    """
    Compares RiVaPy's roll_day result against QuantLib's calendar.adjust().
    """
    date_to_roll = test_case["date_to_roll"]
    calendar_str = test_case["calendar_str"]
    original_start_date_eom = test_case.get("start_date_for_eom") 

    riva_convention_enum = conv_details["riva_enum"]
    ql_convention_enum = conv_details.get("ql_enum") 

    effective_start_date_eom = original_start_date_eom
    default_eom_note = ""
    if riva_convention_enum == RollConvention.MODIFIED_FOLLOWING_EOM and original_start_date_eom is None:
        # If MODIFIED_FOLLOWING_EOM is used and no start_date_for_eom is provided,
        # default to date_to_roll. This ensures the function can be called.
        # The EOM logic will depend on whether date_to_roll is an EOM business day.
        effective_start_date_eom = date_to_roll 
        print(f"  Test Case ID: {test_case['id']}")
        print(f"    Convention: {conv_details['name']}")
        print(f"    Note: 'start_date_for_eom' not provided. Defaulting to date_to_roll ({effective_start_date_eom}) for MODIFIED_FOLLOWING_EOM.")
        default_eom_note = f" (defaulted start_eom to {effective_start_date_eom})"


    # Use a buffer for RiVaPy calendar instantiation to handle rolls across year-end
    # The 'holidays' library typically needs the years it should consider.
    # For a date like 2023-12-31 rolling to 2024-01-02, both years are relevant.
    relevant_years_for_riva = {date_to_roll.year}
    if effective_start_date_eom: # Use effective date here
        relevant_years_for_riva.add(effective_start_date_eom.year)
    # Add adjacent years for safety margin during rolling
    min_rel_year = min(relevant_years_for_riva)
    max_rel_year = max(relevant_years_for_riva)
    year_range_for_riva_cal = list(range(min_rel_year - 1, max_rel_year + 2))
    
    try:
        riva_calendar_obj = get_calendar_object(calendar_str, year_range_for_riva_cal)
    except ValueError as e:
        print(f"  Test Case ID: {test_case['id']}")
        print(f"    Convention: {conv_details['name']}")
        print(f"    Could not get RiVaPy calendar for '{calendar_str}': {e}")
        return 'error_riva_calendar'
        
    ql_calendar_obj = None
    if ql_convention_enum is not None: # Only get QL calendar if we plan to use it
        try:
            ql_calendar_obj = get_ql_calendar_object(calendar_str)
        except ValueError as e:
            print(f"  Test Case ID: {test_case['id']}")
            print(f"    Convention: {conv_details['name']}")
            print(f"    Could not get QL calendar for '{calendar_str}': {e}")
            return 'error_ql_calendar'

    status_to_return = 'error' # Default status
    try:
        # RiVaPy calculation
        start_day_dt_eom = _date_to_datetime(effective_start_date_eom) if effective_start_date_eom else None
        rolled_date_riva_dt = roll_day(
            day=date_to_roll,
            calendar=riva_calendar_obj,
            business_day_convention=riva_convention_enum,
            start_day=start_day_dt_eom # For MODIFIED_FOLLOWING_EOM, this is now non-None
        )
        rolled_date_riva_date = rolled_date_riva_dt.date() # Convert datetime to date for comparison

        if ql_convention_enum is None:
            # print(f"  Test Case ID: {test_case['id']}")
            # print(f"    Convention: {conv_details['name']} (RiVaPy specific convention)")
            # print(f"    Date to roll: {date_to_roll}, Calendar: {calendar_str}" +
            #       (f", Start EOM: {effective_start_date_eom}" if effective_start_date_eom else "") + default_eom_note)
            # print(f"    RiVaPy Rolled: {rolled_date_riva_date}")
            # print(f"    QuantLib: N/A (No QL counterpart)")
            # print("-" * 10)
            return 'riva_only_ran'

        # QuantLib calculation
        ql_date_to_roll = ql.Date(date_to_roll.day, date_to_roll.month, date_to_roll.year)
        # QuantLib's adjust method handles the rolling.
        ql_adjusted_date_obj = ql_calendar_obj.adjust(ql_date_to_roll, ql_convention_enum)
        # Convert QL Date back to Python date
        rolled_date_ql = date(ql_adjusted_date_obj.year(), ql_adjusted_date_obj.month(), ql_adjusted_date_obj.dayOfMonth())

        if rolled_date_riva_date == rolled_date_ql:
            status_to_return = 'match'
        else:
            status_to_return = 'deviation'
            print(f"  Test Case ID: {test_case['id']}")
            print(f"    Convention: {conv_details['name']}")
            print(f"    Date to roll: {date_to_roll}, Calendar: {calendar_str}" +
                  (f", Start EOM: {effective_start_date_eom}" if effective_start_date_eom else "") + default_eom_note)
            print(f"    RiVaPy Rolled:   {rolled_date_riva_date}")
            print(f"    QuantLib Rolled: {rolled_date_ql}")
            print(f"    Difference: RiVaPy ({rolled_date_riva_date}) != QuantLib ({rolled_date_ql})")
            print("-" * 10)

    except Exception as e:
        # status_to_return is already 'error'
        print(f"  Test Case ID: {test_case['id']}")
        print(f"    Convention: {conv_details['name']}")
        print(f"    Error during comparison for test case '{test_case['id']}': {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        print("-" * 10)
        
    return status_to_return


def compare_business_days():
    """
    Compares business day rolling conventions from rivapy.tools.datetools
    against QuantLib's calendar.adjust() method.
    """
    print("Starting Business Day Convention Comparison with QuantLib\n" + "="*50)

    conventions_to_test = [
        {"name": "Unadjusted", "riva_enum": RollConvention.UNADJUSTED, "ql_enum": ql.Unadjusted},
        {"name": "Following", "riva_enum": RollConvention.FOLLOWING, "ql_enum": ql.Following},
        {"name": "Modified Following", "riva_enum": RollConvention.MODIFIED_FOLLOWING, "ql_enum": ql.ModifiedFollowing},
        {"name": "Preceding", "riva_enum": RollConvention.PRECEDING, "ql_enum": ql.Preceding},
        {"name": "Modified Preceding", "riva_enum": RollConvention.MODIFIED_PRECEDING, "ql_enum": ql.ModifiedPreceding},
        {"name": "Nearest", "riva_enum": RollConvention.NEAREST, "ql_enum": ql.Nearest},
        # RiVaPy specific conventions (no direct QL enum for comparison via simple 'adjust')
        # These will be run as 'riva_only_ran'
        {"name": "Modified Following EOM (RiVaPy)", "riva_enum": RollConvention.MODIFIED_FOLLOWING_EOM, "ql_enum": None},
        {"name": "Modified Following BiMonthly (RiVaPy)", "riva_enum": RollConvention.MODIFIED_FOLLOWING_BIMONTHLY, "ql_enum": None},
    ]

    total_tests = 0
    successful_matches = 0
    reported_deviations = 0
    error_cases = 0 # Covers general errors, and specific calendar init errors
    riva_only_runs = 0
    skipped_invalid_param_cases = 0 # For cases skipped due to invalid RiVaPy params

    for conv_detail_item in conventions_to_test:
        print(f"\n--- Comparing: {conv_detail_item['name']} ---")
        
        convention_total_tests = 0
        convention_matches = 0
        convention_deviations = 0
        convention_errors = 0
        convention_riva_only = 0
        convention_skipped_invalid_params = 0

        for test_case_details in test_business_days:
            # If a convention is RiVaPy-specific, but the test case ID doesn't indicate it's for RiVaPy,
            # we might want to skip it to avoid cluttering output, or run it as RiVaPy only.
            # For now, we run all test cases for all defined conventions.
            # If ql_enum is None, it will naturally fall into 'riva_only_ran'.

            total_tests += 1 # Counts every combination tried
            convention_total_tests +=1
            
            status = run_business_day_comparison(
                conv_details=conv_detail_item,
                test_case=test_case_details
            )

            if status == 'match':
                successful_matches += 1
                convention_matches += 1
            elif status == 'deviation':
                reported_deviations += 1
                convention_deviations += 1
            elif status == 'error' or status == 'error_ql_calendar' or status == 'error_riva_calendar':
                error_cases += 1
                convention_errors += 1
            elif status == 'riva_only_ran':
                riva_only_runs += 1
                convention_riva_only +=1
            elif status == 'skipped_invalid_params':
                skipped_invalid_param_cases += 1
                convention_skipped_invalid_params += 1
        
        if convention_total_tests == 0:
             print(f"--- No test cases executed for {conv_detail_item['name']} ---")
        else:
            summary_line = f"--- Summary for {conv_detail_item['name']}: "
            summary_parts = [f"Total Tests: {convention_total_tests}"]

            if conv_detail_item.get("ql_enum") is not None:
                # Convention is comparable with QL
                summary_parts.append(f"Matches: {convention_matches}")
                summary_parts.append(f"Deviations: {convention_deviations}")
                # For QL comparable, riva_only_runs should be 0 unless QL calendar failed (which is an error)
            else:
                # RiVaPy-specific convention (no QL enum)
                summary_parts.append(f"RiVaPy Only Ran: {convention_riva_only}")
            
            summary_parts.append(f"Errors: {convention_errors}")
            if convention_skipped_invalid_params > 0:
                summary_parts.append(f"Skipped (Invalid Params): {convention_skipped_invalid_params}")
            
            print(summary_line + ", ".join(summary_parts) + " ---")

        print("=" * 30)
    print("\n" + "="*50)
    print("Overall Business Day Convention Comparison Summary:")
    print("="*50)
    print(f"Total Test Scenarios Attempted (Convention x Test Case): {total_tests}")
    print(f"Successful Matches with QuantLib: {successful_matches}")
    print(f"Reported Deviations from QuantLib: {reported_deviations}")
    print(f"Errors during comparison (incl. calendar init): {error_cases}")
    if skipped_invalid_param_cases > 0:
        print(f"Skipped (RiVaPy param pre-requisite not met): {skipped_invalid_param_cases}")
    print(f"RiVaPy-specific conventions/scenarios processed (no QL counterpart): {riva_only_runs}")
    print("="*50 + "\nComparison Finished.")

if __name__ == "__main__":
    compare_business_days()
