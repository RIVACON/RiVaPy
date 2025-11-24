# 2025.09.09 Bootstrapping without pyvacon
import unittest
import sys

from tests.setup_logging import setup_logging_for_tests

# Configure logging once per test module
setup_logging_for_tests("tests/rivapy_test.log")
import logging

logger = logging.getLogger("rivapy.tests.test_rivapyweb_batch_run")

import math
import pandas as pd
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

from rivapy.marketdata.bootstrapping import (
    bootstrap_curve,
    error_fn,
    find_bracket,
    get_quote,
)
from rivapy.marketdata.curves import DiscountCurve
from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.instruments.fra_specifications import ForwardRateAgreementSpecification
from rivapy.instruments.ir_swap_specification import (
    InterestRateSwapSpecification,
    IrFixedLegSpecification,
    IrFloatLegSpecification,
    IrOISLegSpecification,
    InterestRateBasisSwapSpecification,
)
from rivapy.tools.enums import DayCounterType, InterpolationType, ExtrapolationType, Instrument
from rivapy.instruments.components import ConstNotionalStructure
from rivapy.tools.datetools import DayCounter, Period, Schedule, calc_end_day, calc_start_day


# for specification from file tests
import rivapy.instruments.specification_from_csv as sfc
from rivapy.tools.holidays_compat import HolidayBase as _HolidayBase, EuropeanCentralBank as _ECB


# Helper functions
def tolerance_from_quote(q: float) -> float:
    """
    Determine an appropriate delta for assertAlmostEqual
    based on the number of decimals in the quote.
    """
    s = format(q, "f").rstrip("0").rstrip(".")
    decimals = len(s.split(".")[1]) if "." in s else 0
    delta = 0.5 * 10 ** (-decimals)
    return delta





def deep_equal(obj1, obj2, path="root"):
    """Recursively compare two objects and print where they differ."""
    if type(obj1) != type(obj2):
        print(f"Type mismatch at {path}: {type(obj1)} != {type(obj2)}")
        return False

    # handle objects with __dict__ (custom classes)
    if hasattr(obj1, "__dict__") and hasattr(obj2, "__dict__"):
        all_equal = True
        keys1, keys2 = set(obj1.__dict__.keys()), set(obj2.__dict__.keys())

        for key in keys1 | keys2:
            if key not in obj1.__dict__:
                print(f"Missing key {path}.{key} in obj1")
                all_equal = False
                continue
            if key not in obj2.__dict__:
                print(f"Missing key {path}.{key} in obj2")
                all_equal = False
                continue

            if not deep_equal(obj1.__dict__[key], obj2.__dict__[key], f"{path}.{key}"):
                all_equal = False
        return all_equal

    # handle lists and tuples
    if isinstance(obj1, (list, tuple)):
        all_equal = True
        for i, (x, y) in enumerate(zip(obj1, obj2)):
            if not deep_equal(x, y, f"{path}[{i}]"):
                all_equal = False
        if len(obj1) != len(obj2):
            print(f"Length mismatch at {path}: {len(obj1)} != {len(obj2)}")
            all_equal = False
        return all_equal

    # base case: primitive comparison
    if obj1 != obj2:
        print(f"Value mismatch at {path}: {obj1} != {obj2}")
        return False

    return True


def update_fra_tenors(df):
    """
    Overwrite UnderlyingTenor for FRA instruments based on Maturity like '3MX6M'.
    """

    def fra_tenor_delta(maturity_str):
        if pd.isna(maturity_str):
            return maturity_str
        maturity_str = maturity_str.upper().strip()
        if "X" in maturity_str:
            try:
                start, end = maturity_str.split("X")

                # Convert start and end to months
                def to_months(s):
                    if s.endswith("D"):
                        return float(s[:-1]) / 30
                    elif s.endswith("W"):
                        return float(s[:-1]) * 7 / 30
                    elif s.endswith("M"):
                        return float(s[:-1])
                    elif s.endswith("Y"):
                        return float(s[:-1]) * 12
                    else:
                        return 0.0

                start_m = to_months(start)
                end_m = to_months(end)
                delta_m = end_m - start_m

                # Return as string with 'M'
                return f"{int(delta_m)}M" if delta_m.is_integer() else f"{delta_m:.2f}M"
            except Exception:
                return maturity_str
        else:
            return maturity_str

    # Only apply to FRA instruments
    mask = df["Instrument"] == "FRA"  # adjust column name if necessary
    df.loc[mask, "UnderlyingTenor"] = df.loc[mask, "Maturity"].apply(fra_tenor_delta)

    return df


def auto_detect_and_build_all_curves(
    ref_date,
    df: pd.DataFrame,
    holidays=None,
    interpolation_type=InterpolationType.LINEAR_LOG,
    extrapolation_type=ExtrapolationType.LINEAR_LOG,
    daycount=DayCounterType.Act365Fixed,
    existing_curves: dict = None,
):
    """
    Auto-detect and build all curves from df filtered to a single DATE + CURRENCY.

    Implementation (simple version, NO TBS logic yet):
      1. Build OIS discount curve
      2. Identify all (index, tenor) pairs from non-TBS instruments
      3. Build each forward curve independently (DEPOSIT/FRA/IRS only)
      4. Log TBS instruments but skip them

    Args:
        df: filtered DataFrame
        ref_date: valuation date
        holidays: holiday calendar
        interpolation_type, extrapolation_type, daycount: bootstrap params
        existing_curves: optional dict of pre-computed curves

    Returns:
        curves: dict
        logs: list of text messages
    """
    #print("DEBUG type(df):", type(df))
    #print("DEBUG head(df):", getattr(df, "head", lambda: "not df")())
    holidays = holidays or _ECB()
    curves = existing_curves.copy() if existing_curves else {}
    logs = []

    # --- Build discount curve ------------------------------------------
    df_ois = df[df["Instrument"].str.upper() == "OIS"]

    if df_ois.empty and "discount_curve" not in curves:
        raise Exception("Cannot build curves: no OIS instruments and no existing discount curve provided.")#use depost as faislsafe?

    if "discount_curve" not in curves:
        instruments_dc = sfc.load_specifications_from_pd(df_ois, ref_date, holidays)
        quotes_dc = df_ois["Quote"].tolist()

        discount_curve = bootstrap_curve(
            ref_date,
            "discount_curve",
            daycount,
            instruments_dc,
            quotes_dc,
            interpolation_type=interpolation_type,
            extrapolation_type=extrapolation_type,
        )
        curves["discount_curve"] = discount_curve
        logs.append(f"Built discount curve using {len(df_ois)} OIS instruments.")
    else:
        discount_curve = curves["discount_curve"]
        logs.append("Using existing provided discount curve.")

    # --- Identify forward curves (NO TBS for now) -----------------------
    df_non_tbs = df[df["Instrument"].str.upper().isin([ "FRA", "IRS"])] # deposit is excluded

    

    forward_keys = set()
    
    print("------------------------")
    print(df_non_tbs['UnderlyingTenor'].unique())
    print(df_non_tbs['UnderlyingTenor'].value_counts())
    for _, r in df_non_tbs.iterrows():
        idx = r.get("UnderlyingIndex", "").strip().upper()
        ten = r.get("UnderlyingTenor", "").strip().upper()
        #print(idx,ten)
        if idx and ten:
            forward_keys.add((idx, ten))

    
    print('------------------------DEBUG: autodetect and build forward curves')
    print(forward_keys)


    logs.append(f"Detected {len(forward_keys)} forward curve groups.")

    # --- Build each forward curve ---------------------------------------
    for idx, ten in forward_keys:
        curve_name = f"{idx}_{ten}"
        if curve_name in curves:
            logs.append(f"Skipped {curve_name}: already exists.")
            continue

        try:
            fwd_curve, msg = build_forward_curve_auto(
                idx,
                ten,
                df_non_tbs,
                ref_date,
                discount_curve,
                holidays=holidays,
                interpolation_type=interpolation_type,
                extrapolation_type=extrapolation_type,
                daycount=daycount,
            )
            curves[curve_name] = fwd_curve
            logs.append(msg)
        except Exception as e:
            logs.append(f"Failed building {curve_name}: {e}")

    # --- TBS is ignored for now ----------------------------------------
    df_tbs = df[df["Instrument"].str.upper() == "TBS"]
    if not df_tbs.empty:
        logs.append(f"NOTE: Found {len(df_tbs)} TBS instruments but TBS processing is not yet implemented.")

    return curves, logs


def build_forward_curve_auto(
    target_index: str,
    target_tenor: str,
    df_instruments: pd.DataFrame,
    ref_date,
    discount_curve,
    holidays=None,
    interpolation_type=InterpolationType.LINEAR_LOG,
    extrapolation_type=ExtrapolationType.LINEAR_LOG,
    daycount=DayCounterType.Act365Fixed,
):
    """
    Build a forward curve for (target_index, target_tenor) using only
    , FRA, IRS instruments found in df_instruments.

    This function does NOT use TBS logic (ignored for now).

    Args:
        target_index (str): e.g. "EURIBOR"
        target_tenor (str): e.g. "3M"
        df_instruments (pd.DataFrame): all instruments filtered by date+currency
        ref_date (datetime): valuation date
        discount_curve (DiscountCurve): the OIS curve
        holidays: holiday calendar for conventions
        interpolation_type: interpolation enum
        extrapolation_type: extrapolation enum
        daycount: DayCounterType enum

    Returns:
        (forward_curve, log_message)
    """

    holidays = holidays or _ECB()

    # --- Select usable base instruments -------------------------
    mask = (
        (df_instruments["UnderlyingIndex"].str.upper() == target_index.upper()) &
        (df_instruments["UnderlyingTenor"].str.upper() == target_tenor.upper()) &
        (df_instruments["Instrument"].str.upper().isin([ "FRA", "IRS"]))
    )

    df_curve = df_instruments[mask].copy()

    if df_curve.empty:
        raise Exception(f"No FRA/IRS instruments found for {target_index} {target_tenor}")

    instruments = sfc.load_specifications_from_pd(df_curve, ref_date, holidays)
    quotes = df_curve["Quote"].tolist()

    print(f"----------------------Checking for duplicate dates for {target_index}_{target_tenor}")
    instruments_by_date = {}
    for i, inst in enumerate(instruments):
        end_date = inst.get_end_date()
        if end_date in instruments_by_date:
            raise Exception(f"Duplicate expiry date found: {end_date}")
        instruments_by_date[end_date] = (quotes[i], inst)

    # --- Bootstrap the curve ------------------------------------
    curve_id = f"{target_index}_{target_tenor}"

    fwd_curve = bootstrap_curve(
        ref_date,
        curve_id,
        daycount,
        instruments,
        quotes,
        curves={"discount_curve": discount_curve},
        interpolation_type=interpolation_type,
        extrapolation_type=extrapolation_type,
    )

    log = f"Built forward curve {curve_id} using {len(df_curve)} instruments."

    return fwd_curve, log




class TestBatchBootstrapCurveInstruments(unittest.TestCase):
    """Test for various combination of instruments supplied to the bootstrapper

    Args:
        unittest (_type_): _description_
    """

    def setUp(self):
        self.ref_date = datetime(2019, 8, 31)
        #self.curve_id = "EUR_DISC"
        self.day_count = DayCounterType.Act365Fixed
        self.interp = InterpolationType.LINEAR_LOG
        self.extrap = ExtrapolationType.LINEAR_LOG
        self.holidays = _ECB()
        """Setup input file"""
        # set directory and file name for Input Quotes
        dirName = "./sample_data"  # "./"
        #fileName = "/inputQuotes_includeFRAs.csv"  # "/inputQuotes.csv"
        fileName = "/multi_dates_tbs.csv"  # "/inputQuotes.csv"

        df = pd.read_csv(dirName + fileName, sep=";", decimal=",")
        column_names = list(df.columns)

        self.df = df
        self.column_names = column_names

    

    def test_auto_detect_and_build_all_curves(self):
        """testing on specific date and currency
        """
        mon = "07"
        day = "23"
        year = "2025"
        date_str = f"{day}.{mon}.{year}"
        refDate = datetime(int(year), int(mon), int(day))
        #date = datetime(2025, 7, 23)
        currency = "EUR"

        df_all = self.df.copy()


        df= df_all[ (df_all["Date"] == date_str)
                & (df_all["Currency"] == currency)]

        df = update_fra_tenors(df)

        print(df)
        print(df["UnderlyingTenor"].unique())
        print(df["UnderlyingTenor"].value_counts())

        ###########applying the function
        # boot_curve, batch_bootstrap_logs = auto_detect_and_build_all_curves(
        #         ref_date = refDate,
        #         df = df_ins,                      # all selected instruments
        #         interpolation_type=self.interp,
        #         extrapolation_type=self.extrap,
        #         holidays=self.holidays,
        #     )

        # print("----------------batch run bootsrap logs--------")
        # for log in batch_bootstrap_logs:
        #     print(log)

        curves = {}
        logs = []

        # --- Build discount curve ------------------------------------------
        df_ois = df[df["Instrument"].str.upper() == "OIS"]

        if df_ois.empty and "discount_curve" not in curves:
            raise Exception("Cannot build curves: no OIS instruments and no existing discount curve provided.")#use depost as faislsafe?

        if "discount_curve" not in curves:
            instruments_dc = sfc.load_specifications_from_pd(df_ois, refDate, self.holidays)
            quotes_dc = df_ois["Quote"].tolist()

            discount_curve = bootstrap_curve(
                refDate,
                "discount_curve",
                self.day_count,
                instruments_dc,
                quotes_dc,
                interpolation_type=self.interp,
                extrapolation_type=self.extrap,
            )
            curves["discount_curve"] = discount_curve
            logs.append(f"Built discount curve using {len(df_ois)} OIS instruments.")
        else:
            discount_curve = curves["discount_curve"]
            logs.append("Using existing provided discount curve.")

        # --- Identify forward curves (NO TBS for now) -----------------------
        df_non_tbs = df[df["Instrument"].str.upper().isin([ "FRA", "IRS"])] # deposit is excluded

        

        forward_keys = set()
        
        print("------------------------")
        print(df_non_tbs['UnderlyingTenor'].unique())
        print(df_non_tbs['UnderlyingTenor'].value_counts())
        for _, r in df_non_tbs.iterrows():
            idx = r.get("UnderlyingIndex", "").strip().upper()
            ten = r.get("UnderlyingTenor", "").strip().upper()
            #print(idx,ten)
            if idx and ten:
                forward_keys.add((idx, ten))

        
        print('------------------------DEBUG: autodetect and build forward curves')
        print(forward_keys)


        logs.append(f"Detected {len(forward_keys)} forward curve groups.")

        # --- Build each forward curve ---------------------------------------
        for idx, ten in forward_keys:
            curve_name = f"{idx}_{ten}"
            if curve_name in curves:
                logs.append(f"Skipped {curve_name}: already exists.")
                continue

            try:
                fwd_curve, msg = build_forward_curve_auto(
                    idx,
                    ten,
                    df_non_tbs,
                    refDate,
                    discount_curve,
                    holidays=self.holidays,
                    interpolation_type=self.interp,
                    extrapolation_type=self.extrap,
                    daycount=self.day_count,
                )
                curves[curve_name] = fwd_curve
                logs.append(msg)
            except Exception as e:
                logs.append(f"Failed building {curve_name}: {e}")

        # --- TBS is ignored for now ----------------------------------------
        df_tbs = df[df["Instrument"].str.upper() == "TBS"]
        if not df_tbs.empty:
            logs.append(f"NOTE: Found {len(df_tbs)} TBS instruments but TBS processing is not yet implemented.")




















if __name__ == "__main__":
    # # Open a file for capturing output
    # with open("test_output.txt", "w") as f:
    #     # Save original stdout
    #     original_stdout = sys.stdout
    #     sys.stdout = f
    #     # Run your tests
    #     unittest.main(argv=["first-arg-is-ignored"], exit=False)

    #     # Restore stdout
    #     sys.stdout = original_stdout
    unittest.main()
