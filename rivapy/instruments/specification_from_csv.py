import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Union as _Union
from holidays import HolidayBase as _HolidayBase
from holidays import EuropeanCentralBank as _ECB
from rivapy.instruments._logger import logger
from rivapy.instruments import (
    DepositSpecification,
    ForwardRateAgreementSpecification,
    InterestRateSwapSpecification,
    IrFixedLegSpecification,
    IrFloatLegSpecification,
    IrOISLegSpecification,
)
from rivapy.instruments.components import ConstNotionalStructure
from rivapy.tools.datetools import (
    DayCounter,
    Period,
    Schedule,
    _date_to_datetime,
    _datetime_to_date_list,
    _term_to_period,
    roll_day,
    calc_start_day,
    calc_end_day,
)


def load_specifications_from_pd(df: pd.DataFrame, ref_date: datetime, calendar: _Union[_HolidayBase, str] = _ECB()):
    """Takes in a pandas data frame which already has the required columns.

    Args:
        df (pd.DataFrame): Contains the column information for the market quotes of a given instrument

    Returns:
        List[Specification]: List of Specification items used in Rivapy, e.g., in yield curve bootstrapping.
    """
    # df = pd.read_csv(file_path, parse_dates=True)
    specs = []
    for _, row in df.iterrows():
        # if row["Maturity"] == "2M": # DEBUG TEST 2025
        #     logger.debug("debugging for spepcific instrument case")
        spec = make_specification_from_row(row, ref_date, calendar)
        specs.append(spec)
    return specs


def make_specification_from_row(row: pd.DataFrame, ref_date: datetime, calendar: _Union[_HolidayBase, str] = _ECB()):
    inst_type = row["Instrument"].upper()

    if inst_type == "DEPOSIT":
        return make_deposit_spec(row, ref_date, calendar)
    elif inst_type == "OIS":
        return make_ois_spec(row, ref_date, calendar)
    elif inst_type == "FRA":
        return make_fra_spec(row, ref_date, calendar)
    elif inst_type == "IRS":
        return make_irswap_spec(row, ref_date, calendar)
    else:
        raise ValueError(f"Unsupported instrument type {inst_type}")


def make_deposit_spec(row: pd.DataFrame, ref_date: datetime, calendar: _Union[_HolidayBase, str] = _ECB()):
    label = row["Instrument"] + "_" + row["Maturity"]

    dep_spec = DepositSpecification(
        obj_id=label,
        fixing_date=ref_date,
        # end_date: _Optional[_Union[date, datetime]] = None,
        # start_date: _Optional[_Union[date, datetime]] = None,
        # maturity_date: _Optional[_Union[date, datetime]] = None,
        currency=row["Currency"],
        # notional: float = 100.0, # we let notional default to 100
        rate=float(row["Quote"]),
        term=row["Maturity"],
        day_count_convention=row["DayCountFloat"],
        business_day_convention=row["RollConventionFloat"],
        # roll_convention: _Union[RollRule, str] = RollRule.EOM, # leave as default
        spot_days=int(row["SpotLag"][:-1]),  # make assumption it is always given in DAYS convert -> int
        calendar=calendar,
        issuer="dummy_issuer",
        securitization_level="NONE",
        # payment_days: int = 0,
        # adjust_start_date: bool = True,
        # adjust_end_date: bool = False,
    )

    return dep_spec


def make_fra_spec(row: pd.DataFrame, ref_date: datetime, calendar: _Union[_HolidayBase, str] = _ECB()):
    label = row["Instrument"] + "_" + row["Maturity"]
    # maturity must be in the form of YYMxZZM where YY and ZZ are integers
    s = row["Maturity"].upper()
    start_str, end_str = s.split("X")
    start_period = int(start_str[:-1])  # remove last "M"
    end_period = int(end_str[:-1])

    spot_date = roll_day(
        day=ref_date + timedelta(days=int(row["SpotLag"][:-1])),
        calendar=calendar,
        business_day_convention=row["RollConventionFloat"],
        start_day=None,
    )

    start_date = roll_day(
        day=spot_date + relativedelta(months=start_period),
        calendar=calendar,
        business_day_convention=row["RollConventionFloat"],
        start_day=None,
    )  # spot_date + start_period #need roll convention: ddc, bdc, holiday, date
    end_date = roll_day(
        day=start_date + relativedelta(months=end_period - start_period),
        calendar=calendar,
        business_day_convention=row["RollConventionFloat"],
        start_day=None,
    )  # start_date + end_period #need roll convention: ddc, bdc, holiday, date

    fra_spec = ForwardRateAgreementSpecification(
        obj_id=label,
        trade_date=ref_date,
        notional=100,
        rate=float(row["Quote"]),
        start_date=start_date,
        end_date=end_date,
        udlID=row["UnderlyingIndex"],
        rate_start_date=start_date,
        rate_end_date=end_date,
        # maturity_date=,
        day_count_convention=row["DayCountFixed"],
        business_day_convention=row["RollConventionFixed"],
        rate_day_count_convention=row["DayCountFloat"],
        rate_business_day_convention=row["RollConventionFloat"],
        calendar=calendar,
        currency=row["Currency"],
        # payment_days: int = 0,
        spot_days=int(row["SpotLag"][:-1]),
        # start_period: int = None,
        # end_period: int = None,
        # index_alias: str = None,
        # issuer: str = None,
    )

    return fra_spec


def make_irswap_spec(row: pd.DataFrame, ref_date: datetime, calendar: _Union[_HolidayBase, str] = _ECB()):

    # the following information is expected:
    instr = row["Instrument"]
    fixDayCount = row["DayCountFixed"]
    floatDayCount = row["DayCountFloat"]
    basisDayCount = row["DayCountBasis"]
    maturity = row["Maturity"]
    underlyingIndex = row["UnderlyingIndex"]
    tenor = row["UnderlyingTenor"]
    underlyingPayFreq = row["UnderlyingPaymentFrequency"]
    basisTenor = row["BasisTenor"]
    basisPayFreq = row["BasisPaymentFrequency"]
    fixPayFreq = row["PaymentFrequencyFixed"]
    rollConvFloat = row["RollConventionFloat"]
    rollConvFix = row["RollConventionFixed"]
    rollConvBasis = row["RollConventionBasis"]
    spotLag = row["SpotLag"]  # expect form "1D", i.e 1 day
    parRate = float(row["Quote"])
    currency = row["Currency"]
    label = instr + "_" + maturity

    # we use the helper function with spotlag in place of maturity to effctively shift the date
    spot_date = calc_end_day(start_day=ref_date, term=spotLag, business_day_convention=rollConvFix, calendar=calendar)
    expiry = calc_end_day(spot_date, maturity, rollConvFix, calendar)  # get expiry of swap (cannot be before last paydate of legs)

    # FIXED LEG
    fix_schedule = Schedule(
        start_day=spot_date, end_day=expiry, time_period=fixPayFreq, business_day_convention=rollConvFix, calendar=calendar, ref_date=ref_date
    ).generate_dates(False)

    fix_start_dates = fix_schedule[:-1]
    fix_end_dates = fix_schedule[1:]
    fix_pay_dates = fix_end_dates

    # # definition of the fixed leg
    fixed_leg = IrFixedLegSpecification(
        fixed_rate=parRate,
        obj_id=label + "_fixed_leg3",
        notional=100.0,
        start_dates=fix_start_dates,
        end_dates=fix_end_dates,
        pay_dates=fix_pay_dates,
        currency=currency,
        day_count_convention=rollConvFix,
    )

    # FLOAT LEG
    flt_schedule = Schedule(
        start_day=spot_date,
        end_day=expiry,
        time_period=underlyingPayFreq,
        business_day_convention=rollConvFloat,
        calendar=calendar,
        ref_date=ref_date,
    ).generate_dates(False)

    flt_start_dates = flt_schedule[:-1]
    flt_end_dates = flt_schedule[1:]
    flt_pay_dates = flt_end_dates

    flt_reset_schedule = Schedule(
        start_day=spot_date, end_day=expiry, time_period=tenor, business_day_convention=rollConvFloat, calendar=calendar, ref_date=ref_date
    ).generate_dates(False)

    flt_reset_dates = flt_reset_schedule[:-1]

    ns = ConstNotionalStructure(100.0)
    spread = 0.00

    # # definition of the floating leg
    float_leg = IrFloatLegSpecification(
        obj_id=label + "_float_leg",
        notional=ns,
        reset_dates=flt_reset_dates,
        start_dates=flt_start_dates,
        end_dates=flt_end_dates,
        rate_start_dates=flt_start_dates,
        rate_end_dates=flt_end_dates,
        pay_dates=flt_pay_dates,
        currency=currency,
        udl_id=underlyingIndex,
        fixing_id="test_fixing_id",
        day_count_convention=rollConvFloat,
        spread=spread,
    )

    # # definition of the IR swap - assume fixed leg is the pay leg
    ir_swap = InterestRateSwapSpecification(
        obj_id=label,
        notional=ns,
        issue_date=ref_date,
        maturity_date=expiry,
        pay_leg=fixed_leg,
        receive_leg=float_leg,
        currency=currency,
        day_count_convention=rollConvFloat,
        issuer="dummy_issuer",
        securitization_level="COLLATERALIZED",
    )

    return ir_swap


def make_ois_spec(row: pd.DataFrame, ref_date: datetime, calendar: _Union[_HolidayBase, str] = _ECB()):
    label = row["Instrument"] + "_" + row["Maturity"]

    # the following information is expected:
    instr = row["Instrument"]
    fixDayCount = row["DayCountFixed"]
    floatDayCount = row["DayCountFloat"]
    basisDayCount = row["DayCountBasis"]
    maturity = row["Maturity"]
    underlyingIndex = row["UnderlyingIndex"]
    tenor = row["UnderlyingTenor"]
    underlyingPayFreq = row["UnderlyingPaymentFrequency"]
    basisTenor = row["BasisTenor"]
    basisPayFreq = row["BasisPaymentFrequency"]
    fixPayFreq = row["PaymentFrequencyFixed"]
    rollConvFloat = row["RollConventionFloat"]
    rollConvFix = row["RollConventionFixed"]
    rollConvBasis = row["RollConventionBasis"]
    spotLag = row["SpotLag"]  # expect form "1D", i.e 1 day
    parRate = float(row["Quote"])
    currency = row["Currency"]
    label = instr + "_" + maturity

    # we use the helper function with spotlag in place of maturity to effctively shift the date
    spot_date = calc_end_day(start_day=ref_date, term=spotLag, business_day_convention=rollConvFix, calendar=calendar)
    expiry = calc_end_day(spot_date, maturity, rollConvFix, calendar)  # get expiry of swap (cannot be before last paydate of legs)
    expiry_unadjusted = calc_end_day(start_day=spot_date, term=maturity, calendar=calendar)

    # FIXED LEG
    fix_schedule = Schedule(
        start_day=spot_date,
        end_day=expiry_unadjusted,
        time_period=fixPayFreq,
        business_day_convention=rollConvFix,
        calendar=calendar,
        ref_date=ref_date,
    ).generate_dates(False)

    if fix_schedule[-1] != expiry:
        logger.error(
            "Unexpected schedule generation for OIS fixed leg: last date in schedule {} does not match adjusted expiry {}".format(
                fix_schedule[-1], expiry
            )
        )

    fix_start_dates = fix_schedule[:-1]
    fix_end_dates = fix_schedule[1:]
    fix_pay_dates = fix_end_dates

    # # definition of the fixed leg
    fixed_leg = IrFixedLegSpecification(
        fixed_rate=parRate,
        obj_id=label + "_fixed_leg3",
        notional=100.0,
        start_dates=fix_start_dates,
        end_dates=fix_end_dates,
        pay_dates=fix_pay_dates,
        currency=currency,
        day_count_convention=rollConvFix,
    )

    # FLOAT LEG - OIS
    flt_schedule = Schedule(
        start_day=spot_date,
        end_day=expiry_unadjusted,
        time_period=underlyingPayFreq,
        business_day_convention=rollConvFloat,
        calendar=calendar,
        ref_date=ref_date,
    ).generate_dates(False)

    flt_start_dates = flt_schedule[:-1]
    flt_end_dates = flt_schedule[1:]
    flt_pay_dates = flt_end_dates

    flt_reset_schedule = Schedule(
        start_day=spot_date, end_day=expiry, time_period=tenor, business_day_convention=rollConvFloat, calendar=calendar, ref_date=ref_date
    ).generate_dates(False)

    flt_reset_dates = flt_reset_schedule[:-1]

    res = IrOISLegSpecification.ois_scheduler_2D(flt_start_dates, flt_end_dates)

    daily_rate_start_dates = res[0]  # 2D list: coupon i -> list of daily starts
    daily_rate_end_dates = res[1]  # 2D list: coupon i -> list of daily ends
    daily_rate_reset_dates = res[2]  # 2D list: coupon i -> list of reset dates
    daily_rate_pay_dates = res[3]

    ns = ConstNotionalStructure(100.0)
    spread = 0.00

    # # definition of the floating leg

    float_leg = IrOISLegSpecification(
        obj_id=label + "_float_leg",
        notional=ns,
        rate_reset_dates=daily_rate_reset_dates,
        start_dates=flt_start_dates,
        end_dates=flt_end_dates,
        rate_start_dates=daily_rate_start_dates,
        rate_end_dates=daily_rate_end_dates,
        pay_dates=daily_rate_pay_dates,
        currency=currency,
        udl_id=underlyingIndex,
        fixing_id="test_fixing_id",
        day_count_convention=floatDayCount,
        rate_day_count_convention=floatDayCount,
        spread=spread,
    )

    # # definition of the IR swap - assume fixed leg is the pay leg
    ois = InterestRateSwapSpecification(
        obj_id=label,
        notional=ns,
        issue_date=ref_date,
        maturity_date=expiry,
        pay_leg=fixed_leg,
        receive_leg=float_leg,
        currency=currency,
        day_count_convention=rollConvFloat,
        issuer="dummy_issuer",
        securitization_level="COLLATERALIZED",
    )

    return ois
