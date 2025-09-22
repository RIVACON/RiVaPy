import pandas as pd
from datetime import datetime
from rivapy.instruments import (
    DepositSpecification,
    ForwardRateAgreementSpecification,
    InterestRateSwapSpecification,
    IrFixedLegSpecification,
    IrFloatLegSpecification,
    IrOISLegSpecification,
)


def load_specifications_from_pd(df: pd.DataFrame):
    """Takes in a pandas data frame which already has the required columns.

    Args:
        df (pd.DataFrame): Contains the column information for the market quotes of a given instrument

    Returns:
        List[Specification]: List of Specification items used in Rivapy, e.g., in yield curve bootstrapping.
    """
    # df = pd.read_csv(file_path, parse_dates=True)
    specs = []
    for _, row in df.iterrows():
        spec = make_specification_from_row(row)
        specs.append(spec)
    return specs


def make_specification_from_row(row):
    inst_type = row["Instrument"].upper()

    if inst_type == "DEPOSIT":
        return make_deposit_spec(row)
    elif inst_type == "OIS":
        return make_ois_spec(row)
    elif inst_type == "FRA":
        return make_fra_spec(row)
    elif inst_type == "SWAP":
        return make_swap_spec(row)
    else:
        raise ValueError(f"Unsupported instrument type {inst_type}")


def make_deposit_spec(row):
    return DepositSpecification(
        obj_id=f"DEP_{row['Currency']}_{row['Maturity']}",
        fixing_date=date.today(),  # normally trade_date or ref_date + spot_lag
        start_date=None,  # let term drive end date
        end_date=None,
        maturity_date=None,
        currency=row["Currency"],
        notional=100.0,
        rate=float(row["Quote"]),
        term=row["Maturity"],  # e.g. "1D", "7D"
        day_count_convention=DayCounterType.from_string(row["DayCountFixed"]),
        business_day_convention=RollConvention.from_string(row["RollConventionFixed"]),
        roll_convention=RollConvention.from_string(row["RollConventionFixed"]),
        spot_lag=int(row["SpotLag"].replace("D", "")),
    )


def make_ois_spec(row):
    # Simplify: one leg floating OIS, one leg fixed
    ois_leg = IrOISLegSpecification(
        obj_id=f"OIS_FLOAT_{row['Currency']}_{row['Maturity']}",
        notional=100.0,
        rate_reset_dates=[],  # these would come from scheduler
        start_dates=[],
        end_dates=[],
        rate_start_dates=[],
        rate_end_dates=[],
        pay_dates=[],
        currency=row["Currency"],
        udl_id=row["UnderlyingIndex"],
        fixing_id=f"{row['UnderlyingIndex']}_FIXING",
        day_count_convention=DayCounterType.from_string(row["DayCountFloat"]),
        rate_day_count_convention=DayCounterType.from_string(row["DayCountFloat"]),
        spread=0.0,
    )

    fixed_leg = IrFixedLegSpecification(
        fixed_rate=float(row["Quote"]),
        obj_id=f"OIS_FIXED_{row['Currency']}_{row['Maturity']}",
        notional=100.0,
        start_dates=[],
        end_dates=[],
        pay_dates=[],
        currency=row["Currency"],
        day_count_convention=DayCounterType.from_string(row["DayCountFixed"]),
    )

    return InterestRateSwapSpecification(
        obj_id=f"OIS_{row['Currency']}_{row['Maturity']}",
        notional=100.0,
        issue_date=date.today(),
        maturity_date=None,  # derive from scheduler
        pay_leg=fixed_leg,
        receive_leg=ois_leg,
        currency=row["Currency"],
    )


def make_fra_spec(row):
    return ForwardRateAgreementSpecification(
        obj_id=f"FRA_{row['Currency']}_{row['Maturity']}",
        trade_date=date.today(),
        notional=100.0,
        rate=float(row["Quote"]),
        start_date=None,  # derive from tenor
        end_date=None,
        udlID=row["UnderlyingIndex"],
        rate_start_date=None,
        rate_end_date=None,
        currency=row["Currency"],
        day_count_convention=DayCounterType.from_string(row["DayCountFixed"]),
        rate_day_count_convention=DayCounterType.from_string(row["DayCountFloat"]),
    )


def make_swap_spec(row):
    fixed_leg = IrFixedLegSpecification(
        fixed_rate=float(row["Quote"]),
        obj_id=f"SWAP_FIXED_{row['Currency']}_{row['Maturity']}",
        notional=100.0,
        start_dates=[],
        end_dates=[],
        pay_dates=[],
        currency=row["Currency"],
        day_count_convention=DayCounterType.from_string(row["DayCountFixed"]),
    )
    float_leg = IrFloatLegSpecification(
        obj_id=f"SWAP_FLOAT_{row['Currency']}_{row['Maturity']}",
        notional=100.0,
        reset_dates=[],
        start_dates=[],
        end_dates=[],
        rate_start_dates=[],
        rate_end_dates=[],
        pay_dates=[],
        currency=row["Currency"],
        udl_id=row["UnderlyingIndex"],
        fixing_id=f"{row['UnderlyingIndex']}_FIXING",
        spread=0.0,
    )
    return InterestRateSwapSpecification(
        obj_id=f"SWAP_{row['Currency']}_{row['Maturity']}",
        notional=100.0,
        issue_date=date.today(),
        maturity_date=None,
        pay_leg=fixed_leg,
        receive_leg=float_leg,
        currency=row["Currency"],
    )
