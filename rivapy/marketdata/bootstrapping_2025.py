# 2025.07.23 Hans Nguyen
# Boostrapping in rivapy indepedent of pyvacon

##########
# Modules
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from typing import Union as _Union, List as _List

# from rivapy.instruments.specifications import DepositSpecification,
from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.instruments.fra_specifications import ForwardRateAgreementSpecification
from rivapy.instruments.ir_swap_specification import (
    InterestRateSwapSpecification,
    IrFixedLegSpecification,
    IrFloatLegSpecification,
    IrSwapLegSpecification,
)
from rivapy.marketdata import DiscountCurve
from rivapy.tools.enums import DayCounterType, RollConvention, RollRule, InterpolationType, ExtrapolationType, Instrument
from rivapy.tools.datetools import DayCounter

from scipy.optimize import brentq


##########
# Classes


##########
# Functions
def bootstrap_curve(
    ref_date: _Union[date, datetime],
    curve_id: str,
    day_count_convention: _Union[DayCounterType, str],
    instruments: _List,
    quotes: _List,
    discount_curve: DiscountCurve = None,
    basis_curve: DiscountCurve = None,
    interpolation_type: InterpolationType = InterpolationType.LINEAR,
    extrapolation_type: ExtrapolationType = ExtrapolationType.NONE,
    tolerance: float = 0.0,
    max_iterations: int = 100000,
) -> DiscountCurve:
    """TODO: implement interpolation/extrapolation type as arguments

    Args:
        ref_date (_Union[date, datetime]): the reference for the new curve
        curve_id (str): Id for the new Curve
        day_count_convention (_Union[DayCounterType, str]): daycounter for the new curve
        instruments (_List): instrument specifications that are used in the calibration (deposits, FRAs, and swaps allowed atm)
        quotes (_List): the rate quotes for the instruments (deposit rates, FRAs and swap rates)
        discount_curve (DiscountCurve, optional): discount curve used for the instruments (if empty, the bootstrapped curve is used for discounting). Defaults to None.
        basis_curve (DiscountCurve, optional): flow curve used for the instruments like IR basis swap (if empty, but needed --> Analytics_FAIL). Defaults to None.

    Returns:
        DiscountCurve: _description_
    """

    # other factors needed in general taken from pyvacon implementation
    # //! @param refDate the reference date for the new curve  - have
    # //! @param objId the object id for the new Curve -have
    # //! @param instruments the instruments used for botstrapping -have
    # //! @param quotes the quotes, in the same order as the instruments -have
    # //! @param curves the curves required for valuing the bootstrap instruments (besides the curve to be bootstrapped)
    # //! @param dcType daycounter for the new curve -have
    # //! @param interType the interpolation type for the new curve - NEEDS TO BE SPECIFIED
    # //! @param extraType the extrapolation type for the new curve - NEEDS TO BE SPECIFIED
    # //! @param baseCurveName if non-empty, the curve will be constructed as a shifted curve over the specified - For FUTURE implementations
    # //! base curve
    # //! @param tolerance the required tolerance in the zero rates - EXPLAIN
    # //! @param maxIterations the maximim number of iterations (after that the bootstrapper fails) - EXPLAAIN

    # Sanity checks:
    assert len(instruments) == len(quotes), "Number of quotes does not equal number of instruments."

    #############################################################
    # initialize: # alternatively..
    yc_dates = [ref_date]
    dfs = [1.0]
    dcc = DayCounter(day_count_convention)

    #############################################################
    # Sort instruments # Obtain dates
    # check for instruments with duplicate end dates # for now, through exceptiion if there is
    # Sort instruments by end date and check for uniqueness
    instruments_by_date = {}
    for i, inst in enumerate(instruments):
        end_date = inst.get_end_date()  # implment for all specs #TODO
        if end_date in instruments_by_date:
            raise Exception(f"Duplicate expiry date found: {end_date}")
        instruments_by_date[end_date] = (quotes[i], inst)

    #############################################################
    # base curve creatiion????

    #############################################################
    # # start with loglinear interpolation to obtain good initial values for all dates
    # bootstrap loop over ordered expiry dates
    for end_date in sorted(instruments_by_date):
        quote, inst = instruments_by_date[end_date]  # use the quote to compare with brentq
        yc_dates.append(end_date)  # next datet
        dfs.append(dfs[-1])  # append a dummy value for the next date

        try:
            solution = brentq(error_fn, 0.00001, 5.0, xtol=1e-5)  # TODO define this error_fn, read on brentq usage
            dfs[-1] = solution
        except Exception as e:
            raise Exception(f"Initial bootstrap failed at {end_date}: {str(e)}")

    # In principle, this will have produced a curve.

    # start with next end date
    # solve for discount factor so that MODEL (bootstrapped) quote matches MARKET (input) quote (use brentq root finding)
    # check for error size between model and market

    #############################################################
    # Iterative refinement with real interpolator
    # this is to improve the values???
    # check for convergence: max change in zero rate estimate must be below tolerance.
    # max_diff = float("inf")
    max_diff = 0.0
    iteration = 0
    while iteration < max_iterations and (max_diff > tolerance or iteration == 0):

        total_evals = 0  # ??number of attempts?

        for i, end_date in enumerate(sorted(instruments_by_date), start=1):  # iterate through all end dates?
            quote, inst = instruments_by_date[end_date]  # use the quote to compare with brentq, must pass into error function somehow

            try:
                # tol_brent = dfs[i] * tolerance * dcc.yf(ref_date, end_date) * 0.1
                tol_brent = dfs[i] * tolerance * dcc.yf(ref_date, end_date) * 0.1
                dfs[i] = brentq(error_fn, 0.00001, 5.0, xtol=tol_brent)
                total_evals += 1
            except Exception as e:
                raise Exception(f"Refinement failed at {end_date}: {str(e)}")

        max_diff = 0.0
        # Convergence check
        for i, end_date in enumerate(sorted(instruments_by_date), start=1):
            # calculate derivative dq/dr using finite differences
            # (q=quote, r=zero rate)
            quote, inst = instruments_by_date[end_date]
            yc = DiscountCurve(dfs, yc_dates, interpolation_type, extrapolation_type)
            q_model = inst.get_quote(
                ref_date, curves + [yc]
            )  # here we need to better define the get_quote function, and which curves arer being passed to it...
            epsilon = 1e-6
            dfs_perturbed = dfs.copy()
            dfs_perturbed[i] += epsilon  # pertrub only at position = i
            yc_perturbed = DiscountCurve(dfs_perturbed, yc_dates, interpolation_type, extrapolation_type)
            q_model_eps = inst.get_quote(
                ref_date, curves + [yc_perturbed]
            )  # here we need to better define the get_quote function, and which curves arer being passed to it...

            dq = (q_model_eps - q_model) / epsilon
            dr = abs((quote - q_model) / (dq * dcc.yf(ref_date, end_date) * dfs[i]))
            max_diff = max(max_diff, dr)

        iteration += 1

    if max_diff > tolerance:
        raise Exception("Bootstrapping did not converge within tolerance.")

    # adding 150Y pillar to avoid expicit extrapolation???

    # create final discount curve
    curve = DiscountCurve(
        id=curve_id,
        refdate=ref_date,
        dates=yc_dates,  # populate with correct dates
        df=dfs,  # populated with corresponding discount factors
        interpolation=InterpolationType.LINEAR,
        extrapolation=ExtrapolationType.NONE,
        daycounter=day_count_convention,
    )

    return curve


# get_end_date (instrument specification) # maturity? or accrual period end date?
# return end date

# get_quote(ref_date, instrumentSpec, yield_curve, discount_curve, basis_curve)
# if basisSwap
# if discoutncurve given:
#    computeBasisSpread from the pricer
# else:
#    computeBasisSpread with different inputs, using yield curve in place of discount curve
# else if IR swap
# if discoutncurve given:
#    computeSwapRate from the pricer
# else:
#    computeSwapRate with different inputs, using yield curve in place of discount curve


# ellseif deposit
# return DepositPricer::impliedSimplyCompoundedRate(refDate, yc, spread, depo);

# Hence return an INTEREST RATE as final results...i.e. a quote


# Compute Error - This method computes the diff between market quote and candidate
def error_fn(df_val: float):
    df_tmp = dfs.copy()
    df_tmp[i] = df_val
    yc = DiscountCurve(df_tmp, yc_dates, interpolation_type, extrapolation_type)
    return inst.get_quote(ref_date, curves + [yc]) - quote


# here we need to better define the get_quote function, and which curves arer being passed to it...


def get_quote(instrument_spec, other_params: dict = {}):  # use pricer for each instrument or equivalent functions...

    if instrument_spec


    pass


# Main

if __name__ == "__main__":
    pass
