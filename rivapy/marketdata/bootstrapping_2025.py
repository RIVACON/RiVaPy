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
from rivapy.marketdata.fixing_table import FixingTable
from rivapy.tools.enums import DayCounterType, RollConvention, RollRule, InterpolationType, ExtrapolationType, Instrument
from rivapy.tools.datetools import DayCounter

from scipy.optimize import brentq


# import quote calculators
from rivapy.pricing.deposit_pricing import DepositPricer  # ?SUBJECT TO CHANGE based on architecture
from rivapy.pricing.fra_pricing import ForwardRateAgreementPricer
from rivapy.pricing.interest_rate_swap_pricing import InterestRateSwapPricer


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
    curves: dict = {},
    # discount_curve: DiscountCurve = None,
    # basis_curve: DiscountCurve = None,
    interpolation_type: InterpolationType = InterpolationType.LINEAR,
    extrapolation_type: ExtrapolationType = ExtrapolationType.LINEAR,
    tolerance: float = 1.0e-8,
    max_iterations: int = 10000,
) -> DiscountCurve:
    """

    Args:
        ref_date (_Union[date, datetime]): the reference for the new curve
        curve_id (str): Id for the new Curve
        day_count_convention (_Union[DayCounterType, str]): daycounter for the new curve
        instruments (_List): instrument specifications that are used in the calibration (deposits, FRAs, and swaps allowed atm)
        quotes (_List): the rate quotes for the instruments (deposit rates, FRAs and swap rates)
        discount_curve (DiscountCurve, optional): discount curve used for the instruments (if empty, the bootstrapped curve is used for discounting). Defaults to None.
        basis_curve (DiscountCurve, optional): flow curve used for the instruments like IR basis swap (if empty, but needed --> raise Exception). Defaults to None.

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
    #
    instruments_by_date = {}
    for i, inst in enumerate(instruments):
        end_date = inst.get_end_date()  # implement for all specs #TODO
        if end_date in instruments_by_date:
            raise Exception(f"Duplicate expiry date found: {end_date}")
        instruments_by_date[end_date] = (quotes[i], inst)

    #############################################################
    # base curve creatiion check
    # given instrument types, check for required curves
    ins_types = []
    flag_irs_bootstrapped_as_fwd = False
    for inst in instruments:
        ins_type = inst.ins_type()
        if ins_type not in ins_types:
            ins_types.append(ins_type)

    if "discount_curve" not in curves:
        curves["discount_curve"] = DiscountCurve(
            "dummy_id_discount", ref_date, yc_dates, dfs, interpolation_type, extrapolation_type, day_count_convention
        )

    if Instrument.IRS in ins_types:
        # check if curves has a fixing curve
        if "fixing_curve" in curves:
            if not isinstance(curves["fixing_curve"], DiscountCurve):
                raise Exception("Fixing curve is not of type DiscountCurve")
        else:
            print("IRS swap present but no fixing curve provided, will use bootstrapped curve in place")
            flag_irs_bootstrapped_as_fwd = True
            curves["fixing_curve"] = curves["discount_curve"]

    #############################################################
    # # start with loglinear interpolation to obtain good initial values for all dates #TODO make logliner interpolator
    # this means i have to pass into the rror function the interpolation types desired which is different
    # from the inter and extra type we want for the final discount curve
    # bootstrap loop over ordered expiry dates which is also sorted here

    lower = 1.0e-5  # DEBUG TODO REMOVE if not implement bracket search
    upper = 5.0

    for end_date in sorted(instruments_by_date):
        print("------------------------------------------------------------------------------------")  # DEBUG TODO REMOVE
        print("------------------------------------------------------------------------------------")  # DEBUG TODO REMOVE
        print("next instrument")  # DEBUG TODO REMOVE
        quote, inst = instruments_by_date[end_date]  # use the quote to compare with brentq
        yc_dates.append(end_date)  # next datet
        dfs.append(dfs[-1])  # append a dummy value for the next date
        # yc_temp = DiscountCurve("bootstrappedYC", ref_date, yc_dates, dfs, interpolation_type, extrapolation_type, day_count_convention) # should we move this out of error_fn?
        # what do you do  when discount curve is given in input???? TODO
        # curves["discount_curve"] = yc_temp
        ARGS = (
            -1,  # since we will look at the latest addition to our discount curve.
            dfs,
            yc_dates,
            inst,
            ref_date,
            quote,
            curves,
            InterpolationType.LINEAR_LOG,  # TODO LOGLINEAR TO BE IMPLEMENTED , default to LINEAR for testing purposes until implemented
            ExtrapolationType.LINEAR_LOG,
            day_count_convention,
            flag_irs_bootstrapped_as_fwd,
        )  # needed to run get_quote inside of error_fn , everythign other than the input guess of discount factor
        print("DEBUG: check if curve is growing incrementally")  # DEBUG
        print(dfs)
        try:
            # DEBUG TODO REMOV
            fa = error_fn(
                df_val=lower,
                index=ARGS[0],
                dfs=ARGS[1],
                yc_dates=ARGS[2],
                instrument_spec=ARGS[3],
                ref_date=ARGS[4],
                ref_quote=ARGS[5],
                curves=ARGS[6],  # or should it be dictionary?
                interpolation_type=ARGS[7],
                extrapolation_type=ARGS[8],
                day_count_convention=ARGS[9],
                flag_irs_bootstrapped_as_fwd=ARGS[10],
            )
            fb = error_fn(
                df_val=upper,
                index=ARGS[0],
                dfs=ARGS[1],
                yc_dates=ARGS[2],
                instrument_spec=ARGS[3],
                ref_date=ARGS[4],
                ref_quote=ARGS[5],
                curves=ARGS[6],  # or should it be dictionary?
                interpolation_type=ARGS[7],
                extrapolation_type=ARGS[8],
                day_count_convention=ARGS[9],
                flag_irs_bootstrapped_as_fwd=ARGS[10],
            )
            print("brentq evaluation at bounds")
            print(f"f({lower}) = {fa}")
            print(f"f({upper}) = {fb}")
            print(f"f(a)*f(b) = {fa*fb}")
            # initial_guess = dfs[-1]
            # lower, upper = find_bracket(error_fn, initial_guess, ARGS)

            # solution = brentq(error_fn, 0.00001, 5.0, xtol=1e-5)  # TODO define this error_fn, read on brentq usage
            # solution = brentq(error_fn, 0.00001, 5.0, ARGS, xtol=1e-5)  # TODO define this error_fn, read on brentq usage
            solution = brentq(error_fn, lower, upper, ARGS, xtol=1e-6)  # TODO define this error_fn, read on brentq usage
            dfs[-1] = solution

            if flag_irs_bootstrapped_as_fwd == True:  # meaning the passed forward curve needs to be updating alongside the discount curve
                curves["discount_curve"] = DiscountCurve(
                    "bootstrappedYC", ref_date, yc_dates, dfs, interpolation_type, extrapolation_type, day_count_convention
                )
                curves["fixing_curve"] = curves["discount_curve"]

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

            ARGS = (
                i,
                dfs,
                yc_dates,  # by now this is the full list of dates
                inst,
                ref_date,
                quote,
                curves,
                interpolation_type,  # TODO TO BE IMPLEMENTED , default to LINEAR for testing purposes until implemented
                extrapolation_type,
                day_count_convention,
                flag_irs_bootstrapped_as_fwd,
            )

            try:
                # used to determine the tolerance for brentq??? scaled by discoutn factor and maturity and a heuristic 10% ontop to keep from over fitting
                tol_brent = dfs[i] * tolerance * dcc.yf(ref_date, end_date) * 0.1
                print("refinement tolerance:")
                print(f"{i} DF:{dfs[i]} * {tolerance} * {dcc.yf(ref_date, end_date)} * 0.1 = {tol_brent}")
                # dfs[i] = brentq(error_fn, 0.00001, 5.0, xtol=tol_brent)
                dfs[i] = brentq(error_fn, 0.00001, 5.0, ARGS, xtol=tol_brent)
                total_evals += 1
            except Exception as e:
                raise Exception(f"Refinement failed at {end_date}: {str(e)}")

        max_diff = 0.0
        # Convergence check
        for i, end_date in enumerate(sorted(instruments_by_date), start=1):
            # calculate derivative dq/dr using finite differences
            # (q=quote, r=zero rate)
            quote, inst = instruments_by_date[end_date]
            # yc = DiscountCurve(dfs, yc_dates, interpolation_type, extrapolation_type)
            yc = DiscountCurve("dummy_id", ref_date, yc_dates, dfs, interpolation_type, extrapolation_type, day_count_convention)
            curves["discount_curve"] = yc
            if flag_irs_bootstrapped_as_fwd == True:  # meaning the passed forward curve needs to be updating alongside the discount curve
                curves["fixing_curve"] = curves["discount_curve"]
            q_model = get_quote(ref_date, inst, curves)  # this curves dict needs to have the updated YC
            # here we need to better define the get_quote function, and which curves arer being passed to it...
            epsilon = 1e-6
            dfs_perturbed = dfs.copy()
            dfs_perturbed[i] += epsilon  # pertrub only at position = i
            # yc_perturbed = DiscountCurve(dfs_perturbed, yc_dates, interpolation_type, extrapolation_type)
            yc_perturbed = DiscountCurve(
                "dummy_id_perturbed", ref_date, yc_dates, dfs_perturbed, interpolation_type, extrapolation_type, day_count_convention
            )
            curves["discount_curve"] = yc_perturbed
            if flag_irs_bootstrapped_as_fwd == True:  # meaning the passed forward curve needs to be updating alongside the discount curve
                curves["fixing_curve"] = curves["discount_curve"]
            q_model_eps = get_quote(ref_date, inst, curves)
            # here we need to better define the get_quote function, and which curves arer being passed to it...

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
        interpolation=interpolation_type,
        extrapolation=extrapolation_type,
        daycounter=day_count_convention,
    )

    return curve


# Compute Error - This method computes the diff between market quote and candidate
def error_fn(
    df_val: float,
    index: int,
    dfs: _List,
    yc_dates: _List,
    instrument_spec: _Union[DepositSpecification, ForwardRateAgreementSpecification, InterestRateSwapSpecification],
    ref_date: _Union[date, datetime],
    ref_quote: float,
    curves: dict,  # or should it be dictionary?
    interpolation_type: InterpolationType,
    extrapolation_type: ExtrapolationType,
    day_count_convention: DayCounterType = DayCounterType.ACT360,
    flag_irs_bootstrapped_as_fwd: bool = False,
):
    """_summary_

    Args:
        df_val (float): _description_
        dfs (_List): _description_
        yc_dates (_List): _description_
        interpolation_type (InterpolationType): _description_
        extrapolation_type (ExtrapolationType): _description_
        ref_date (_Union[date, datetime]):
        ref_quote (float): _description_
        curves (_List): _description_

    Returns:
        _type_: _description_
    """
    df_tmp = dfs.copy()
    df_tmp[index] = df_val
    # here reference date is used as placeholder
    yc = DiscountCurve("bootstrappedYC", ref_date, yc_dates, df_tmp, interpolation_type, extrapolation_type, day_count_convention)
    curves_copy = curves.copy()
    curves_copy["discount_curve"] = yc
    if flag_irs_bootstrapped_as_fwd:
        curves_copy["fixing_curve"] = yc

    print("----------------")
    print("Error function trial curve")
    print(yc.get_df())
    return get_quote(ref_date, instrument_spec, curves_copy) - ref_quote


def find_bracket(error_fn, initial_guess, *args):
    lower = initial_guess * 0.5
    upper = initial_guess * 1.5
    f_lower = error_fn(lower, *args)
    f_upper = error_fn(upper, *args)

    count = 0
    while f_lower * f_upper > 0 and count < 50:
        lower *= 0.5
        upper *= 1.5
        f_lower = error_fn(lower, *args)
        f_upper = error_fn(upper, *args)
        count += 1

    if f_lower * f_upper > 0:
        raise RuntimeError(f"Could not find a sign change around initial guess {initial_guess}")
    return lower, upper


# here we need to better define the get_quote function, and which curves arer being passed to it...


def get_quote(
    ref_date: _Union[date, datetime],
    instrument_spec: _Union[DepositSpecification, ForwardRateAgreementSpecification, InterestRateSwapSpecification],
    curve_dict: dict = {},
):  # use pricer for each instrument or equivalent functions...

    quote = 0.0
    if instrument_spec.ins_type() == Instrument.Deposit:

        # old
        discount_curve = curve_dict["discount_curve"]
        # spread_curve=curve_dict["spread_curve"]
        quote = DepositPricer.implied_simply_compounded_rate(ref_date, instrument_spec, discount_curve)  # assume no spread curve

    elif instrument_spec.ins_type() == Instrument.FRA:

        curve_used = curve_dict["discount_curve"]
        quote = ForwardRateAgreementPricer.compute_fair_rate(ref_date, instrument_spec, forward_curve=curve_used)

        # rate_start = instrument_spec.start_date
        # rate_end = instrument_spec.end_date

        # quote = ForwardRateAgreementPricer.compute_fair_rate(val_date=ref_date, forward_curve=curve_used, rate_start_date=rate_start, rate_end_date=rate_end)

    elif instrument_spec.ins_type() == Instrument.IRS:

        yc_discount = curve_dict["discount_curve"]  # TODO decide how to pass which curves
        yc_forward = curve_dict["fixing_curve"]
        # according to pyvayon example, the fixing table is assumed to defaulted to empty to allow the code to run...
        fixing_table = FixingTable()

        float_leg = instrument_spec.get_float_leg()
        fixed_leg = instrument_spec.get_fixed_leg()
        fixing_grace_period = 0  # TODO take in as parameter? in pyvacon example, the extra swap parameters are assumed to be empty, only the curves were passed as arguments...

        # parameters specific to ir swap bootstrapping, in regards to the fixed leg for calculating the fair swap rate
        pricing_params = {"fixing_grace_period": fixing_grace_period, "set_rate": True, "desired_rate": 1.0}

        quote = InterestRateSwapPricer.compute_swap_rate(ref_date, yc_discount, yc_forward, float_leg, fixed_leg, fixing_table, pricing_params)
        # InterestRateSwapPricer::computeSwapRate(
        # 		refDate, ycDiscount, ycForward, swapSpec->getFloatLeg(), swapSpec->getFixedLeg(),
        # 		std::make_shared<const FixingTable>(),
        # 		std::make_shared<const InterestRateSwapPricingParameter>()

    elif instrument_spec.ins_type() == Instrument.OIS:
        pass
    elif instrument_spec.ins_type() == Instrument.TBS:  # tenor basis swap
        # 	return InterestRateSwapPricer::computeBasisSpread(
        # refDate, ycDiscount, ycFwdReceive, ycFwdPay,
        # basisSwap->getReceiveLeg(), basisSwap->getPayLeg(), basisSwap->getSpreadLeg(),
        # std::make_shared<const FixingTable>(),
        # std::make_shared<const InterestRateSwapPricingParameter>()
        pass
    elif instrument_spec.ins_type() == Instrument.FXF:  # fx forward
        pass

    # DEBUG TODO REMOVE
    print(f"Calculated quote for {instrument_spec.ins_type()} is {quote}")
    return quote


# Main

if __name__ == "__main__":
    pass
