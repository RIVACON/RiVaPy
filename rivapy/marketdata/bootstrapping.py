# 2025.07.23 Hans Nguyen
# Boostrapping in rivapy indepedent of pyvacon


from rivapy.marketdata._logger import logger


##########
# Modules
import copy
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
    InterestRateBasisSwapSpecification,
)
from rivapy.marketdata import DiscountCurve
from rivapy.marketdata.fixing_table import FixingTable
from rivapy.tools.enums import DayCounterType, RollConvention, RollRule, InterpolationType, ExtrapolationType, Instrument
from rivapy.tools.datetools import DayCounter

from scipy.optimize import brentq


# import quote calculators # TODO SUBJECT TO CHANGE based on architecture
from rivapy.pricing.deposit_pricing import DepositPricer
from rivapy.pricing.fra_pricing import ForwardRateAgreementPricer
from rivapy.pricing.interest_rate_swap_pricing import InterestRateSwapPricer


##########
# Classes


######################################################
##########
# Functions
def bootstrap_curve(
    ref_date: _Union[date, datetime],
    curve_id: str,
    day_count_convention: _Union[DayCounterType, str],
    instruments: _List,
    quotes: _List,
    curves: dict = None,
    # discount_curve: DiscountCurve = None,
    # basis_curve: DiscountCurve = None,
    interpolation_type: InterpolationType = InterpolationType.LINEAR,
    extrapolation_type: ExtrapolationType = ExtrapolationType.LINEAR,
    tolerance: float = 1.0e-6,
    max_iterations: int = 10000,
) -> DiscountCurve:
    """

    Args:
        ref_date (_Union[date, datetime]): the reference for the new curve
        curve_id (str): Id for the new Curve
        day_count_convention (_Union[DayCounterType, str]): daycounter for the new curve
        instruments (_List): instrument specifications that are used in the calibration (deposits, FRAs, and swaps allowed atm)
        quotes (_List): the rate quotes for the instruments (deposit rates, FRAs and swap rates)
        curves (dict): curves to be used during bootstrapping such as discount curve and forward curve if given. Defaults to Empty
        interpolation_type (InterpolationType): interpolation method to be used by the final curve. defaults to LINEAR
        extrapolation_type (ExtrapolationType): extrapolation method to be used by the final curve. defaults to LINEAR
        tolerance (float): tolerance value used in refinement of the zero rates
        max_iterations (int): the maximim number of iterations (after that the bootstrapper fails)


    Returns:
        DiscountCurve: bootstrapped discount curve
    """
    # print("USING BOOTSTRAPPER V1----------------########################------------")  # DEBUG and REMOVE
    logger.info("Starting bootstrapper.")

    # Sanity checks:
    logger.info("Input sanity checks")
    assert len(instruments) == len(quotes), "Number of quotes does not equal number of instruments."
    # TODO implement more input quality checks:
    # curves given of correct type that match instrument type - or will this be done in the "market container" class?

    logger.info("Curve dictionary import")
    if curves == None:
        curves = {}
        logger.info("* curves dictionary is empty, will bootstrap single discount curve")
    else:
        logger.info("* curves dictionary provided, will bootstrap forward curve")

    #############################################################
    # initialize:
    logger.info("discount curve value init")

    yc_dates = [ref_date]
    dfs = [1.0]

    #############################################################
    # NEW FEATURE: Handle curve extension via curves["initial_curve"] - if provided
    # useful for using e.g. TBS to extend existing curves
    initial_curve = None
    initial_dates_set = set()
    if curves is not None and "initial_curve" in curves and curves["initial_curve"] is not None:
        initial_curve = curves["initial_curve"]
        if not isinstance(initial_curve, DiscountCurve):
            raise Exception("initial_curve must be a DiscountCurve instance")
        ic_dates = initial_curve.get_dates()
        ic_dfs = initial_curve.get_df()
        if len(ic_dates) == 0 or len(ic_dfs) == 0:
            raise Exception("initial_curve must contain at least one date and discount factor")
        for d, df in zip(ic_dates, ic_dfs):
            if d <= ref_date:
                continue
            yc_dates.append(d)
            dfs.append(df)
            initial_dates_set.add(d)
        logger.info(f"Prepopulated curve with {len(initial_dates_set)} initial points from initial_curve.")

    # init DCC
    dcc = DayCounter(day_count_convention)
    if isinstance(day_count_convention, str):  # normalizes type
        day_count_convention = DayCounterType(day_count_convention)

    #############################################################
    # Sort instruments # Obtain dates
    # check for instruments with duplicate end dates # for now, through exceptiion if there is
    #
    logger.info("Duplicate instrument date filter")
    instruments_by_date = {}
    for i, inst in enumerate(instruments):
        end_date = inst.get_end_date()
        if end_date in instruments_by_date:
            raise Exception(f"Duplicate expiry date found: {end_date}")
        instruments_by_date[end_date] = (quotes[i], inst)

    if initial_dates_set:
        filtered_instruments_by_date = {}
        for d, val in instruments_by_date.items():
            if d in initial_dates_set:
                logger.info(f"Skipping instrument at {d} because it exists in initial_curve.")
                continue
            filtered_instruments_by_date[d] = val
        instruments_by_date = filtered_instruments_by_date

    #############################################################
    # Determine instrument types provided
    logger.info("Determine instrument types provided")
    ins_types = set(inst.ins_type() for inst in instruments)
    has_fra = Instrument.FRA in ins_types
    has_irs = Instrument.IRS in ins_types
    has_bs = Instrument.BS in ins_types
    has_deposit = Instrument.DEPOSIT in ins_types

    # Determine single vs multi-curve bootstrap
    if "discount_curve" not in curves:
        # Single-curve bootstrap: create discount curve from instruments
        flag_multi_curve = False
        curves["discount_curve"] = DiscountCurve(
            "bootstrapped_discount",
            ref_date,
            yc_dates,
            dfs,
            interpolation_type,
            extrapolation_type,
            day_count_convention,
        )
        # Forward curve for IRS = discount curve (if any IRS present)
        flag_irs_bootstrapped_as_fwd = has_irs or has_bs
        if flag_irs_bootstrapped_as_fwd:
            curves["fixing_curve"] = curves["discount_curve"]

    else:
        # Multi-curve bootstrap: discount curve is provided, we bootstrap forward curve
        flag_multi_curve = True

        # FRAs and IRS will build forward curve
        if "fixing_curve" not in curves:
            # Create empty forward curve placeholder
            curves["fixing_curve"] = DiscountCurve(
                "bootstrapped_forward",
                ref_date,
                yc_dates,
                dfs,  # optionally start with empty DFs
                interpolation_type,
                extrapolation_type,
                day_count_convention,
            )
        flag_irs_bootstrapped_as_fwd = False

    # Sanity checks
    if has_deposit and flag_multi_curve:
        raise Exception("Deposits cannot be used in multi-curve bootstrapping")

    # Logging
    logger.info(f"Bootstrap mode: {'multi-curve' if flag_multi_curve else 'single-curve'}")
    logger.info(f"Instruments present: {ins_types}")
    logger.info(f"IRS uses bootstrapped curve as forward: {flag_irs_bootstrapped_as_fwd}")

    #############################################################
    # # start with loglinear interpolation to obtain good initial values for all dates

    lower = 1.0e-5  # initial bracket values for brentq
    upper = 5.0

    logger.info("Sort instruments by date and start bootstrapping")
    for end_date in sorted(instruments_by_date):
        # If end_date already exists (from initial_curve), skip solving it
        # if end_date in yc_dates:
        #     logger.info(f"Skipping instrument at {end_date}: already present in initial_curve.")
        #     continue
        quote, inst = instruments_by_date[end_date]  # use the market quote to compare with brentq
        yc_dates.append(end_date)  # next date
        prev_df = dfs[-1]  # previous discount factor for bracket search
        dfs.append(prev_df)  # append a dummy value for the next date

        # arguments to be passed to the error function for the brentq root solver
        ARGS = (
            -1,  # since we will look at the latest addition to our discount curve.
            dfs,
            yc_dates,
            inst,
            ref_date,
            quote,
            curves,
            InterpolationType.LINEAR_LOG,
            ExtrapolationType.LINEAR_LOG,
            day_count_convention,
            flag_irs_bootstrapped_as_fwd,
            flag_multi_curve,
        )

        try:

            # solution = brentq(error_fn, lower, upper, ARGS, xtol=1e-6)
            # dfs[-1] = solution
            # logger.debug(f"Bootstrapped DF for {end_date}: {solution} for {inst.ins_type()}")

            # -  DEBUG 11.2025
            # just before calling find_bracket for the failing end_date
            # print("INSIDE LOOP FOR BOOTSTRAP")
            # print("---- DEBUG START for end_date:", end_date, "quote(raw):", quote)
            # print("prev_df (guess):", prev_df)

            # # Evaluate error_fn at a few DF points (inside realistic DF support (1e-8, 1.0]))
            # test_dfs = [max(1e-10, prev_df * 0.5), max(1e-10, prev_df * 0.9), min(0.9999999, prev_df * 1.0), min(0.9999999, prev_df * 1.1)]
            # for td in test_dfs:
            #     try:
            #         val = error_fn(
            #             td,
            #             -1,
            #             dfs,
            #             yc_dates,
            #             inst,
            #             ref_date,
            #             quote,
            #             curves,
            #             # InterpolationType.HAGAN_DF,
            #             # ExtrapolationType.CONSTANT_DF,
            #             InterpolationType.LINEAR_LOG,
            #             ExtrapolationType.LINEAR_LOG,
            #             day_count_convention,
            #             flag_irs_bootstrapped_as_fwd,
            #             flag_multi_curve,
            #         )
            #     except Exception as e:
            #         val = f"EXC:{e}"
            # print(f"error_fn({td:.12f}) = {val}")

            # Also quickly check the sign/units of quote here:
            # print("Raw quote value (from market):", quote, "— are these bps? If so, convert: quote = quote*1e-4")
            # -

            lower, upper = find_bracket(error_fn, prev_df, ARGS)
            # lower, upper = prev_df * 0.8, prev_df * 1.2  # this is not good enouhg to work for all cases c.f. above

            logger.debug(f"Finding lower: {lower} and upper: {upper} bracket for root finding")

            # -  DEBUG 11.2025
            # print(
            #     f"-----------------------------------------[BOOTSTRAP] Solving for DF of {end_date}, initial guess: {prev_df}, market quote: {quote}"
            # )
            # # -  DEBUG 11.2025

            solution, result = brentq(
                error_fn,
                lower,
                upper,
                args=ARGS,
                xtol=1e-6,
                full_output=True,  # <--- enables access to iteration info
                disp=True,  # optional: prints if solver fails
            )
            dfs[-1] = solution
            # Log detailed solver diagnostics
            logger.debug(
                f"Bootstrapped DF for {end_date}: {solution:.10f} "
                f"for {inst.ins_type()} | "
                f"iterations={result.iterations}, "
                f"function_calls={result.function_calls}, "
                f"converged={result.converged}"
            )

            # -  DEBUG 11.2025
            # print(f"-----------------------------------------[BOOTSTRAP] Solved DF({end_date}) = {solution}")

        except Exception as e:
            raise Exception(f"Initial bootstrap failed at {end_date}: {str(e)}")

    # In principle, this will have produced a curve. It can be improved upon with refinement
    logger.info("Initial curve produced")
    #############################################################
    # Iterative refinement with real interpolator
    # this is to improve the values for the whole curve, as each subsequent point is dependant on the previous ones
    # check for convergence: max change in zero rate estimate must be below tolerance.
    # max_diff = float("inf")
    logger.info("Iterative refinement step")
    max_diff = 0.0
    iteration = 0
    # --- NEW: map end_date to correct index in dfs --- for the case of extension via initial_curve creating offset
    end_date_to_index = {d: yc_dates.index(d) for d in yc_dates if d in instruments_by_date}
    while iteration < max_iterations and (max_diff > tolerance or iteration == 0):

        total_evals = 0  #

        # for i, end_date in enumerate(sorted(instruments_by_date), start=1):  # iterate through all end dates

        # --- Use only end_dates from instruments_by_date, get correct dfs index ---
        for end_date in sorted(instruments_by_date):  # iterate through all end dates
            i = end_date_to_index[end_date]  # <--- CHANGED: dynamically compute index instead of enumerate
            quote, inst = instruments_by_date[end_date]  # use the quote to compare with brentq

            ARGS = (
                i,
                dfs,  # At this stage, these are all the solved for discount factors
                yc_dates,  # At this stage, this is the full list of dates of the discount curve
                inst,
                ref_date,
                quote,
                curves,
                interpolation_type,
                extrapolation_type,
                day_count_convention,
                flag_irs_bootstrapped_as_fwd,
                flag_multi_curve,
            )

            try:
                # used to determine the tolerance for brentq - scaled by discount factor and maturity and a heuristic 10% ontop to keep from over fitting
                tol_brent = dfs[i] * tolerance * dcc.yf(ref_date, end_date) * 0.1
                # print("------------------------refinement tolerance:")
                # print(f"{i} DF:{dfs[i]} * {tolerance} * {dcc.yf(ref_date, end_date)} * 0.1 = {tol_brent}")
                dfs[i] = brentq(error_fn, 0.00001, 5.0, ARGS, xtol=tol_brent)
                total_evals += 1
                logger.debug(f" refinement DF for {end_date}: {dfs[i]} for {inst.ins_type()} with total evals: {total_evals}")

            except Exception as e:
                raise Exception(f"Refinement failed at {end_date}: {str(e)} total evals: {total_evals}")

        max_diff = 0.0
        # Convergence check
        # for i, end_date in enumerate(sorted(instruments_by_date), start=1):
        for end_date in sorted(instruments_by_date):
            i = end_date_to_index[end_date]  # <--- CHANGED: use mapped index

            # calculate derivative dq/dr using finite differences
            # (q=quote, r=zero rate)
            quote, inst = instruments_by_date[end_date]
            yc = DiscountCurve("dummy_id", ref_date, yc_dates, dfs, interpolation_type, extrapolation_type, day_count_convention)

            # Multi-curve logic possible logic and single curve
            bootstrap_context = {}  # only need to do one time for base and epsilon
            if flag_multi_curve:
                # This is a forward curve — use Given discount curve for discounting
                bootstrap_context["flag_multi_curve"] = True
                curves["fixing_curve"] = yc
                # Keep discount_curve unchanged
            else:
                # Single-curve: updating discount curve itself
                curves["discount_curve"] = yc
                if flag_irs_bootstrapped_as_fwd:  # if it is an irs instrument that needs the forward curve as well as it was not provided
                    curves["fixing_curve"] = yc

            q_model = get_quote(ref_date, inst, curves, bootstrap_context)  # this curves dict needs to have the updated YC

            epsilon = 1e-6
            dfs_perturbed = dfs.copy()
            dfs_perturbed[i] += epsilon  # perturb only at position = i
            yc_perturbed = DiscountCurve(
                "dummy_id_perturbed", ref_date, yc_dates, dfs_perturbed, interpolation_type, extrapolation_type, day_count_convention
            )

            # Multi-curve and single curve flag logic
            if flag_multi_curve:
                # This is a forward curve — use Given discount curve for discounting
                curves["fixing_curve"] = yc_perturbed
                # Keep discount_curve unchanged
            else:
                # Single-curve: updating discount curve itself
                curves["discount_curve"] = yc_perturbed
                if flag_irs_bootstrapped_as_fwd:  # if it is an irs instrument that needs the forward curve as well as it was not provided
                    curves["fixing_curve"] = yc_perturbed

            q_model_eps = get_quote(ref_date, inst, curves, bootstrap_context)

            dq = (q_model_eps - q_model) / epsilon
            # print(f"dq: {dq}")
            # print(f"yf: {dcc.yf(ref_date, end_date)}")
            # print(f"df: {dfs[i]}")
            dr = abs((quote - q_model) / (dq * dcc.yf(ref_date, end_date) * dfs[i]))
            max_diff = max(max_diff, dr)

        iteration += 1

    if max_diff > tolerance:
        raise Exception("Bootstrapping did not converge within tolerance.")

    # TODO adding 150Y pillar to avoid explicit extrapolation???

    # create final discount curve
    logger.info("Curve Complete and output")
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
    flag_multi_curve: bool = False,
):
    """Error function used for the bootstrapper using a brentq solver.
    Returns the differnce between an input target value and calculated
    model value.

    Given a list of corresponding dates and discount factors, create a disount curve object
    and update the curve dictionary necessary.

    Pass relevant instrument information in order to calculate the fair rate given the current
    curve data.

    #TODO think about how to IMPROVE implementation in the case where forward curve is the same as discount curve


    Args:
        df_val (float): discount factor value used as guess for next value of the bootstrapped discount curve
        index (int): list index of where to insert df_val. usually -1 is passed to ensure it is the last entry
        dfs (_List): list of predetermined discount factors
        yc_dates (_List): corresponding datetime objects
        instrument_spec (): instrument specific data
        ref_date (_Union[date, datetime]): reference date
        ref_quote (float): target quote to compare to
        curves (dict): dictionary of relevant curve data
        interpolation_type (InterpolationType): the interpolation method to be used by the curves
        extrapolation_type (ExtrapolationType): the extrapolation method to be used by the curves
        day_count_convention: day coutn convention to be used for the dummy curve built
        flag_irs_bootstrapped_as_fwd (bool): Flag to trigger if fixing curve is the same as discount curve
        flag_multi_curve (bool): Flag to trigger if multi-curve bootstrapping is there

    Returns:
        float: difference between target quote and calculated quote
    """
    df_tmp = copy.deepcopy(dfs)
    df_tmp[index] = df_val
    # here reference date is used as placeholder
    yc = DiscountCurve("bootstrappedYC", ref_date, yc_dates, df_tmp, interpolation_type, extrapolation_type, day_count_convention)
    curves_copy = copy.deepcopy(curves)
    bootstrap_context = {"flag_multi_curve": flag_multi_curve}
    # Multi-curve logic possible logic and ssingle curve
    if flag_multi_curve:
        # This is a forward curve — use Given discount curve for discounting
        curves_copy["fixing_curve"] = yc
        # Keep discount_curve unchanged
        # print("UPDATING ONLY FIXING CURVE IN MULTI CURVE BOOTSTRAP")
    else:
        # Single-curve: updating discount curve itself
        curves_copy["discount_curve"] = yc
        if flag_irs_bootstrapped_as_fwd:  # if it is an irs instrument that needs the forward curve as well or TBS
            curves_copy["fixing_curve"] = yc

    calc_quote = get_quote(ref_date, instrument_spec, curves_copy, bootstrap_context)

    # # DEBUG statement
    # print("----------------")
    # print("Error function trial curve")
    # print(yc.get_df())
    # print("----------------")
    # print(f"using {df_val} -> calc_quote: {calc_quote} - ref_quote: {ref_quote} = {calc_quote - ref_quote}")

    # inside error_fn, after constructing yc and curves_copy and computing calc_quote
    # compute model_residual = calc_quote - ref_quote or whichever sign convention you use
    # print(f"[DEBUG error_fn] df_val={df_val:.12f}, calc_quote={calc_quote}, ref_quote={ref_quote}, residual={calc_quote - ref_quote}")
    # optionally print underlying leg PVs (you can return them from compute_basis_spread or log inside).

    return calc_quote - ref_quote


def find_bracket(error_fn, guess, args, expand=2.0, max_tries=10, min_lower=1e-8, max_upper=10.0):
    """
    Tries to find a [lower, upper] bracket where error_fn(lower) and error_fn(upper)
    have opposite signs, indicating a root lies between them.

    Parameters
    ----------
    error_fn : callable
        Your pricing error function (same as passed to brentq).
    guess : float
        A rough estimate for the root (e.g., previous DF).
    args : tuple
        Extra arguments passed to error_fn.
    expand : float
        Factor by which to widen the bracket each iteration.
    max_tries : int
        Maximum number of expansions before giving up.
    min_lower, max_upper : float
        Hard limits to keep brackets within safe numeric bounds.
    """
    lower = max(guess * 0.8, min_lower)
    upper = min(guess * 1.2, max_upper)

    f_lower = error_fn(lower, *args)
    f_upper = error_fn(upper, *args)

    tries = 0
    while f_lower * f_upper > 0 and tries < max_tries:
        # Expand symmetrically outward
        lower = max(lower / expand, min_lower)
        upper = min(upper * expand, max_upper)
        f_lower = error_fn(lower, *args)
        f_upper = error_fn(upper, *args)
        tries += 1

    if f_lower * f_upper > 0:
        raise RuntimeError("Could not find valid bracket for brentq")

    return lower, upper


def get_quote(
    ref_date: _Union[date, datetime],
    instrument_spec: _Union[
        DepositSpecification, ForwardRateAgreementSpecification, InterestRateSwapSpecification, InterestRateBasisSwapSpecification
    ],
    curve_dict: dict,
    bootstrap_context=None,
):
    """Get the instrument specific fair quote calculation result to be used in the bootstrapper.

    Args:
        ref_date (_Union[date, datetime]):
        instrument_spec (_Union[DepositSpecification, ForwardRateAgreementSpecification, InterestRateSwapSpecification]):
        curve_dict (dict): Dictionary containing the market data curves needed for discounting or fwd rates.

    Returns:
        float: calculated fair rate
    """

    quote = 0.0
    if instrument_spec.ins_type() == Instrument.DEPOSIT:

        # old
        discount_curve = curve_dict["discount_curve"]
        # spread_curve=curve_dict["spread_curve"]
        quote = DepositPricer.get_implied_simply_compounded_rate(ref_date, instrument_spec, discount_curve)  # TODO assumes no spread curve for now

    elif instrument_spec.ins_type() == Instrument.FRA:
        if bootstrap_context is None or bootstrap_context.get("flag_multi_curve", False) == False:
            # single curve case
            curve_used = curve_dict["discount_curve"]
        else:
            # multicurve, where discount given, FRA builds forward
            curve_used = curve_dict["fixing_curve"]
        quote = ForwardRateAgreementPricer.compute_fair_rate(ref_date, instrument_spec, discount_curve=curve_used)

    elif instrument_spec.ins_type() == Instrument.IRS:

        yc_discount = curve_dict["discount_curve"]
        yc_forward = curve_dict["fixing_curve"]
        # according to pyvayon example, the fixing table is assumed to default to empty to allow the code to run...
        fixing_table = FixingTable()

        float_leg = instrument_spec.get_float_leg()
        fixed_leg = instrument_spec.get_fixed_leg()
        fixing_grace_period = 0  # TODO take in as parameter? in pyvacon example, the extra swap parameters are assumed to be empty, only the curves were passed as arguments...

        # parameters specific to ir swap bootstrapping, in regards to the fixed leg for calculating the fair swap rate
        # needed to pass these settings onto the InterestRateSwapPricer.price_leg
        pricing_params = {"fixing_grace_period": fixing_grace_period, "set_rate": True, "desired_rate": 1.0}

        quote = InterestRateSwapPricer.compute_swap_rate(ref_date, yc_discount, yc_forward, float_leg, fixed_leg, fixing_table, pricing_params)

    elif instrument_spec.ins_type() == Instrument.BS:  #  basis swap # E.g. tenor basis swap

        yc_discount = curve_dict["discount_curve"]
        yc_forward = curve_dict["fixing_curve"]  # THIS IS THE CURVE TO BE SOLVED
        yc_basis_curve = curve_dict.get("basis_curve", None)  # THIS IS THE EXISTING KNOWN CURVE - assume is for SHORT
        # NOTE - the rerquirement is that error_fn has the flags to determine if discoutn curve is the same as fixing curve or not already

        if yc_basis_curve is None:
            raise Exception("Missing basis curve for pricing TBS")

        fixing_table = FixingTable()

        pay_leg = instrument_spec.get_pay_leg()
        receive_leg = instrument_spec.get_receive_leg()
        spread_leg = instrument_spec.get_spread_leg()
        fixing_grace_period = 0  # TODO take in as parameter? in pyvacon example, the extra swap parameters are assumed to be empty, only the curves were passed as arguments...
        pricing_params = {
            "fixing_grace_period": fixing_grace_period,
            "set_rate": True,
            "desired_rate": 1.0,
        }  # need annuity again for spread_leg(modeled as fixed leg)

        quote = InterestRateSwapPricer.compute_basis_spread(
            ref_date,
            discount_curve=yc_discount,
            payLegFixingCurve=yc_basis_curve,
            receiveLegFixingCurve=yc_forward,
            pay_leg=pay_leg,
            receive_leg=receive_leg,
            spread_leg=spread_leg,
            fixing_map=fixing_table,
            pricing_params=pricing_params,
        )

    # # DEBUG
    # print(f"Calculated quote for {instrument_spec.ins_type()} is {quote}")
    return quote


def bootstrap_curve_from_quote_table(input_data):
    pass


# Main

if __name__ == "__main__":
    pass
