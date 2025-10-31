# 2025.07.23 Hans Nguyen
# Boostrapping in rivapy indepedent of pyvacon


from rivapy.marketdata._logger import logger


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
    # initialize: # alternatively..
    logger.info("discount curve value init")
    yc_dates = [ref_date]
    dfs = [1.0]
    if isinstance(day_count_convention, str):  # normalizes type
        day_count_convention = DayCounterType(day_count_convention)

    dcc = DayCounter(day_count_convention)

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

    #############################################################
    # base curve creatiion check #TODO think about improving how to handle input curves if given for multicurve bootstrapping
    # given instrument types, check for required curves

    logger.info("Determine instrument types provided")
    ins_types = []
    flag_irs_bootstrapped_as_fwd = False
    for inst in instruments:
        ins_type = inst.ins_type()
        if ins_type not in ins_types:
            ins_types.append(ins_type)

    logger.info("Toggle single/multi curve bootstrapping")
    if "discount_curve" not in curves:
        flag_multi_curve = False
        curves["discount_curve"] = DiscountCurve(
            "dummy_id_discount", ref_date, yc_dates, dfs, interpolation_type, extrapolation_type, day_count_convention
        )
        # this means this is the target output curve
    else:  # This means the discount curve was given. We therefore want to output a FORWARD curve, e.g. 3M, 6M,...
        flag_multi_curve = True

    # cannot multicurve bootstrap with deposits involved

    if Instrument.DEPOSIT in ins_types and flag_multi_curve == True:
        raise Exception("Deposits cannot be used in multicurve bootstrapping")

    if Instrument.IRS or Instrument.BS in ins_types:
        # check if curves has a fixing curve
        if "fixing_curve" in curves:
            if not isinstance(curves["fixing_curve"], DiscountCurve):
                raise Exception("Fixing curve is not of type DiscountCurve")

        else:
            logger.info("IRS swap present but no fixing curve provided, will use bootstrapped curve in place")
            flag_irs_bootstrapped_as_fwd = True
            if flag_multi_curve:
                curves["fixing_curve"] = DiscountCurve(
                    "dummy_id_fixing", ref_date, yc_dates, dfs, interpolation_type, extrapolation_type, day_count_convention
                )
            else:
                curves["fixing_curve"] = curves["discount_curve"]

    #############################################################
    # # start with loglinear interpolation to obtain good initial values for all dates
    # this means i have to pass into the rror function the interpolation types desired which is different
    # from the inter and extra type we want for the final discount curve
    # bootstrap loop over ordered expiry dates which is also sorted here

    lower = 1.0e-5  # initial bracket values for brentq
    upper = 5.0

    logger.info("Sort instruments by date and start bootstrapping")
    for end_date in sorted(instruments_by_date):
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

            lower, upper = find_bracket(error_fn, prev_df, ARGS)
            # lower, upper = prev_df * 0.8, prev_df * 1.2  # this is not good enouhg to work for all cases c.f. above

            logger.debug(f"Finding lower: {lower} and upper: {upper} bracket for root finding")

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

        except Exception as e:
            raise Exception(f"Initial bootstrap failed at {end_date}: {str(e)}")

    # In principle, this will have produced a curve. It can be improved upon with refinement

    #############################################################
    # Iterative refinement with real interpolator
    # this is to improve the values for the whole curve, as each subsequent point is dependant on the previous ones
    # check for convergence: max change in zero rate estimate must be below tolerance.
    # max_diff = float("inf")
    logger.info("Iterative refinement step")
    max_diff = 0.0
    iteration = 0
    while iteration < max_iterations and (max_diff > tolerance or iteration == 0):

        total_evals = 0  #

        for i, end_date in enumerate(sorted(instruments_by_date), start=1):  # iterate through all end dates
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
        for i, end_date in enumerate(sorted(instruments_by_date), start=1):
            # calculate derivative dq/dr using finite differences
            # (q=quote, r=zero rate)
            quote, inst = instruments_by_date[end_date]
            yc = DiscountCurve("dummy_id", ref_date, yc_dates, dfs, interpolation_type, extrapolation_type, day_count_convention)

            # Multi-curve logic possible logic and single curve
            if flag_multi_curve:
                # This is a forward curve — use Given discount curve for discounting
                curves["fixing_curve"] = yc
                # Keep discount_curve unchanged
            else:
                # Single-curve: updating discount curve itself
                curves["discount_curve"] = yc
                if flag_irs_bootstrapped_as_fwd:  # if it is an irs instrument that needs the forward curve as well as it was not provided
                    curves["fixing_curve"] = yc

            q_model = get_quote(ref_date, inst, curves)  # this curves dict needs to have the updated YC

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

            q_model_eps = get_quote(ref_date, inst, curves)

            dq = (q_model_eps - q_model) / epsilon
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
    df_tmp = dfs.copy()
    df_tmp[index] = df_val
    # here reference date is used as placeholder
    yc = DiscountCurve("bootstrappedYC", ref_date, yc_dates, df_tmp, interpolation_type, extrapolation_type, day_count_convention)
    curves_copy = curves.copy()

    # Multi-curve logic possible logic and ssingle curve
    if flag_multi_curve:
        # This is a forward curve — use Given discount curve for discounting
        curves_copy["fixing_curve"] = yc
        # Keep discount_curve unchanged
    else:
        # Single-curve: updating discount curve itself
        curves_copy["discount_curve"] = yc
        if flag_irs_bootstrapped_as_fwd:  # if it is an irs instrument that needs the forward curve as well or TBS
            curves_copy["fixing_curve"] = yc

    calc_quote = get_quote(ref_date, instrument_spec, curves_copy)

    # # DEBUG statement
    # print("----------------")
    # print("Error function trial curve")
    # print(yc.get_df())
    # print("----------------")
    # print(f"using {df_val} -> calc_quote: {calc_quote} - ref_quote: {ref_quote} = {calc_quote - ref_quote}")
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

        curve_used = curve_dict["discount_curve"]
        quote = ForwardRateAgreementPricer.compute_fair_rate(ref_date, instrument_spec, forward_curve=curve_used)

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
        # 	return InterestRateSwapPricer::computeBasisSpread(
        # refDate, ycDiscount, ycFwdReceive, ycFwdPay,
        # basisSwap->getReceiveLeg(), basisSwap->getPayLeg(), basisSwap->getSpreadLeg(),
        # std::make_shared<const FixingTable>(),
        # std::make_shared<const InterestRateSwapPricingParameter>()

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

        # TODO NEED TO HANDLE WHICH SITUATION WE ARE IN in case which curves are given etc...
        # HERE IS THE GENRAL GET QUOTE ARGUMENTS
        #         double YieldCurveBootstrapper::getQuote(const boost::posix_time::ptime& refDate,
        # 	const std::shared_ptr<const BaseSpecification>& instrument,
        # 	const std::shared_ptr<const DiscountCurve>& yc,
        # 	const std::shared_ptr<const DiscountCurve>& discountCurve,
        # 	const std::shared_ptr<const DiscountCurve>& basisCurve)
        # {
        # WHAT TO DO IN CASE OF BASIS SWAP
        # 	const std::shared_ptr<const InterestRateBasisSwapSpecification> basisSwap = std::dynamic_pointer_cast<const InterestRateBasisSwapSpecification>(instrument);
        # 	if (basisSwap != nullptr) {
        # 		if (basisCurve != nullptr) {
        # 			if (discountCurve != nullptr)
        # 				return InterestRateSwapPricer::computeBasisSpread(
        # 					refDate, discountCurve, yc, basisCurve,
        # 					basisSwap->getReceiveLeg(), basisSwap->getPayLeg(), basisSwap->getSpreadLeg(),
        # 					std::make_shared<const FixingTable>(),
        # 					std::make_shared<const InterestRateSwapPricingParameter>()
        # 				);
        # 			else
        # 				return InterestRateSwapPricer::computeBasisSpread(
        # 					refDate, yc, yc, basisCurve,
        # 					basisSwap->getReceiveLeg(), basisSwap->getPayLeg(), basisSwap->getSpreadLeg(),
        # 					std::make_shared<const FixingTable>(),
        # 					std::make_shared<const InterestRateSwapPricingParameter>()
        # 				);
        # 		}
        # 		else {
        # 			Analytics_FAIL("Missing basis curve for pricing basis swap");
        # 		}
        # 	}

    # # DEBUG
    # print(f"Calculated quote for {instrument_spec.ins_type()} is {quote}")
    return quote


def bootstrap_curve_from_quote_table(input_data):
    pass


# Main

if __name__ == "__main__":
    pass
