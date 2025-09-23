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
from rivapy.pricing.deposit_pricing import DepositPricer  # TODO SUBJECT TO CHANGE based on architecture
from rivapy.pricing.fra_pricing import ForwardRateAgreementPricer
from rivapy.pricing.interest_rate_swap_pricing import InterestRateSwapPricer


##########
# Classes

# TODO reading from table
# class InstrumentSpecFromTable:
#     """
#     Definition of input instruments for IR boostrapping from a given .CSV
#     with predefined columns.
#     """

#     def __init__(self, ref_date, input_data, holidays):

#         self.refDate = ref_date
#         self.instr = input_data["Instrument"]
#         self.fixDayCount = input_data["DayCountFixed"]
#         self.floatDayCount = input_data["DayCountFloat"]
#         self.basisDayCount = input_data["DayCountBasis"]
#         self.maturity = input_data["Maturity"]
#         self.tenor = input_data["UnderlyingTenor"]
#         self.underlyingPayFreq = input_data["UnderlyingPaymentFrequency"]
#         self.basisTenor = input_data["BasisTenor"]
#         self.basisPayFreq = input_data["BasisPaymentFrequency"]
#         self.fixPayFreq = input_data["PaymentFrequencyFixed"]
#         self.rollConvFloat = input_data["RollConventionFloat"]
#         self.rollConvFix = input_data["RollConventionFixed"]
#         self.rollConvBasis = input_data["RollConventionBasis"]
#         self.spotLag = input_data["SpotLag"]
#         self.label = self.instr + "_" + self.maturity
#         self.currency = input_data["Currency"]
#         self.holidays = holidays
#         self.parRate = input_data["Quote"]

#     def get_instrument(self):
#         """
#         Instrument specification based on the "Instrument" field the input data
#         """
#         if self.instr.upper() == Instrument.IRS:
#             instrument = self.get_irs_spec()
#         elif self.instr.upper() == Instrument.OIS:
#             instrument = self.get_irs_spec()
#         elif self.instr.upper() == Instrument.TBS:
#             instrument = self.get_tbs_spec()
#         elif self.instr.upper() == Instrument.DEPOSIT:
#             instrument = self.get_deposit_spec()
#         elif self.instr.upper() == Instrument.FRA:
#             instrument = self.get_fra_spec()
#         else:
#             raise ValueError("Unknown instrument type")
#         return instrument

#     def get_irs_spec(self):  # TODO
#         """
#         Specification for interest rate swaps
#         """
#         # get floating leg schedule
#         floatleg = self.get_float_leg(self.underlyingPayFreq, self.tenor, self.rollConvFloat, self.spotLag)
#         # get fix leg schedule
#         fixedleg = self.get_fix_leg(self.fixPayFreq, self.rollConvFix, self.spotLag)

#         # get expiry of swap (cannot be before last paydate of legs)
#         spot_date = get_end_date(self.refDate, self.spotLag)
#         expiry = get_end_date(spot_date, self.maturity)

#         # SecuritizationLevel is not used in the bootstrapping algorithm
#         ir_swap = pyvacon.finance.specification.InterestRateSwapSpecification(
#             self.label, "dummy_issuer", "COLLATERALIZED", self.currency, expiry, fixedleg, floatleg
#         )
#         return ir_swap

#     def get_tbs_spec(self):  # TODO
#         """
#         Specification for tenor basis swaps
#         """
#         # get floating leg schedule
#         floatleg = self.get_float_leg(self.underlyingPayFreq, self.tenor, self.rollConvFloat, self.spotLag)
#         floatleg_basis = self.get_float_leg(self.basisPayFreq, self.basisTenor, self.rollConvBasis, self.spotLag)

#         # get fix leg schedule
#         fixedleg = self.get_fix_leg(self.fixPayFreq, self.rollConvFix, self.spotLag)

#         # get expiry of swap (cannot be before last paydate of legs)
#         spot_date = get_end_date(self.refDate, self.spotLag)
#         expiry = get_end_date(spot_date, self.maturity)

#         # the basis leg should be the pay leg
#         basis_swap = pyvacon.finance.specification.InterestRateBasisSwapSpecification(
#             self.label, "dummy_issuer", "COLLATERALIZED", self.currency, expiry, floatleg_basis, floatleg, fixedleg
#         )
#         return basis_swap

#     def get_deposit_spec(self):  # TODO
#         """
#         Specification for deposits
#         """

#         # get spot date
#         spot_date = get_end_date(self.refDate, self.spotLag)
#         # end date of the accrual period
#         end_date = get_end_date(spot_date, self.maturity)

#         # start date of FRA is endDate - tenor
#         start_date = get_start_date(end_date, self.tenor)

#         # specification of the deposit
#         deposit = pyvacon.finance.specification.DepositSpecification(
#             self.label, "dummy_issuer", "NONE", self.currency, self.refDate, start_date, end_date, 100, self.parRate, self.floatDayCount
#         )
#         return deposit

#     def get_fra_spec(self):  # TODO
#         """
#         Specification for FRAs/Futures
#         """
#         # get spot date
#         spot_date = get_end_date(self.refDate, self.spotLag)

#         # end date of the accrual period
#         end_date = get_end_date(spot_date, self.maturity)

#         # start date of FRA is endDate - tenor
#         start_date = get_start_date(end_date, self.tenor)

#         # expiry of FRA is the fixing date
#         expiry_date = get_start_date(start_date, self.spotLag)

#         # specification of the deposit
#         fra = pyvacon.finance.specification.InterestRateFutureSpecification(
#             self.label, "dummy_issuer", "NONE", self.currency, "dummy_udlId", expiry_date, 100, start_date, end_date, self.floatDayCount
#         )

#         return fra

#     def get_float_leg(self, pay_freq, reset_freq, roll_conv, spot_lag="0D"):  # TODO

#         # get swap leg schedule
#         flt_schedule = get_schedule(self.refDate, self.maturity, pay_freq, roll_conv, self.holidays, spot_lag)

#         # get start dates
#         flt_start_dates = flt_schedule[:-1]

#         # get end dates
#         flt_end_dates = flt_schedule[1:]
#         flt_pay_dates = flt_end_dates

#         # get reset dates
#         flt_reset_schedule = get_schedule(self.refDate, self.maturity, reset_freq, roll_conv, self.holidays, spot_lag)
#         flt_reset_dates = flt_reset_schedule[:-1]

#         flt_notionals = [1.0 for _ in range(len(flt_start_dates))]
#         floatleg = pyvacon.finance.specification.IrFloatLegSpecification(
#             flt_notionals, flt_reset_dates, flt_start_dates, flt_end_dates, flt_pay_dates, self.currency, "dummy_undrl", self.floatDayCount, 0.0
#         )
#         return floatleg

#     def get_fix_leg(self, pay_freq, roll_conv, spot_lag="0D"):  # TODO
#         # get fix leg schedule
#         fix_schedule = get_schedule(self.refDate, self.maturity, pay_freq, roll_conv, self.holidays, spot_lag)
#         fix_start_dates = fix_schedule[:-1]
#         fix_end_dates = fix_schedule[1:]
#         fix_pay_dates = fix_end_dates
#         fix_notionals = [1.0 for _ in range(len(fix_start_dates))]
#         fixedleg = pyvacon.finance.specification.IrFixedLegSpecification(
#             self.parRate, fix_notionals, fix_start_dates, fix_end_dates, fix_pay_dates, self.currency, self.fixDayCount
#         )
#         return fixedleg


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

    # Sanity checks:
    assert len(instruments) == len(quotes), "Number of quotes does not equal number of instruments."
    # TODO implement more input qualit checks:
    # curves given of correct type that match instrument type - or will this be done in the "market container" class?

    if curves == None:
        curves = {}
        print("* curves dictionary is empty, will bootstrap single discount curve")
    else:
        print("* curves dictionary provided, will bootstrap forward curve")

    #############################################################
    # initialize: # alternatively..
    yc_dates = [ref_date]
    dfs = [1.0]
    if isinstance(day_count_convention, str):  # normalizes type
        day_count_convention = DayCounterType(day_count_convention)

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
    # base curve creatiion check #TODO think about improving how to handle input curves if given for multicurve bootstrapping
    # given instrument types, check for required curves
    ins_types = []
    flag_irs_bootstrapped_as_fwd = False
    for inst in instruments:
        ins_type = inst.ins_type()
        if ins_type not in ins_types:
            ins_types.append(ins_type)

    if "discount_curve" not in curves:
        flag_multi_curve = False
        curves["discount_curve"] = DiscountCurve(
            "dummy_id_discount", ref_date, yc_dates, dfs, interpolation_type, extrapolation_type, day_count_convention
        )
        # this means this is the target output curve
    else:  # TODO #This means the discount curve was given. We thereforer want to output a FORWARD  curve, e.g. 3M, 6M,...
        flag_multi_curve = True

    # cannot multicurve bootstrap with deposits involved

    if Instrument.DEPOSIT in ins_types and flag_multi_curve == True:
        raise Exception("Deposits cannot be used in multicurve bootstrapping")

    if Instrument.IRS in ins_types:
        # check if curves has a fixing curve
        if "fixing_curve" in curves:
            if not isinstance(curves["fixing_curve"], DiscountCurve):
                raise Exception("Fixing curve is not of type DiscountCurve")

        else:
            print("IRS swap present but no fixing curve provided, will use bootstrapped curve in place")
            flag_irs_bootstrapped_as_fwd = True
            if flag_multi_curve:
                curves["fixing_curve"] = DiscountCurve(
                    "dummy_id_fixing", ref_date, yc_dates, dfs, interpolation_type, extrapolation_type, day_count_convention
                )
            else:
                curves["fixing_curve"] = curves["discount_curve"]

    #############################################################
    # # start with loglinear interpolation to obtain good initial values for all dates #TODO make logliner interpolator
    # this means i have to pass into the rror function the interpolation types desired which is different
    # from the inter and extra type we want for the final discount curve
    # bootstrap loop over ordered expiry dates which is also sorted here

    lower = 1.0e-5  # DEBUG TODO REMOVE if not implement bracket search
    upper = 5.0

    for end_date in sorted(instruments_by_date):
        quote, inst = instruments_by_date[end_date]  # use the market quote to compare with brentq
        yc_dates.append(end_date)  # next date
        dfs.append(dfs[-1])  # append a dummy value for the next date

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

            solution = brentq(error_fn, lower, upper, ARGS, xtol=1e-6)
            dfs[-1] = solution

            # TODO if clause here to say which curve is being updated....
            # curves dict needs to be updated before final interation check ...
            # if flag_irs_bootstrapped_as_fwd == True:  # meaning the passed forward curve needs to be updating alongside the discount curve
            #     curves["discount_curve"] = DiscountCurve(
            #         "bootstrappedYC", ref_date, yc_dates, dfs, interpolation_type, extrapolation_type, day_count_convention
            #     )
            #     curves["fixing_curve"] = curves["discount_curve"]

        except Exception as e:
            raise Exception(f"Initial bootstrap failed at {end_date}: {str(e)}")

    # In principle, this will have produced a curve.

    #############################################################
    # Iterative refinement with real interpolator
    # this is to improve the values for the whole curve
    # check for convergence: max change in zero rate estimate must be below tolerance.
    # max_diff = float("inf")
    max_diff = 0.0
    iteration = 0
    while iteration < max_iterations and (max_diff > tolerance or iteration == 0):

        total_evals = 0  # ??number of attempts?

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

            except Exception as e:
                raise Exception(f"Refinement failed at {end_date}: {str(e)}")

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

            # Multi-curve logic possible logic and single curve
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

    #TODO think about how to better implement in the case where forward curve is the same as discount curve

    #TODO what to do in case discount curve is GIVEN, i.e. in multicurve bootstrapping

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

    # #In single curve this is fine...
    # curves_copy["discount_curve"] = yc
    # if flag_irs_bootstrapped_as_fwd:
    #     curves_copy["fixing_curve"] = yc

    # Multi-curve logic possible logic and ssingle curve
    if flag_multi_curve:
        # This is a forward curve — use Given discount curve for discounting
        curves_copy["fixing_curve"] = yc
        # Keep discount_curve unchanged
    else:
        # Single-curve: updating discount curve itself
        curves_copy["discount_curve"] = yc
        if flag_irs_bootstrapped_as_fwd:  # if it is an irs instrument that needs the forward curve as well
            curves_copy["fixing_curve"] = yc

    calc_quote = get_quote(ref_date, instrument_spec, curves_copy)

    # DEBUG statement
    # print("----------------")
    # print("Error function trial curve")
    # print(yc.get_df())
    # print("----------------")
    # print(f"using {df_val} -> calc_quote: {calc_quote} - ref_quote: {ref_quote} = {calc_quote - ref_quote}")
    return calc_quote - ref_quote


def find_bracket(error_fn, initial_guess, *args):
    """Optional function to help find an applicable upper and lower bound
    for the brentq solver to ensure a sign change across the given error function
    applied over the boundary limits

    Args:
        error_fn (function): Error function
        initial_guess (float: initial guess of the correct result from which to find the boundary limits
    Raises:
        RuntimeError: _description_

    Returns:
       floats: lower and upper bound
    """
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


def get_quote(
    ref_date: _Union[date, datetime],
    instrument_spec: _Union[DepositSpecification, ForwardRateAgreementSpecification, InterestRateSwapSpecification],
    curve_dict: dict,
):
    """Get the instrument specific fair quote calculation result to be used in the bootstrapper.

    Args:
        ref_date (_Union[date, datetime]): _description_
        instrument_spec (_Union[DepositSpecification, ForwardRateAgreementSpecification, InterestRateSwapSpecification]): _description_
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

        yc_discount = curve_dict["discount_curve"]  # TODO decide how to pass which curves
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

    elif instrument_spec.ins_type() == Instrument.TBS:  # tenor basis swap
        # 	return InterestRateSwapPricer::computeBasisSpread(
        # refDate, ycDiscount, ycFwdReceive, ycFwdPay,
        # basisSwap->getReceiveLeg(), basisSwap->getPayLeg(), basisSwap->getSpreadLeg(),
        # std::make_shared<const FixingTable>(),
        # std::make_shared<const InterestRateSwapPricingParameter>()
        pass
    elif instrument_spec.ins_type() == Instrument.FXF:  # fx forward
        pass

    # # DEBUG TODO REMOVE
    # print(f"Calculated quote for {instrument_spec.ins_type()} is {quote}")
    return quote


def bootstrap_curve_from_quote_table(input_data):
    pass


# Main

if __name__ == "__main__":
    pass
