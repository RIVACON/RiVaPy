# 2025.07.24 Hans Nguyen
from datetime import datetime, date
from scipy.optimize import brentq
from rivapy.tools.interfaces import BaseDatedCurve
from rivapy.instruments.specifications import HasExpectedCashflows
from rivapy.marketdata import DiscountCurveParametrized, ConstantRate, DiscountCurve
from rivapy.pricing.pricing_request import PricingRequest
from rivapy.pricing._logger import logger
from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.instruments.fra_specifications import ForwardRateAgreementSpecification
from rivapy.instruments.ir_swap_specification import (
    IrFixedLegSpecification,
    IrFloatLegSpecification,
    IrOISLegSpecification,
    InterestRateSwapSpecification,
    IrSwapLegSpecification,
)

# from rivapy.pricing.pricing_data import InterestRateSwapPricingData, InterestRateSwapLegPricingData, InterestRateSwapFloatLegPricingData
from rivapy.pricing.pricing_data import (
    InterestRateSwapPricingData_rivapy,
    InterestRateSwapLegPricingData_rivapy,
    InterestRateSwapFloatLegPricingData_rivapy,
)
from rivapy.pricing.pricing_request import InterestRateSwapPricingRequest
from typing import List as _List, Union as _Union, Tuple, Dict, Any
from rivapy.tools.datetools import DayCounter
from rivapy.instruments.cashflow import CashFlow
from rivapy.marketdata.fixing_table import FixingTable
from rivapy.instruments.notional_structure import *
import numpy as np


from rivapy.tools.enums import IrLegType
from rivapy.tools._validators import print_member_values


class InterestRateSwapPricer:

    def __init__(
        self,
        val_date: _Union[date, datetime],
        spec: InterestRateSwapSpecification,
        discount_curve_pay_leg: DiscountCurve,
        discount_curve_receive_leg: DiscountCurve,
        fixing_curve_pay_leg: DiscountCurve,
        fixing_curve_receive_leg: DiscountCurve,
        fx_fwd_curve_pay_leg: DiscountCurve,  # TODO FxForwardCurve ... do we need anotheer class
        fx_fwd_curve_receive_leg: DiscountCurve,
        pricing_request: InterestRateSwapPricingRequest,
        pricing_param: Dict = None,  # mutable argument should be defaulted to none
        fixing_map: FixingTable = None,
        fx_pay_leg: float = 1.0,
        fx_receive_leg: float = 1.0,
    ):
        """Initializes the Interest Rate Swap Pricer with all required curves, specifications, and parameters.

        Args:
            val_date (date | datetime): The valuation date for pricing the swap. This is the anchor date for all time-dependent calculations.
            spec (InterestRateSwapSpecification): The swap's structural details (legs, notionals, schedules, etc.).
            discount_curve_pay_leg (DiscountCurve): Discount curve used to present value the pay leg.
            discount_curve_receive_leg (DiscountCurve): Discount curve used to present value the receive leg.
            fixing_curve_pay_leg (DiscountCurve): Curve used to forecast forward rates for the pay leg (typically for floating legs).
            fixing_curve_receive_leg (DiscountCurve): Curve used to forecast forward rates for the receive leg.
            fx_fwd_curve_pay_leg (DiscountCurve): FX forward curve to convert the pay leg currency to the pricing currency (if applicable).
            fx_fwd_curve_receive_leg (DiscountCurve): FX forward curve to convert the receive leg currency to the pricing currency.
            pricing_request (InterestRateSwapPricingRequest): Contains the pricing type, metrics requested (e.g., PV), and other flags. Not yet used properly
            pricing_param (Dict, optional): Additional pricing parameters, such as day count conventions, compounding rules, etc.
            fixing_map (FixingTable, optional): Historical fixings for floating legs that reference past periods.
            fx_pay_leg (float, optional): FX rate multiplier to convert the pay leg currency to base. Default is 1.0 (i.e., same currency).
            fx_receive_leg (float, optional): FX rate multiplier to convert the receive leg currency to base. Default is 1.0.

        """

        self._val_date = val_date
        self._spec = spec
        self._pay_leg = spec.pay_leg
        self._receive_leg = spec.receive_leg

        self._discount_curve_pay_leg = discount_curve_pay_leg
        self._discount_curve_receive_leg = discount_curve_receive_leg

        self._fixing_curve_pay_leg = fixing_curve_pay_leg  # const std::shared_ptr<const DiscountCurve>& fixingCurvePayLeg,
        self._fixing_curve_receive_leg = fixing_curve_receive_leg  # const std::shared_ptr<const DiscountCurve>& fixingCurveReceiveLeg,

        self._fx_fwd_curve_pay_leg = fx_fwd_curve_pay_leg  # const std::shared_ptr<const FxForwardCurve>& fxForwardCurvePayLeg,
        self._fx_fwd_curve_receive_leg = fx_fwd_curve_receive_leg  # const std::shared_ptr<const FxForwardCurve>& fxForwardCurveReceiveLeg,

        self._pricing_request = pricing_request  # const PricingRequest& pricingRequest,
        if pricing_param is None:
            pricing_param = {}
        self._pricing_param = pricing_param  # std::shared_ptr<const InterestRateSwapPricingParameter> pricingParam,
        self._fixing_map = fixing_map  # std::shared_ptr<const FixingTable> fixingMap,

        self._fx_pay_leg = fx_pay_leg  # double fxPayLeg,
        self._fx_receive_leg = fx_receive_leg  # double fxReceiveLeg)

        self._pricing_param = pricing_param

    @staticmethod
    def _populate_cashflows_fix(
        val_date: _Union[date, datetime],
        fixed_leg_spec: IrFixedLegSpecification,
        discount_curve: DiscountCurve,
        # forward_curve: DiscountCurve,
        fx_forward_curve: DiscountCurve,
        fixing_map: FixingTable,
        set_rate: bool = False,
        desired_rate: float = 1.0,
    ) -> _List[CashFlow]:
        """Generate a list of CashFlow objects, each with the cashflow amount for the given accrual period
        and additional information added to descrive the cashflow. For the fixed leg of a swap.

        Args:
            val_date (_Union[date, datetime]): valuation date
            fixed_leg_spec (IrFixedLegSpecification): specification of the fixed leg of the IR Swap
            discount_curve (DiscountCurve): The discount curve used for discounting to calculated the present value
            fx_forward_curve (DiscountCurve): fxCurve used for currency conversion for applicable swap (not yet implemented)
            fixing_map (FixingTable): Fixing map of historial values (not yet implemented)
            fixing_grace_period (float):
            set_rate (bool, optional): Flag to manually set a Fixed rate. Defaults to False.
            spread (float, optional): Desired rate value. Defaults to 1 (which then calculates annuity).

        Returns:
            _List[CashFlow]: list of CashFlow objects.
        """

        # overwrite fixed interest if desired
        fixed_rate = fixed_leg_spec.fixed_rate
        if set_rate:
            fixed_rate = desired_rate  # if desired_rate = 1.0, this is to calculate the annuity, for the calculation of fair swap rate.

        # init output storage object
        entries = []
        dcc = DayCounter(discount_curve.daycounter)

        # get projected notionals
        leg_notional_structure = fixed_leg_spec.get_NotionalStructure()
        # define number of cashflows
        num_of_start_dates = len(fixed_leg_spec.start_dates)

        notionals = get_projected_notionals(
            val_date=val_date,
            notional_structure=leg_notional_structure,
            start_period=0,
            end_period=num_of_start_dates,
            fx_forward_curve=fx_forward_curve,
            fixing_map=fixing_map,
        )  # output is a lsit of floats

        for i in range(len(notionals)):

            notional_start_date = leg_notional_structure.get_pay_date_start(i)
            notional_end_date = leg_notional_structure.get_pay_date_end(i)

            if notional_start_date:  # i.e. not None or empty
                # add an intional notional OUTFLOW or not
                notional_entry = CashFlow()
                notional_entry.pay_date = notional_start_date

                if val_date <= notional_entry.pay_date:  # TODO recheck this business logic
                    notional_entry.discount_factor = discount_curve.rivapy_value(val_date, notional_entry.pay_date)
                else:
                    notional_entry.discount_factor = 0.0

                notional_entry.pay_amount = -1 * notionals[i]
                notional_entry.present_value = notional_entry.pay_amount * notional_entry.discount_factor
                notional_entry.notional_cashflow = True
                entries.append(notional_entry)

            entry = CashFlow()
            entry.start_date = fixed_leg_spec.start_dates[i]
            entry.end_date = fixed_leg_spec.end_dates[i]
            entry.pay_date = fixed_leg_spec.pay_dates[i]
            entry.notional = notionals[i]
            entry.rate = fixed_rate
            entry.interest_yf = dcc.yf(entry.start_date, entry.end_date)  # gives back SINGLE yearfraction
            if val_date < entry.pay_date:
                entry.discount_factor = discount_curve.rivapy_value(val_date, entry.pay_date)
            else:
                entry.discount_factor = 0.0
            entry.interest_amount = entry.notional * entry.rate * entry.interest_yf
            entry.pay_amount = entry.interest_amount
            entry.present_value = entry.pay_amount * entry.discount_factor

            # setting main cashflow value
            entry.val = entry.pay_amount
            entry.interest_cashflow = True
            entries.append(entry)

            # # TEMPORARY TEST - inthe case of constant notional structure, but i want a final notional cashflow like a bond
            # if i == len(notionals) - 1:  # this checks for the last entry
            #     notional_end_date = entry.end_date

            if notional_end_date:
                # add an intional notional INFLOW or not
                notional_entry = CashFlow()
                notional_entry.pay_date = notional_end_date

                if val_date <= notional_entry.pay_date:  # TODO recheck this business logic
                    notional_entry.discount_factor = discount_curve.rivapy_value(val_date, notional_entry.pay_date)
                else:
                    notional_entry.discount_factor = 0.0

                notional_entry.pay_amount = notionals[i]  # positive
                notional_entry.present_value = notional_entry.pay_amount * notional_entry.discount_factor
                notional_entry.notional_cashflow = True
                entries.append(notional_entry)

        return entries  # a LIST of ENTRY objects, where each object has the PV

    @staticmethod
    def _populate_cashflows_float(
        val_date: _Union[date, datetime],
        float_leg_spec: IrFloatLegSpecification,
        discount_curve: DiscountCurve,
        forward_curve: DiscountCurve,
        fx_forward_curve: DiscountCurve,
        fixing_map: FixingTable,
        fixing_grace_period: float,
        set_spread: bool = False,
        spread: float = None,
    ) -> _List[CashFlow]:
        """Generate a list of CashFlow objects, each with the cashflow amount for the given accrual period
        and additional information added to descrive the cashflow.

        Args:
            val_date (_Union[date, datetime]): valuation date
            float_leg_spec (IrFloatLegSpecification): specification of the floating leg of the IR Swap
            discount_curve (DiscountCurve): The discount curve used for discounting to calculated the present value
            forward_curve (DiscountCurve): forward curve used to determine the forward rate to calculate the interest
            fx_forward_curve (DiscountCurve): fxCurve used for currency conversion for applicable swap (not yet implemented)
            fixing_map (FixingTable): Fixing map of historial values (not yet implemented)
            fixing_grace_period (float):
            setSpread (bool, optional): Flag to manually set a spread value. Defaults to False.
            spread (float, optional): Desired spread value. Defaults to None.

        Raises:
            ValueError: No underlying index ID found in fixing table

        Returns:
            _List[CashFlow]: list of CashFlow objects.
        """
        entries = []
        udl = float_leg_spec.udl_id

        # swap day count convention
        dcc = DayCounter(discount_curve.daycounter)
        # rate day count convention # note that the specification also has dcc but without the curve...
        rate_dcc = DayCounter(float_leg_spec.rate_day_count_convention)

        # overwrite spread if desired
        leg_spread = float_leg_spec.spread
        if set_spread:
            leg_spread = spread

        # obtain the notionals and create a list of notionals to be used for each cashflow
        leg_notional_structure = float_leg_spec.get_NotionalStructure()
        # define the number of cashflows
        num_of_start_dates = len(float_leg_spec.start_dates)

        notionals = get_projected_notionals(
            val_date=val_date,
            notional_structure=leg_notional_structure,
            start_period=0,
            end_period=num_of_start_dates,
            fx_forward_curve=fx_forward_curve,
            fixing_map=fixing_map,
        )  # output is a list of floats

        for i in range(len(notionals)):

            notional_start_date = leg_notional_structure.get_pay_date_start(i)
            notional_end_date = leg_notional_structure.get_pay_date_end(i)

            if notional_start_date:  # i.e. not None or empty
                # add an initial notional OUTFLOW or not
                notional_entry = CashFlow()
                notional_entry.pay_date = notional_start_date

                if val_date <= notional_entry.pay_date:  # TODO recheck this business logic
                    notional_entry.discount_factor = discount_curve.rivapy_value(val_date, notional_entry.pay_date)
                else:
                    notional_entry.discount_factor = 0.0

                notional_entry.pay_amount = -1 * notionals[i]
                notional_entry.present_value = notional_entry.pay_amount * notional_entry.discount_factor
                notional_entry.notional_cashflow = True
                entries.append(notional_entry)

            entry = CashFlow()
            entry.start_date = float_leg_spec.start_dates[i]
            entry.end_date = float_leg_spec.end_dates[i]
            entry.pay_date = float_leg_spec.pay_dates[i]
            entry.notional = notionals[i]
            entry.interest_yf = dcc.yf(entry.start_date, entry.end_date)
            rate_yf = rate_dcc.yf(float_leg_spec.rate_start_dates[i], float_leg_spec.rate_end_dates[i])
            # # DEBUG TODO REMOVE
            # print(f"notional {i}:{notionals[i]} for {entry.end_date}")
            if val_date <= float_leg_spec.reset_dates[i]:
                # print("swap calculating fwd_rate for floating leg")  # DEBUG TODO REMOVE
                fwd_rate = forward_curve.rivapy_valueFWD(val_date, float_leg_spec.rate_start_dates[i], float_leg_spec.rate_end_dates[i])
                entry.rate = leg_spread + (1.0 / fwd_rate - 1.0) / rate_yf

            else:
                fixing = fixing_map.get_fixing(udl, float_leg_spec.reset_dates[i])  # TODO IMPLEMENT FIXING TABLE CLASS
                if fixing is None:  # i.e. no fixing available

                    if val_date - float_leg_spec.reset_dates[i] > fixing_grace_period:
                        raise ValueError(f"Missing fixing for {udl} on {float_leg_spec.reset_dates[i]}")

                    else:
                        # fix value of payment i in future based on current discount curve and a period between
                        # valDate and valDate+length of original period (workaround if fixing is not available)
                        # taken from pyvacon
                        if entry.pay_date >= val_date:  # TODO understand the logic
                            time_delta = entry.end_date - entry.start_date
                            fixing = (1.0 / forward_curve.rivapy_valueFWD(val_date, val_date, val_date + time_delta) - 1) / entry.interest_yf

                entry.rate = fixing + spread

            if val_date <= entry.pay_date:
                entry.discount_factor = discount_curve.rivapy_value(val_date, entry.pay_date)
            else:
                entry.discount_factor = 0.0

            # given rate, notional, and yf, calc interest
            entry.interest_amount = entry.notional * entry.rate * entry.interest_yf

            # scale by forward rate???? #TODO
            if val_date <= entry.end_date:
                entry.pay_amount = entry.interest_amount / forward_curve.rivapy_valueFWD(val_date, entry.end_date, entry.pay_date)
            else:
                if entry.pay_date >= val_date:
                    entry.pay_amount = entry.interest_amount / forward_curve.rivapy_valueFWD(val_date, val_date, entry.pay_date)
                else:
                    entry.pay_amount = 0.0

            # given total cashflow amount - discount it
            # #DEBUG TODO
            # print(f"forward rate: {entry.rate}")
            # print(f"delta_t: {entry.interest_yf}")
            # print(f"discount_factor: {entry.discount_factor}")
            entry.present_value = entry.pay_amount * entry.discount_factor
            entry.interest_cashflow = True
            entries.append(entry)

            # # TEMPORARY TEST - inthe case of constant notional structure, but i want a final notional cashflow like a bond
            # if i == len(notionals) - 1:  # this checks for the last entry
            #     notional_end_date = entry.end_date

            if notional_end_date:
                # add an intional notional INFLOW or not
                notional_entry = CashFlow()
                notional_entry.pay_date = notional_end_date

                if val_date <= notional_entry.pay_date:  # TODO recheck this business logic
                    notional_entry.discount_factor = discount_curve.rivapy_value(val_date, notional_entry.pay_date)
                else:
                    notional_entry.discount_factor = 0.0

                notional_entry.pay_amount = notionals[i]  # positive
                notional_entry.present_value = notional_entry.pay_amount * notional_entry.discount_factor
                notional_entry.notional_cashflow = True
                entries.append(notional_entry)

        return entries

    @staticmethod
    def _populate_cashflows_ois(
        val_date: _Union[date, datetime],
        ois_leg_spec: IrOISLegSpecification,
        discount_curve: DiscountCurve,
        forward_curve: DiscountCurve,
        fx_forward_curve: DiscountCurve,
        fixing_map: FixingTable,
        fixing_grace_period: float,
        set_spread: bool = False,
        spread: float = None,
    ) -> _List[CashFlow]:
        """Generate a list of CashFlow objects, each with the cashflow amount for the given accrual period
        and additional information added to describe the cashflow. To be used for OIS leg specifications.

        Args:
            val_date (_Union[date, datetime]): valuation date
            float_leg_spec (IrFloatLegSpecification): specification of the floating leg of the IR Swap
            discount_curve (DiscountCurve): The discount curve used for discounting to calculated the present value
            forward_curve (DiscountCurve): forward curve used to determine the forward rate to calculate the interest
            fx_forward_curve (DiscountCurve): fxCurve used for currency conversion for applicable swap (not yet implemented)
            fixing_map (FixingTable): Fixing map of historial values (not yet implemented)
            fixing_grace_period (float):
            setSpread (bool, optional): Flag to manually set a spread value. Defaults to False.
            spread (float, optional): Desired spread value. Defaults to None.

        Raises:
            ValueError: No underlying index ID found in fixing table

        Returns:
            _List[CashFlow]: list of CashFlow objects.
        """
        entries = []  # output container to be returned
        udl = ois_leg_spec.udl_id  # get the ID of the underlying

        # Parameters to consider
        # const std::vector<std::vector<boost::posix_time::ptime>>& dailyRateStartDates = oisLeg->getDailyRateStartDates();
        # const std::vector<std::vector<boost::posix_time::ptime>>& dailyRateEndDates = oisLeg->getDailyRateEndDates();
        # const std::vector<std::vector<boost::posix_time::ptime>>& dailyResetDates = oisLeg->getDailyResetDates();
        # const std::vector<boost::posix_time::ptime>& startDates = oisLeg->getStartDates();
        # const std::vector<boost::posix_time::ptime>& endDates = oisLeg->getEndDates();
        # const std::vector<boost::posix_time::ptime>& payDates = oisLeg->getPayDates();
        # std::shared_ptr<const NotionalStructure> notionalStructure = leg->getNotionalStructure();
        # std::vector<double> notionals(leg->getStartDates().size());

        # swap day count convention
        dcc = DayCounter(discount_curve.daycounter)
        # rate day count convention # note that the specification also has dcc but without the curve...
        rate_dcc = DayCounter(ois_leg_spec.rate_day_count_convention)

        # overwrite spread if desired
        leg_spread = ois_leg_spec.spread
        if set_spread:
            leg_spread = spread

        # obtain the notionals and create a list of notionals to be used for each cashflow
        leg_notional_structure = ois_leg_spec.get_NotionalStructure()
        # define the number of cashflows
        num_of_start_dates = len(ois_leg_spec.start_dates)

        notionals = get_projected_notionals(
            val_date=val_date,
            notional_structure=leg_notional_structure,
            start_period=0,
            end_period=num_of_start_dates,
            fx_forward_curve=fx_forward_curve,
            fixing_map=fixing_map,
        )  # output is a list of floats

        # obtained from ois_leg_spec # expect 2D structure
        daily_rate_start_dates = ois_leg_spec.rate_start_dates
        # [[0],[1]] # i:start datet fo accrual period j:rate start days within the accrual period # e.g i:accrual over 3M, j:reset daily
        daily_rate_end_dates = ois_leg_spec.rate_end_dates
        daily_rate_reset_dates = ois_leg_spec.rate_reset_dates

        # for each notional, but essentially each accrual period
        for i in range(len(notionals)):

            notional_start_date = leg_notional_structure.get_pay_date_start(i)
            notional_end_date = leg_notional_structure.get_pay_date_end(i)

            if notional_start_date:  # i.e. not None or empty
                # add an initial notional OUTFLOW or not
                notional_entry = CashFlow()
                notional_entry.pay_date = notional_start_date

                if val_date <= notional_entry.pay_date:  # TODO recheck this business logic
                    notional_entry.discount_factor = discount_curve.rivapy_value(val_date, notional_entry.pay_date)
                else:
                    notional_entry.discount_factor = 0.0

                notional_entry.pay_amount = -1 * notionals[i]
                notional_entry.present_value = notional_entry.pay_amount * notional_entry.discount_factor
                notional_entry.notional_cashflow = True
                entries.append(notional_entry)

            entry = CashFlow()
            entry.start_date = ois_leg_spec.start_dates[i]
            entry.end_date = ois_leg_spec.end_dates[i]
            entry.pay_date = ois_leg_spec.pay_dates[i]
            entry.notional = notionals[i]
            entry.interest_yf = dcc.yf(entry.start_date, entry.end_date)

            # loop over each resetting datet, i.e. daily to calculate the daily compounded interest to use to calculae coupon for this time period/entry
            accDf = 1.0  # accrual discount factor

            for j in range(len(daily_rate_start_dates[i])):

                rate_yf = rate_dcc.yf(daily_rate_start_dates[i][j], daily_rate_end_dates[i][j])  # should be a day

                if daily_rate_reset_dates[i][j] >= val_date:  # rate not yet fixed
                    daily_fwd = forward_curve.rivapy_valueFWD(val_date, daily_rate_start_dates[i][j], daily_rate_end_dates[i][j])
                    daily_rate = leg_spread + (1.0 / daily_fwd - 1.0) / rate_yf

                else:  # rate already fixed

                    fixing = fixing_map.get_fixing(udl, daily_rate_reset_dates[i][j])

                    if np.isnan(fixing):  # math.isnan
                        if (val_date - daily_rate_reset_dates[i][j]) > fixing_grace_period:
                            raise RuntimeError(f"Fixing for udl {udl}, date {daily_rate_reset_dates[i][j]} not provided")
                        else:
                            if entry.pay_date >= val_date:
                                # fix value of payment i in future based on current discount curve and a period between valDate and valDate+length of original period (workaround if fixing is not available)
                                time_delta = entry.end_date - entry.start_date  # do we need? we assume daily...
                                fixing = (
                                    1.0 / forward_curve.rivapy_valueFWD(val_date, daily_rate_start_dates[i][j], daily_rate_end_dates[i][j]) - 1.0
                                ) / rate_yf

                    daily_rate = leg_spread + fixing

                accDf *= 1.0 + daily_rate * rate_yf  # compounded daily, hence the multiplication
            # accDf now calulated

            rate_yf = rate_dcc.yf(daily_rate_start_dates[i][0], daily_rate_end_dates[i][-1])  # total accrual period Yf
            entry.rate = (accDf - 1.0) / rate_yf  # the -1 gives then just the interest portion of the compounded daily

            if val_date <= entry.pay_date:
                entry.discount_factor = discount_curve.rivapy_value(val_date, entry.pay_date)
            else:
                entry.discount_factor = 0.0

            # given rate, notional, and yf, calc interest
            entry.interest_amount = entry.notional * entry.rate * entry.interest_yf
            entry.pay_amount = entry.interest_amount

            # # scale by forward rate???? #TODO
            # if val_date <= entry.end_date:
            #     entry.pay_amount = entry.interest_amount / forward_curve.rivapy_valueFWD(val_date, entry.end_date, entry.pay_date)
            # else:
            #     if entry.pay_date >= val_date:
            #         entry.pay_amount = entry.interest_amount / forward_curve.rivapy_valueFWD(val_date, val_date, entry.pay_date)
            #     else:
            #         entry.pay_amount = 0.0

            # given total cashflow amount - discount it
            # #DEBUG TODO
            # print(f"forward rate: {entry.rate}")
            # print(f"delta_t: {entry.interest_yf}")
            # print(f"discount_factor: {entry.discount_factor}")
            entry.present_value = entry.pay_amount * entry.discount_factor
            entry.interest_cashflow = True
            entries.append(entry)

            # # TEMPORARY TEST - inthe case of constant notional structure, but i want a final notional cashflow like a bond
            # if i == len(notionals) - 1:  # this checks for the last entry
            #     notional_end_date = entry.end_date

            if notional_end_date:
                # add an intional notional INFLOW or not
                notional_entry = CashFlow()
                notional_entry.pay_date = notional_end_date

                if val_date <= notional_entry.pay_date:  # TODO recheck this business logic
                    notional_entry.discount_factor = discount_curve.rivapy_value(val_date, notional_entry.pay_date)
                else:
                    notional_entry.discount_factor = 0.0

                notional_entry.pay_amount = notionals[i]  # positive
                notional_entry.present_value = notional_entry.pay_amount * notional_entry.discount_factor
                notional_entry.notional_cashflow = True
                entries.append(notional_entry)

        return entries

    @staticmethod
    def price_leg_pricing_data(val_date, pricing_data: InterestRateSwapLegPricingData_rivapy, param):
        """Pricing a single Leg using Pricing Data architecture (not yet fully integrated, to be done once architecture clarified)

        Args:
            val_date (_type_): _description_
            pricing_data (InterestRateSwapLegPricingData): float leg or base leg pricing data ...
            param (_type_): extra parameters (not yet used)

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # get leg info from PricingData ->spec
        leg_spec = pricing_data.spec  # what kind of leg?

        if leg_spec.leg_type == IrLegType.FIXED:
            if pricing_data.desired_rate is not None:
                set_rate = True
            else:
                set_rate = False
            cashflow_table = InterestRateSwapPricer._populate_cashflows_fix(
                val_date,
                pricing_data.spec,
                pricing_data.discount_curve,
                pricing_data.forward_curve,
                pricing_data.fixing_map,
                set_rate=set_rate,
                desired_rate=pricing_data.desired_rate,
            )
        elif leg_spec.leg_type == IrLegType.FLOAT:
            if pricing_data.spread is not None:
                set_spread = True
            else:
                set_spread = False

            # REMOVE THIS TEST?
            if isinstance(pricing_data, InterestRateSwapFloatLegPricingData_rivapy):
                cashflow_table = InterestRateSwapPricer._populate_cashflows_float(
                    val_date,
                    pricing_data.spec,
                    pricing_data.discount_curve,
                    pricing_data.forward_curve,
                    pricing_data.fixing_curve,
                    pricing_data.fixing_map,
                    pricing_data.fixing_grace_period,
                    set_spread=set_spread,
                    spread=pricing_data.spread,
                )

            else:
                raise ValueError("pricing data is not of type 'InterestRateSwapFloatLegPricingData_rivapy' ")  # TODO UPDATE

        # elif leg_spec.leg_type ==  IrLegType.OIS:
        #    InterestRateSwapPricer._populate_cashflows_ois
        else:
            raise ValueError(f"Unknown leg type {leg_spec.type}")

        PV = 0
        for entry in cashflow_table:

            PV += entry.present_value

        # return PV
        return PV * pricing_data.fx_rate

    @staticmethod
    def price_leg(
        val_date,
        discount_curve: DiscountCurve,
        forward_curve: DiscountCurve,
        fxForward_curve: DiscountCurve,
        spec: _Union[IrFixedLegSpecification, IrFloatLegSpecification],
        fixing_map: FixingTable = None,
        fixing_grace_period: float = 0,
        pricing_params: dict = {"set_rate": False, "desired_rate": 1.0},
    ):
        """Price a leg of a IR swap.

        Args:
            val_date (_type_): Valuation date
            discount_curve (DiscountCurve): discount curve for discounting cashflows
            forward_curve (DiscountCurve): forward curve for the forward rate calculations
            fxForward_curve (DiscountCurve): fx forward curve for currency conversions for applicable instruments (not yet implemented)
            spec (_Union[IrFixedLegSpecification, IrFloatLegSpecification]): Specification object for the leg to be priced
            fixing_map (FixingTable, optional): Historical fixing data (not yet implemented). Defaults to None.
            fixing_grace_period (float, optional): . Defaults to 0.
            pricing_params (_type_, optional): Additional parameters needed for swap pricing. Defaults to {"set_rate": False, "desired_rate": 1.0}.

        Raises:
            ValueError: if unknown leg type is passed

        Returns:
            float: return the aggregated Present value of the  leg of the swap.
        """

        # get leg info from PricingData ->spec
        leg_spec = spec  # what kind of leg?

        if leg_spec.leg_type == IrLegType.FIXED:
            # print("generating cashflow table for FIXED leg")
            set_rate = pricing_params["set_rate"]  # if we want to set the rate
            desired_rate = pricing_params["desired_rate"]
            cashflow_table = InterestRateSwapPricer._populate_cashflows_fix(
                val_date, leg_spec, discount_curve, forward_curve, fixing_map, set_rate, desired_rate
            )  # TODO rethink framework of passing set_rate and desired rate
        elif leg_spec.leg_type == IrLegType.FLOAT:
            # print("generating cashflow table for FLOAT leg")
            cashflow_table = InterestRateSwapPricer._populate_cashflows_float(
                val_date, leg_spec, discount_curve, forward_curve, fxForward_curve, fixing_map, fixing_grace_period
            )  # TODO implement spread

        elif leg_spec.leg_type == IrLegType.OIS:
            cashflow_table = InterestRateSwapPricer._populate_cashflows_ois(
                val_date, leg_spec, discount_curve, forward_curve, fxForward_curve, fixing_map, fixing_grace_period
            )
            # TODO implement spread
        else:
            raise ValueError(f"Unknown leg type {leg_spec.type}")

        PV = 0
        # print("----------------------------------------------------------")
        # print("DEBUG: price leg pv values")  # DEBUG TODO REMOVE
        for entry in cashflow_table:
            # print_member_values(entry)
            # print(entry.present_value)
            PV += entry.present_value

        return PV
        # return PV* pricing_data.fx_rate

    def price(self):
        """price a full swap, with a pay leg and a receive leg"""

        # # PricingResults& results, -> implement also
        # val_date = self._val_date    # const  boost::posix_time::ptime& valDate,
        # discount_curve_pay_leg = self. # const std::shared_ptr<const DiscountCurve>& discountCurvePayLeg,
        # discount_curve_receive_leg= self. # const std::shared_ptr<const DiscountCurve>& discountCurveReceiveLeg,
        # fixing_curve_pay_leg = # const std::shared_ptr<const DiscountCurve>& fixingCurvePayLeg,
        # fixing_curve_receive_leg= # const std::shared_ptr<const DiscountCurve>& fixingCurveReceiveLeg,
        # fx_fwd_curve_pay_leg = # const std::shared_ptr<const FxForwardCurve>& fxForwardCurvePayLeg,
        # fx_fwd_curve_receive_leg = # const std::shared_ptr<const FxForwardCurve>& fxForwardCurveReceiveLeg,
        # interest_rate_swap_spec = # const std::shared_ptr<const InterestRateSwapSpecification>& spec,
        # pricing_request = # const PricingRequest& pricingRequest,
        # pricing_param = # std::shared_ptr<const InterestRateSwapPricingParameter> pricingParam,
        # fixing_map = # std::shared_ptr<const FixingTable> fixingMap,
        # fx_pay_leg = # double fxPayLeg,
        # fx_receive_leg = # double fxReceiveLeg)

        # unit in days
        fixing_grace_period = self._pricing_param["fixing_grace_period"]  # TODO - check when pricing params are properly implemented

        # TODO check structure again....
        # price = InterestRateSwapPricer.price_leg(valDate, discountCurveReceiveLeg, fixingCurveReceiveLeg, fxForwardCurveReceiveLeg, spec->getReceiveLeg(), fixingMap, fixingGracePeriod) * fxReceiveLeg;
        # price -= InterestRateSwapPricer.price_leg(valDate, discountCurvePayLeg, fixingCurvePayLeg, fxForwardCurvePayLeg, spec->getPayLeg(), fixingMap, fixingGracePeriod) * fxPayLeg;

        #
        aggregated_price = InterestRateSwapPricer.price_leg(
            self._val_date,
            discount_curve=self._discount_curve_receive_leg,
            forward_curve=self._fixing_curve_receive_leg,
            fxForward_curve=self._fx_fwd_curve_receive_leg,
            spec=self._receive_leg,
            fixing_map=self._fixing_map,
            fixing_grace_period=fixing_grace_period,
        )
        aggregated_price -= InterestRateSwapPricer.price_leg(
            self._val_date,
            discount_curve=self._discount_curve_pay_leg,
            forward_curve=self._fixing_curve_receive_leg,
            fxForward_curve=self._fx_fwd_curve_receive_leg,
            spec=self._pay_leg,
            fixing_map=self._fixing_map,
            fixing_grace_period=fixing_grace_period,
        )
        # results.setPrice(price);
        # aggregated_price is already discounted to present value inside the price_leg method
        return aggregated_price

    # static method also then?
    @staticmethod
    def compute_swap_rate(
        ref_date: _Union[date, datetime],
        discount_curve: DiscountCurve,
        fixing_curve: DiscountCurve,
        float_leg: IrFloatLegSpecification,
        fixed_leg: IrFixedLegSpecification,
        fixing_map: FixingTable = None,
        pricing_params: dict = {"fixing_grace_period": 0.0},
    ) -> float:
        """To calculate the fair swap rate, i.e. the fixed rate(r*) such that
                PV_fixed(r*) = PV_float
            where
                PV_fixed(r*) = r* * Annuity
            where
                Annuity = sum(notional_i * DF_i * YF_i*)
        Args:
            ref_date (_Union[date, datetime]): reference date
            discount_curve (DiscountCurve): discoutn curve to determine present value
            fixing_curve (DiscountCurve): curve used for fwd rates for the floating leg
            float_leg (IrFloatLegSpecification): IR swap float leg specification
            fixed_leg (IrFixedLegSpecification): IR swap fix leg specification
            fixing_map (FixingTable, optional): historical fixing data (TODO). Defaults to None.
            fixing_grace_period (int, optional): . Defaults to 0.

        Returns:
            float: fair rate for the swap
        """

        fixing_grace_period = pricing_params["fixing_grace_period"]

        float_leg_PV = InterestRateSwapPricer.price_leg(ref_date, discount_curve, fixing_curve, None, float_leg, fixing_map, fixing_grace_period)
        fixed_leg_annuity = InterestRateSwapPricer.price_leg(
            ref_date, discount_curve, fixing_curve, None, fixed_leg, fixing_map, fixing_grace_period, pricing_params
        )
        return float_leg_PV / fixed_leg_annuity

    # TODO
    def compute_swap_spread(self):
        # ref date
        # discount curve pay leg
        # forward curve pay leg
        # fx forward curve pay leg
        # discount curve rec leg
        # forward curve rec leg
        # fx forward curve rec leg
        # pay leg spec
        # rec leg spec
        # fixing map
        # extra param: InterestRateSwapPricingParameter
        # fx pay
        # fx rec
        # fixing grace period comes from the extra param

        # convert all prices into the currency of the swap
        pv_pay = 0
        pv_rec_s0 = 1
        py_rec_s1 = 0
        # pv_pay =  fxPay * price_leg(refDate, discountCurvePay, forwardCurvePay, fxForwardCurvePay, floatLegPay, fixingMap, fixingGracePeriod);
        # pv_rec_s0 = fxRec * price_leg(refDate, discountCurveRec, forwardCurveRec, fxForwardCurveRec, floatLegRec, fixingMap, fixingGracePeriod, true, 0.);set_spread=True, desired_spread = 0.0 #for float it is spread
        # py_rec_s1 = fxRec * price_leg(refDate, discountCurveRec, forwardCurveRec, fxForwardCurveRec, floatLegRec, fixingMap, fixingGracePeriod, true, 1.);set_spread=True, desired_spread = 1.0
        # note that the current price leg doesnt take spreads as options for the moment, it is left as a # TODO for now...
        return (pv_pay - pv_rec_s0) / (py_rec_s1 - pv_rec_s0)

    # TODO
    def compute_basis_spread(self):
        # ref date
        # discount curve
        # receiveLegFixingCurve
        # payLegFixingCurve
        # receiveLeg spec #floatIRspec
        # payLeg spec #floatIRspec
        # fixed leg spec # fixedIRspec
        # fixing grace period comes from the extra param

        # noFxFowardCruve, set to null

        receive_leg_PV = 1  # price_leg(refDate, discountCurve, receiveLegFixingCurve, nullptr, receiveLeg, fixingMap, fixingGracePeriod)
        pay_leg_PV = 1  # price_leg(refDate, discountCurve, payLegFixingCurve,     nullptr, payLeg, fixingMap, fixingGracePeriod);
        fixed_leg_PV01 = 1
        # price_leg(refDate, discountCurve, std::shared_ptr<const DiscountCurve>(), nullptr, fixedLeg, fixingMap, fixingGracePeriod, true, 1.);

        # # for fixed, we are setting the rate to 1

        return (receive_leg_PV - pay_leg_PV) / fixed_leg_PV01


#########################################################################
# FUNCTIONS
def get_projected_notionals(
    val_date: _Union[date, datetime],
    notional_structure: NotionalStructure,
    start_period: int,
    end_period: int,
    fx_forward_curve: DiscountCurve,
    fixing_map: FixingTable = None,
) -> _List[float]:
    """
    Generate a list with projected notionals, using FX forward curve if applicable, or fixing table(not yet implemented).

    Args:
        val_date (datetime): The valuation date.
        notional_structure (NotionalStructure): The notional structure class object.
        start_period (int): Start index of the period range.
        end_period (int): End index of the period range (exclusive).
        fx_forward_curve (FxForwardCurve): Required for resetting notionals.
        fixing_map (FixingTable): Not used (yet).
    """

    result = []

    # Check if this is a resetting notional structure
    if isinstance(notional_structure, ResettingNotionalStructure):
        if fx_forward_curve is None:
            raise ValueError("No FX forward curve provided for resetting leg!")

        for i in range(start_period, end_period):
            fixing_date = notional_structure.get_fixing_date(i)
            fx = fx_forward_curve.rivapy_value(val_date, fixing_date)
            result.append(notional_structure.get_amount(i) * fx)
    elif isinstance(notional_structure, ConstNotionalStructure):
        for i in range(start_period, end_period):
            # print(i)
            # print(notional_structure.get_amount(i))
            result.append(notional_structure.get_amount(0))
    else:
        for i in range(start_period, end_period):
            # print(i)
            # print(notional_structure.get_amount(i))
            result.append(notional_structure.get_amount(i))

    return result


# getPricingData


# difference between func price and priceImpl???


# computeSwapSpread


# computeBasisSpread
if __name__ == "__main__":
    pass
    # InterestRateSwapPricer
