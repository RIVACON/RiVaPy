from datetime import datetime, date
from scipy.optimize import brentq
from rivapy.tools.interfaces import BaseDatedCurve, HasExpectedCashflows
from rivapy.marketdata import DiscountCurveParametrized, ConstantRate, DiscountCurve
from rivapy.pricing.pricing_request import PricingRequest
from rivapy.pricing._logger import logger
from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.instruments.fra_specifications import ForwardRateAgreementSpecification
from rivapy.instruments.ir_swap_specification import IrFixedLegSpecification, IrFloatLegSpecification, InterestRateSwapSpecification
from typing import List as _List, Union as _Union, Tuple
from rivapy.tools.datetools import DayCounter


import numpy as np


# If we follow pyvacon implementation
# Makes uses of a cashFlowEntry class
# Makes use of a cashFlowTable class? this wew dont use for now...


class CashFlow:
    # goal is to define a dynamically growing class that is still able to use
    # type validation and dot-access e.g. class.variable
    # In the end, it might be better to just define clearly the CashFlow class with
    # strict attributes ... #TODO

    # Define expected types here
    # Can be expanded when we know for sure which features we want to ensure typing for
    _schema = {
        "start_date": datetime,
        "end_date": datetime,
        "ccy": str,
        "amortization": bool,
        "prepayment_risk": bool,
    }

    def __init__(self, val: float = None):
        self.val = val
        self._attributes = {}

    def __getattr__(self, name: str) -> Any:
        """overwritting default getter for dynamically growing one

        Args:
            name (str): _description_

        Raises:
            AttributeError: _description_

        Returns:
            Any: _description_
        """
        try:
            return self._attributes[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        """overwriting default setter fr dynamically growing one
        which also checks for expected type validation.

        Args:
            name (str): _description_
            value (Any): _description_

        Raises:
            TypeError: _description_
        """
        if name in {"val", "_attributes"}:  # avoid infinite recursion
            super().__setattr__(name, value)  # use the the normal attribute storage from base class
        else:  # logic for new attirbute storage
            expected_type = self._schema.get(name)  # if it doesnt exist, can attempt to set new attribute
            if expected_type is not None and not isinstance(value, expected_type):
                raise TypeError(f"Attribute '{name}' must be of type {expected_type}, got {type(value)}")
            self._attributes[name] = value

    def __delattr__(self, name: str):
        if name in self._attributes:
            del self._attributes[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def keys(self):
        return list(self._attributes.keys())

    def items(self):
        return self._attributes.items()

    def __dir__(self):
        """overwritten in order to show dynamically store attributes as well.

        Returns:
            _type_: _description_
        """
        return super().__dir__() + list(self._attributes.keys())


class InterestRateSwapPricer:

    def __init__(
        self,
        val_date: _Union[date, datetime],
        spec: InterestRateSwapSpecification,
        discount_curve: DiscountCurve,
        forward_curve: DiscountCurve,
    ):
        """_summary_

        Args:
            val_date (_Union[date, datetime]): _description_
            fra_spec (ForwardRateAgreementSpecification): _description_
            discount_curve (DiscountCurve): _description_
            forward_curve(): from underlying index...
        """

        self._val_date = val_date
        self._spec = spec
        self._fixed_leg = spec.fixed_leg
        self._float_leg = spec.float_leg
        self._discount_curve = discount_curve
        self._forward_curve = forward_curve

    # need cashFlowEntry
    # need cashFlowTable

    @staticmethod
    def populateCashFlowTableFix(
        val_date: _Union[date, datetime],
        fixed_leg_spec: IrFixedLegSpecification,
        discount_curve: DiscountCurve,
        forward_curve: DiscountCurve,
        set_rate: bool = False,
        desired_rate: float = None,
    ) -> _List[CashFlow]:
        # fxForwardCurve
        # fixingMap? -means?
        # setRate? - means? boolean if you want to overide the fixed rate used compared to the specification
        # rate? - means the chosen rate to be used instead of the specification rate

        # get start dates - is a vector
        # get end dates - is a vector
        # get pay dates - is a vector
        # get notional structure - is a vector, for our first case, it should be constant ?
        # size saved, and taken from notional structure

        # overwrite fixed interest if desired
        fixed_rate = fixed_leg_spec.fixed_rate
        if set_rate:
            fixed_rate = desired_rate

        # init output storage object
        entries = []
        dcc = DayCounter(discount_curve.daycounter)

        # get projected notionals

        # What for?????? - in the case that the notional changes over the lifetime of the swap
        # e.g. if the notional "resets"
        # getProjectedNotionals(
        #     notionals,
        #     valDate,
        #     leg->getNotionalStructure(),
        #     0,
        #     notionals.size(),
        #     fxForwardCurve,
        #     fixingMap);

        # notionals = #does this deteermine the schedule? #for now, assume notional is always the same

        notionals = fixed_leg_spec.notional * np.ones(
            len(fixed_leg_spec.pay_dates)
        )  # leg_spec.get_projected_notionals(val_date, leg_data.fx_forward_curve, fixing_table)

        for i in range(len(notionals)):
            entry = CashFlow()
            entry.start_date = fixed_leg_spec.start_dates[i]
            entry.end_date = fixed_leg_spec.end_dates[i]
            entry.pay_date = fixed_leg_spec.pay_dates[i]
            entry.notional = notionals[i]
            entry.rate = fixed_rate
            entry.interest_yf = dcc.yf(entry.start_date, entry.end_date)
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

        # TODO: NOTIONAL CASHFLOWS NOT YET INCLUDED ...

        return entries  # a LIST of ENTRY objects, where each object has the PV

    # TODO: FIXING TABLE CLASS

    @staticmethod
    def _populate_cashflows_float(
        val_date: _Union[date, datetime],
        float_leg_spec: IrFloatLegSpecification,
        discount_curve: DiscountCurve,
        forward_curve: DiscountCurve,
        fx_forward_curve: DiscountCurve,
        fixing_table,
        fixing_grace_period,
        setSpread: bool = False,
        spread: float = None,
    ) -> _List[CashFlow]:
        """_summary_

        Args:
            val_date (_Union[date, datetime]): _description_
            float_leg_spec (IrFloatLegSpecification): _description_
            discount_curve (DiscountCurve): _description_
            forward_curve (DiscountCurve): _description_
            fx_forward_curve (DiscountCurve): _description_
            fixing_table (_type_): _description_
            fixing_grace_period (_type_): Given in units of days, including weekends, and holidays e.g. ISDA
            setSpread (bool, optional): _description_. Defaults to False.
            spread (float, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _List[CashFlow]: _description_
        """
        entries = []
        udl = float_leg_spec.udl_id

        # swap day count convention
        dcc = DayCounter(discount_curve.daycounter)
        # rate day count convention # note that the specification also has dcc but withoout the curve...
        rate_dcc = DayCounter(fx_forward_curve.daycounter)

        # overwrite spread if desired
        leg_spread = float_leg_spec.spread
        if setSpread:
            leg_spread = spread

        # TODO: implement
        # getProjectedNotionals(
        #     notionals,
        #     valDate,
        #     leg->getNotionalStructure(),
        #     0,
        #     notionals.size(),
        #     fxForwardCurve,
        #     fixingMap);

        # important vectors
        # start dates
        # end dates
        # rate start dates
        # rate end dates
        # pay dates
        # reset dates

        # notionals = leg_spec.get_projected_notionals(val_date, leg_data.fx_forward_curve, fixing_table)
        notionals = float_leg_spec.notional * np.ones(len(float_leg_spec.pay_dates))

        for i in range(len(notionals)):
            entry = CashFlow()
            entry.start_date = float_leg_spec.start_dates[i]
            entry.end_date = float_leg_spec.end_dates[i]
            entry.pay_date = float_leg_spec.pay_dates[i]
            entry.notional = notionals[i]
            entry.interest_yf = dcc.yf(entry.start_date, entry.end_date)
            rate_yf = rate_dcc.yf(float_leg_spec.rate_start_dates[i], float_leg_spec.rate_end_dates[i])

            if val_date <= float_leg_spec.reset_dates[i]:
                fwd_rate = forward_curve.rivapy_valueFWD(val_date, float_leg_spec.rate_start_dates[i], float_leg_spec.rate_end_dates[i])
                entry.rate = leg_spread + (1.0 / fwd_rate - 1.0) / rate_yf

            else:
                fixing = fixing_table.get_fixing(udl, float_leg_spec.reset_dates[i])  # TODO FIXING TABLE CLASS
                if fixing is None:  # i.e. no fixing available

                    if val_date - float_leg_spec.reset_dates[i] > fixing_grace_period:
                        raise ValueError(f"Missing fixing for {udl} on {float_leg_spec.reset_dates[i]}")

                    else:
                        # fix value of payment i in future based on current discount curve and a period between
                        # valDate and valDate+length of original period (workaround if fixing is not available)
                        # taken from pyvacon
                        if entry.pay_date >= val_date:  # TODO unerstand the logic
                            time_delta = entry.end_date - entry.start_date
                            fixing = (1.0 / forward_curve.rivapy_valueFWD(val_date, val_date, val_date + time_delta) - 1) / entry.interest_yf

                entry.rate = fixing + spread

            if val_date <= entry.pay_date:
                entry.discount_factor = discount_curve.rivapy_value(val_date, entry.pay_date)
            else:
                entry.discount_factor = 0.0

            # given rate, notional, and yf, calcl interest
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
            entry.present_value = entry.pay_amount * entry.discount_factor

            entries.append(entry)

            # NOTE START AND END NOTIONALS CASHFLOWS NOT YET IMPLEMENTED

        return entries



    @staticmethod
    def price_leg(val_date, pricing_data: InterestRateSwapLegPricingData, fixing_map, param):
        """Pricing a single Leg using Pricing Data architecture

        Args:
            val_date (_type_): _description_
            pricing_data (InterestRateSwapLegPricingData): _description_
            fixing_map (_type_): _description_
            param (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        #get leg info from PricingData ->spec 
        leg_spec = pricing_data.leg_spec # what kind of leg?
        #get discount curve data from PricingData -> discoutn curve
        discount_curve = pricing_data.discount_curve

        if leg_spec.type == IrLegType.FIXED:
            cashflow_table = InterestRateSwapPricer.populateCashFlowTableFix()
        elif leg_spec.type == IrLegType.FLOAT:
            cashflow_table = InterestRateSwapPricer.populateCashFlowTableFloat() 
        #elif leg_spec.type ==  IrLegType.OIS:
        #    populateCashFlowTableOIS
        else:
            raise ValueError(f"Unknown leg type {leg_spec.type}")

        PV=0
        for entry in cashflow_table:

            PV += entry.present_value

        return PV
        #retunr PV* pricing_data.fx_rate ????


    def price(self):
        """price a full swap, with a pay leg and a receive leg
        """
        # PricingResults& results, -> implement also?
        val_date = self._val_date    # const  boost::posix_time::ptime& valDate,
        discount_curve_pay_leg = self. # const std::shared_ptr<const DiscountCurve>& discountCurvePayLeg,
        discount_curve_receive_leg= self. # const std::shared_ptr<const DiscountCurve>& discountCurveReceiveLeg,
        fixing_curve_pay_leg = # const std::shared_ptr<const DiscountCurve>& fixingCurvePayLeg,
        fixing_curve_receive_leg= # const std::shared_ptr<const DiscountCurve>& fixingCurveReceiveLeg,
        fx_fwd_curve_pay_leg = # const std::shared_ptr<const FxForwardCurve>& fxForwardCurvePayLeg,
        fx_fwd_curve_receive_leg = # const std::shared_ptr<const FxForwardCurve>& fxForwardCurveReceiveLeg,
        interest_rate_swap_spec = # const std::shared_ptr<const InterestRateSwapSpecification>& spec,
        pricing_request = # const PricingRequest& pricingRequest,
        pricing_param = # std::shared_ptr<const InterestRateSwapPricingParameter> pricingParam,
        fixing_map = # std::shared_ptr<const FixingTable> fixingMap,
        fx_pay_leg = # double fxPayLeg,
        fx_receive_leg = # double fxReceiveLeg)

        #unit in days
        fixing_grace_period = pricing_param["fixing_grace_period"] # TODO - check when pricing params are properly implemented

        #TODO check structure again....
	    price = InterestRateSwapPricer.price_leg(valDate, discountCurveReceiveLeg, fixingCurveReceiveLeg, fxForwardCurveReceiveLeg, spec->getReceiveLeg(), fixingMap, fixingGracePeriod) * fxReceiveLeg;
		price -= InterestRateSwapPricer.price_leg(valDate, discountCurvePayLeg, fixingCurvePayLeg, fxForwardCurvePayLeg, spec->getPayLeg(), fixingMap, fixingGracePeriod) * fxPayLeg;
		#results.setPrice(price);



        PV = 0
        return PV


# getPricingData

# populateCashFlowTableFix

# populateCashFlowTableFloat

# getProjectedNotionals ???

# populateCashFlowTableOIS


# FUNCTION: price ,
# return PV single or array?


# priceLeg


# difference between func price and priceImpl???


# computeSwapRate

# computeSwapSpread


# computeBasisSpread
if __name__ == "__main__":
    pass
    # InterestRateSwapPricer
