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


class InterestRateSwapPricer:

    def __init__(
        self,
        val_date: _Union[date, datetime],
    ):
        """_summary_

        Args:
            val_date (_Union[date, datetime]): _description_
            fra_spec (ForwardRateAgreementSpecification): _description_
            discount_curve (DiscountCurve): _description_
            forward_curve(): from underlying index...
            spread_curve (_Union[DiscountCurve, float]): _description_
        """

    def price(self):
        """return prersent value (i.e. discoutned)"""

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
