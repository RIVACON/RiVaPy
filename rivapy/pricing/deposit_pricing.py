
from datetime import datetime
from rivapy.marketdata import DiscountCurve
from rivapy.pricing.pricing_request import PricingRequest
from rivapy.pricing._logger import logger
from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.tools.datetools import DayCounter
from rivapy.pricing.bond_pricing import SimpleCashflowPricer

class DepositPricer(SimpleCashflowPricer):

    # def __init__(
    #     self,
    #     val_date: _Union[date, datetime],
    #     deposit_spec: DepositSpecification,
    #     discount_curve: DiscountCurve,
    #     spread_curve: _Union[DiscountCurve, float] = 0.0,
    # ):
    #     """_summary_

    #     Args:
    #         val_date (_Union[date, datetime]): specific date for which the value of the financial instrument is calculated.
    #         deposit_spec (DepositSpecification): Specification object with deposit specific parameters.
    #         discount_curve (DiscountCurve): Discount curve used for discounting.
    #         spread_curve (_Union[DiscountCurve, float]): Spread curve
    #     """

    #     self._val_date = val_date
    #     self._spec = deposit_spec
    #     self._discount_curve = discount_curve
    #     self._spread_curve = spread_curve
    #     self._validate_pricer_dates()

    # def _validate_pricer_dates(self):
    #     """Validates consistency of valuation date, curve reference date, and deposit fixing date"""
    #     self._discount_curve.refdate, self._spec._fixing_date = _check_start_at_or_before_end(self._discount_curve.refdate, self._spec._fixing_date)
    #     self._discount_curve.refdate, self._val_date = _check_start_at_or_before_end(self._discount_curve.refdate, self._val_date)

    @staticmethod
    def price(val_date: datetime, specification: DepositSpecification, discount_curve: DiscountCurve) -> float:
        """Calculate the present value of the specified deposit given a discount curve and daycount convention

        Returns:
           float: present value of a deposit based on simple compounding
        """

        return SimpleCashflowPricer.pv_cashflows(val_date, specification, discount_curve)

    @staticmethod
    def implied_simply_compounded_rate(val_date: datetime, specification: DepositSpecification, discount_curve: DiscountCurve) -> float:
        """Calculates the fair simply compounded rate for a deposit contract such that the contract has zero value.
        The function assumes a simply compounded rate, i.e., D(t) = 1 / (1 + rate(t) * dt), and computes the implied rate
        using the provided discount curve and deposit specification.
        
        Parameters:
            val_date (datetime): The valuation date for the calculation.
            specification (DepositSpecification): The deposit contract specification, including start date, maturity date, and day count convention.
            discount_curve (DiscountCurve): The discount curve used to obtain discount factors, should match the issuer specific discount curve.
            
        Returns:
            float: The implied simply compounded rate that makes the contract value zero.
            
        Raises:
            ValueError: If the provided discount_curve is not of type DiscountCurve.
  
        """

        start_date = specification.start_date
        maturity_date = specification.maturity_date
        daycountconvention = specification.day_count_convention

        if isinstance(discount_curve, DiscountCurve):
            cont_df = discount_curve.rivapy_valueFWD(val_date, start_date, maturity_date)
        else:
            raise ValueError("Discount curve must be of type DiscountCurve")

        dcc = DayCounter(daycountconvention) 
        dt = dcc.yf(start_date, maturity_date)
        simple_rate = ((1 / cont_df) - 1) / dt

        return simple_rate
