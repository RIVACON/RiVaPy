from datetime import datetime, date
from rivapy.marketdata import DiscountCurve
from rivapy.pricing.pricing_request import PricingRequest
from rivapy.pricing._logger import logger
from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.tools.datetools import DayCounter
from rivapy.pricing.bond_pricing import DeterministicCashflowPricer
from typing import Tuple, Union as _Union, List as _List
from rivapy.tools._validators import _check_start_at_or_before_end


class DepositPricer(DeterministicCashflowPricer):

    def __init__(
        self,
        val_date: _Union[date, datetime],
        deposit_spec: DepositSpecification,
        discount_curve: DiscountCurve,
    ):
        """Create a pricer for a deposit instrument.

        Args:
            val_date (date | datetime): Valuation date used for pricing.
            deposit_spec (DepositSpecification): Deposit specification containing schedule
                and contractual parameters.
            discount_curve (DiscountCurve): Discount curve used for discounting cashflows.
        """

        self._val_date = val_date
        self._spec = deposit_spec
        self._discount_curve = discount_curve
        self._validate_pricer_dates()

    def _validate_pricer_dates(self):
        """Validate that the pricer valuation date is consistent with the discount curve refdate.

        This updates ``self._discount_curve.refdate`` and ``self._val_date`` to a
        canonical ordering using the project's date validators.
        """
        self._discount_curve.refdate, self._val_date = _check_start_at_or_before_end(self._discount_curve.refdate, self._val_date)

    def expected_cashflows(self) -> _List[Tuple[datetime, float]]:
        """Return expected cashflows for the configured deposit and valuation date.

        Returns:
            List[Tuple[datetime, float]]: List of (pay_date, amount) tuples.
        """
        return DeterministicCashflowPricer.expected_cashflows(self._spec, self._val_date)

    @staticmethod
    def get_expected_cashflows(
        specification: DepositSpecification, val_date: _Union[datetime.date, datetime, None] = None
    ) -> _List[Tuple[datetime, float]]:
        """Static helper: get expected cashflows for a deposit specification.

        Args:
            specification (DepositSpecification): The deposit specification.
            val_date (date | datetime, optional): Valuation date to use. If ``None``,
                the specification's default behavior is used.

        Returns:
            List[Tuple[datetime, float]]: List of (pay_date, amount) tuples.
        """
        if val_date is None:
            return DeterministicCashflowPricer.get_expected_cashflows(specification)
        else:
            return DeterministicCashflowPricer.get_expected_cashflows(specification, val_date)

    def price(self) -> float:
        """Return the present value of the configured deposit using the provided curves.

        Returns:
            float: Present value.
        """
        return self.get_price(self._val_date, self._spec, self._discount_curve)

    @staticmethod
    def get_price(val_date: datetime, specification: DepositSpecification, discount_curve: DiscountCurve) -> float:
        """Calculate present value of a deposit using the provided discount curve.

        Args:
            val_date (date | datetime): Valuation date.
            specification (DepositSpecification): Deposit contract specification.
            discount_curve (DiscountCurve): Discount curve used for discounting.

        Returns:
            float: Present value (PV) computed from discounted expected cashflows.
        """

        return DeterministicCashflowPricer.get_pv_cashflows(val_date, specification, discount_curve)

    def implied_simply_compounded_rate(self) -> float:
        """Return the implied simply compounded rate for the configured deposit.

        The implied simply compounded rate is the rate that makes the deposit's PV
        equal to zero under a simple-compounding assumption.

        Returns:
            float: Implied simply compounded rate.
        """
        return DepositPricer.get_implied_simply_compounded_rate(self._val_date, self._spec, self._discount_curve)

    @staticmethod
    def get_implied_simply_compounded_rate(val_date: datetime, specification: DepositSpecification, discount_curve: DiscountCurve) -> float:
        """Compute the implied simply compounded rate that sets deposit PV to zero.

        The implementation assumes simple compounding: D(t) = 1 / (1 + rate * dt).

        Args:
            val_date (date | datetime): Valuation date.
            specification (DepositSpecification): Deposit specification with start/end dates
                and day count convention.
            discount_curve (DiscountCurve): Discount curve used to obtain forward discount
                factor between start and end.

        Returns:
            float: The implied simply compounded rate.

        Raises:
            ValueError: If ``discount_curve`` is not an instance of DiscountCurve.
        """

        start_date = specification.start_date
        # maturity_date = specification.maturity_date # date of legal end of the deposit and when payment needs to be made
        end_date = (
            specification.end_date
        )  # period over which interest is calculated # this is the period we need to use as it is the actual accrual period, regardless of when payment is due
        daycountconvention = specification.day_count_convention

        if isinstance(discount_curve, DiscountCurve):
            cont_df = discount_curve.value_fwd(val_date, start_date, end_date)
        else:
            raise ValueError("Discount curve must be of type DiscountCurve")

        dcc = DayCounter(daycountconvention)
        dt = dcc.yf(start_date, end_date)
        simple_rate = ((1 / cont_df) - 1) / dt
        return simple_rate
