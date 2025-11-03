# TODO:
# - consider proper end date handling
# - move date handling to hasexpectedcf
# - correct _frequency, _dcc issues...

from abc import abstractmethod as _abstractmethod
from typing import List as _List, Union as _Union, Tuple, Optional as _Optional
import numpy as np
import logging
from rivapy.instruments.bond_specifications import DeterministicCashflowBondSpecification
from datetime import datetime, date, timedelta
from rivapy.tools.holidays_compat import HolidayBase as _HolidayBase, ECB as _ECB
from dateutil.relativedelta import relativedelta
from rivapy.instruments.components import Issuer, NotionalStructure

from rivapy.tools.datetools import (
    Period,
    _date_to_datetime,
    _term_to_period,
    calc_end_day,
    calc_start_day,
    roll_day,
    next_or_previous_business_day,
    is_business_day,
    serialize_date,
)
from rivapy.tools.enums import DayCounterType, InterestRateIndex, RollConvention, SecuritizationLevel, Currency, Rating, RollRule, Instrument

import rivapy.tools.interfaces as interfaces

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DepositSpecification(DeterministicCashflowBondSpecification):

    def __init__(
        self,
        obj_id: str,
        issue_date: _Optional[_Union[date, datetime]] = None,
        maturity_date: _Optional[_Union[date, datetime]] = None,
        currency: _Union[Currency, str] = "EUR",
        notional: _Union[NotionalStructure, float] = 100.0,
        rate: float = 0.00,
        term: _Optional[_Union[Period, str]] = None,
        day_count_convention: _Union[DayCounterType, str] = "ACT360",
        business_day_convention: _Union[RollConvention, str] = "ModifiedFollowing",
        roll_convention: _Union[RollRule, str] = "EOM",
        spot_days: int = 2,
        calendar: _Union[_HolidayBase, str] = _ECB(),
        issuer: _Optional[_Union[Issuer, str]] = None,
        securitization_level: _Union[SecuritizationLevel, str] = "NONE",
        payment_days: int = 0,
        adjust_start_date: bool = True,
        adjust_end_date: bool = False,
    ):
        """Create a short-term deposit specification.

        Accrual start is adjusted according to the provided business day convention. Payment
        occurs on the maturity date (plus any settlement/payment days). For overnight ("O/N")
        and tomorrow-next ("T/N") deposits the :pyarg:`spot_days` is set to 0 and 1,
        respectively.

        Args:
            obj_id (str): Identifier for the deposit (e.g. ISIN or internal id).
            issue_date (date | datetime, optional): Fixing date and start date (of accrual period) of the deposit
                is calculated based on the provided issue date given :pyarg:`spot_days` and :pyarg:`adjust_start_date`. Required
                if :pyarg:`maturity_date` is computed from :pyarg:`term`.
            maturity_date (date | datetime, optional): Maturity date. If ``None`` and
                :pyarg:`term` is provided, the maturity will be derived from
                :pyarg:`issue_date` and :pyarg:`term`. Corresponds to end date (of accrual period). If non business day, always adjusted according to
                :pyarg:`business_day_convention` while end date adjustment is controlled by :pyarg:`adjust_end_date`.
            currency (Currency | str, optional): Currency code or enum. Defaults to "EUR".
            notional (NotionalStructure | float, optional): Face value; maybe passed as float or notional structure, amount must be positive. Defaults to 100.0.
            rate (float, optional): Fixed deposit rate (coupon). Defaults to 0.0.
            term (Period | str, optional): Tenor of the deposit (e.g. "3M", "1Y", "O/N", "T/N").
            day_count_convention (DayCounterType | str, optional): Day count convention.
                Defaults to :pydata:`DayCounterType.ACT360`.
            business_day_convention (RollConvention | str, optional): Business day convention
                used for rolling dates. Defaults to :pydata:`RollConvention.MODIFIED_FOLLOWING`.
            roll_convention (RollRule | str, optional): Roll rule when building schedules.
                Defaults to :pydata:`RollRule.EOM`.
            spot_days (int, optional): Settlement lag in days. Defaults to 2; overridden to 0
                for O/N and 1 for T/N when :pyarg:`term` is set accordingly.
            calendar (HolidayBase | str, optional): Holiday calendar to use. Defaults to ECB.
            issuer (Issuer | str, optional): Issuer identifier.
            securitization_level (SecuritizationLevel | str, optional): Securitization level.
                Defaults to :pydata:`SecuritizationLevel.NONE`.
            payment_days (int, optional): Days after maturity when payment occurs. Defaults to 0.
            adjust_start_date (bool, optional): If True, roll :pyarg:`issue_date` forward to a
                business day when required, to ensure accrual starts on a business day. The adjusted date will be used for calculations. Defaults to True.
            adjust_end_date (bool, optional): If True, roll :pyarg:`maturity_date` forward to a
                business day when required, to ensure accrual ends on a business day. The adjusted date will be used for calculations. Defaults to False.

        Raises:
            ValueError: If neither :pyarg:`maturity_date` nor :pyarg:`term` is provided, or if
                :pyarg:`issue_date` is required to compute :pyarg:`maturity_date` but is missing.
        """
        self.rate = rate

        # check and adjust spot_days for O/N and T/N deposits
        if term == "O/N":
            spd = 0
            logger.info("Setting spot_days to 0: O/N deposit.")
        elif term == "T/N":
            spd = 1
            logger.info("Setting spot_days to 1: T/N deposit.")
        else:
            spd = spot_days

        if maturity_date is None and term is None:
            raise ValueError("Either maturity_date or term must be provided for DepositSpecification.")
        elif maturity_date is None and term is not None:
            # calculate maturity date from term and start date
            if issue_date is None:
                raise ValueError("issue_date must be provided if maturity_date is to be calculated from term.")
            # calculate maturity date from term and start date
            # roll_day signature: roll_day(day, calendar, business_day_convention, ...)
            # previously the calendar and business day convention were passed in the wrong order
            if adjust_start_date:
                help_date = roll_day(issue_date, calendar, business_day_convention)
            else:
                help_date = issue_date
            # _term_to_period returns a Period(years, months, days)
            period = _term_to_period(term)
            maturity_date = help_date + relativedelta(years=period.years, months=period.months, days=period.days)
        if isinstance(issue_date, date):
            issue_date = datetime.combine(issue_date, datetime.min.time())
        if isinstance(maturity_date, date):
            maturity_date = datetime.combine(maturity_date, datetime.min.time())

        if term is None:
            term = f"{(maturity_date - issue_date).days}D"
        else:
            term = term

        super().__init__(
            obj_id=obj_id,
            spot_days=spd,
            issue_date=issue_date,
            maturity_date=maturity_date,
            notional=notional,
            currency=currency,
            coupon=rate,
            frequency=term,
            day_count_convention=day_count_convention,
            business_day_convention=business_day_convention,
            roll_convention=roll_convention,
            calendar=calendar,
            notional_exchange=True,
            payment_days=payment_days,
            issuer=issuer,
            securitization_level=securitization_level,
            adjust_end_date=adjust_end_date,
            adjust_start_date=adjust_start_date,
        )

    @staticmethod
    def _create_sample(
        n_samples: int, seed: int = None, ref_date=None, issuers: _List[str] = None, sec_levels: _List[str] = None, currencies: _List[str] = None
    ) -> _List["DepositSpecification"]:
        if seed is not None:
            np.random.seed(seed)
        if ref_date is None:
            ref_date = datetime.now()
        else:
            ref_date = _date_to_datetime(ref_date)
        if issuers is None:
            issuers = ["Issuer_" + str(i) for i in range(int(n_samples / 2))]
        result = []
        if currencies is None:
            currencies = list(Currency)
        if sec_levels is None:
            sec_levels = list(SecuritizationLevel)
        for i in range(n_samples):
            days = int(15.0 * 365.0 * np.random.beta(2.0, 2.0)) + 1
            start_date = ref_date + timedelta(days=np.random.randint(low=-365, high=0))
            result.append(
                DepositSpecification(
                    obj_id=f"Deposit_{i}",
                    start_date=start_date,
                    maturity_date=ref_date + timedelta(days=days),
                    currency=np.random.choice(currencies),
                    notional=np.random.choice([100.0, 1000.0, 10_000.0, 100_0000.0]),
                    rate=np.random.choice([0.01, 0.02, 0.03, 0.04, 0.05]),
                    issuer=np.random.choice(issuers),
                    securitization_level=np.random.choice(sec_levels),
                )
            )
        return result

    def _to_dict(self) -> dict:
        result = {
            "obj_id": self.obj_id,
            "issue_date": serialize_date(self.issue_date),
            "maturity_date": serialize_date(self.maturity_date),
            "currency": self.currency,
            "notional": self.notional,
            "rate": self.rate,
            "day_count_convention": self.day_count_convention,
            "roll_convention": self._roll_convention,
            "spot_days": self._spot_days,
            "business_day_convention": self.business_day_convention,
            "issuer": self.issuer,
            "securitization_level": self._securitization_level,
            "payment_days": self._payment_days,
        }
        return result

        # region properties

    def ins_type(self):
        """Return instrument type

        Returns:
            Instrument: Forward rate agreement
        """
        return Instrument.DEPOSIT

    # temp placeholder
    def get_end_date(self):
        return self._end_date
