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
from holidays import HolidayBase as _HolidayBase
from holidays import EuropeanCentralBank as _ECB
from dateutil.relativedelta import relativedelta
from rivapy.instruments.components import Issuer

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
        fixing_date: _Optional[_Union[date, datetime]] = None,
        maturity_date: _Optional[_Union[date, datetime]] = None,
        currency: _Union[Currency, str] = "EUR",
        notional: float = 100.0,
        rate: float = 0.00,
        term: _Optional[_Union[Period, str]] = None,
        day_count_convention: _Union[DayCounterType, str] = DayCounterType.ACT360,
        business_day_convention: _Union[RollConvention, str] = RollConvention.MODIFIED_FOLLOWING,
        roll_convention: _Union[RollRule, str] = RollRule.EOM,
        spot_days: int = 2,
        calendar: _Union[_HolidayBase, str] = _ECB(),
        issuer: _Optional[_Union[Issuer, str]] = None,
        securitization_level: _Union[SecuritizationLevel, str] = SecuritizationLevel.NONE,
        payment_days: int = 0,
        adjust_start_date: bool = True,
        adjust_end_date: bool = False,
        index: _Optional[_Union[InterestRateIndex, str]] = None,
    ):
        """
        Deposit specification.
        Accrual start is adjusted according business day conventions, payment occurs on the maturity date (plus settlement days), spot days are adjusted to 1 or 0 respectively if deposit is T/N or O/N.

        Args:
            obj_id (str): (Preferably) Unique label of the deposit.
            fixing_date (_Union[date, datetime]): Date on which the reference rate is set. Must be a business day and must lie at or before the start_date. is rolled to business day according to business_day_convention if not provided as business day.
            start_date (_Union[date, datetime]): Date on when deposits begins for accrual or settlement. Is rolled to business day according to the business_day_convention if adjust_start_date is True. Must lie at or after the fixing_date.
            end_date (_Union[date, datetime]): Date on which the deposit ends for accrual or settlement. May be a holiday according to applicable calendar. Must lie after the start_date and will be rolled according to business day convention if adjust_end_date is True.
            maturity_date (_Union[date, datetime]): Date when deposit matures formally, lies on a good business day. Must lie at or after the end_date.
            currency (str, optional): Currency as alphabetic, Defaults to 'EUR'.
            notional (float, optional): Deposit's notional/face value. Must be positive. Defaults to 100.0.
            rate (float): Deposit fixed rate.
            term (_Union[Period, str], optional): Deposit term. If provided and no end date is given, it is used to calculate the end date from the start date.
            day_count_convention (Union[DayCounter, str], optional): Day count convention for determining period length. Defaults to DayCounter.ThirtyU360.
            business_day_convention (Union[RollConvention, str], optional): Set of rules defining the adjustment of  days to ensure each date being a business day with respect to a given holiday calendar. Defaults to RollConvention.FOLLOWING
            roll_convention (Union[RollRule],str], optional): Roll convention to be applied when building a schedule. Defaults to RollRule.NONE.
            spot_days (int, optional): Number of days after fixing date when the deposit is actually settled. Defaults to 2 and is set to 0, if start_date == fixing_date or O/N deposit, and is set to 1 for T/N deposit or start_date = fixing_date+1.
            calendar (Union[HolidayBase, str], optional): Holiday calendar to be used for business day adjustment. Defaults to ECB calendar.
            issuer (str, optional): Name/id of issuer. Defaults to None.
            securitization_level (_Union[SecuritizationLevel, str], optional): Securitization level. Defaults to None.
            payment_days (int, optional): Number of days after maturity date when the cashflow is actually paid. Defaults to 2.
            adjust_start_date (bool, optional): Whether to adjust the start date to the next business day if it falls on a holiday. Defaults to True.
            adjust_end_date (bool, optional): Whether to adjust the end date to the next business day if it falls on a holiday. Defaults to False.
        """
        self.rate = rate

        # Store original input of fixing date
        self.fixing_date = fixing_date

        # check and adjust spot_days for O/N and T/N deposits
        if term == "O/N":
            spd = 0
            if term != None and term != "O/N":
                logger.error(f"term given as {term} and not as 'O/N' but fixing_date == start_date -> inconsistent data")
            elif term == "O/N":
                logger.error(f"term given as {term} but fixing_date != start_date -> inconsistent data")
            logger.info("Setting spot_days to 0: O/N deposit or fixing_date equal to start_date.")
        elif term == "T/N":
            spd = 1
            if term != None and term != "T/N":
                logger.error(f"term given as {term} and not as 'T/N' but fixing_date + 1 day == start_date -> inconsistent data")
            elif term == "T/N":
                logger.error(f"term given as {term} but fixing_date + 1 day != start_date -> inconsistent data")
            logger.info("Setting spot_days to 1: T/N deposit or fixing_date + 1 day equal to start_date.")
        else:
            spd = spot_days

        if maturity_date is None and term is None:
            raise ValueError("Either maturity_date or term must be provided for DepositSpecification.")
        elif maturity_date is None and term is not None:
            # calculate maturity date from term and start date
            if issue_date is None:
                raise ValueError("issue_date must be provided if maturity_date is to be calculated from term.")
            maturity_date = (
                calc_end_day(issue_date, term, business_day_convention, calendar) + relativedelta(_term_to_period(term))
                if adjust_start_date
                else issue_date + relativedelta(_term_to_period(term))
            )

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
            index=index,
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
