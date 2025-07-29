# ToDo:
# - consider proper end date handling
# - move date handling to hasexpectedcf
# - correct _frequency, _dcc issues...

from abc import abstractmethod as _abstractmethod
from typing import List as _List, Union as _Union, Tuple, Optional as _Optional
import numpy as np
from rivapy.instruments import HasExpectedCashflows
from datetime import datetime, date, timedelta
from holidays import HolidayBase as _HolidayBase
from holidays import EuropeanCentralBank as _ECB
from dateutil.relativedelta import relativedelta
from rivapy.tools.datetools import Period, _date_to_datetime, _term_to_period, calc_end_day, calc_start_day, roll_day
from rivapy.tools.enums import DayCounterType, RollConvention, SecuritizationLevel, Currency, Rating, RollRule, Instrument

import rivapy.tools.interfaces as interfaces
from rivapy.tools.datetools import Period, is_business_day


class DepositSpecification(HasExpectedCashflows):

    def __init__(
        self,
        obj_id: str,
        fixing_date: _Optional[_Union[date, datetime]] = None,
        end_date: _Optional[_Union[date, datetime]] = None,
        start_date: _Optional[_Union[date, datetime]] = None,
        maturity_date: _Optional[_Union[date, datetime]] = None,
        currency: _Union[Currency, str] = "EUR",
        notional: float = 100.0,
        rate: float = 0.00,
        term: _Optional[_Union[Period, str]] = None,
        day_count_convention: _Union[DayCounterType, str] = DayCounterType.ACT360,
        business_day_convention: _Union[RollConvention, str] = RollConvention.MODIFIED_FOLLOWING,
        roll_convention: _Union[RollRule, str] = RollRule.EOM,
        spot_lag: int = 2,
        calendar: _Union[_HolidayBase, str] = _ECB(),
        issuer: _Optional[str] = None,
        securitization_level: _Union[SecuritizationLevel, str] = SecuritizationLevel.NONE,
        settlement_days: int = 2,
        adjust_start_date: bool = True,
        adjust_end_date: bool = False,
    ):
        """
        Deposit specification.
        Accrual start is adjusted according business day conventions, payment occurs on the maturity date (plus settlement days), spot days are adjusted to 1 or 0 respectively if deposit is T/N or O/N.

        Args:
            obj_id (str): (Preferably) Unique label of the deposit.
            fixing_date (_Union[date, datetime]): Date on which the reference rate is set. Must lie at or before the start_date.
            start_date (_Union[date, datetime]): Date on when deposits begins for accrual or settlement. Is rolled to business day according to the business_day_convention. Must lie at or before the fixing_date.
            end_date (_Union[date, datetime]): Date on which the deposit ends for accrual or settlement. May be a holiday according to applicable calendar. Must lie after the start_date.
            maturity_date (_Union[date, datetime]): Date when deposit is matures formally, lies on a good business day. Must lie at or after the start_date.
            currency (str, optional): Currency as alphabetic, Defaults to 'EUR'.
            notional (float, optional): Deposit's notional/face value. Must be positive. Defaults to 100.0.
            rate (float): Deposit fixed rate.
            term (_Union[Period, str], optional): Deposit term. If provided, it is used to calculate the maturity date from the start date.
            day_count_convention (Union[DayCounter, str], optional): Day count convention for determining period
                                                                     length. Defaults to DayCounter.ThirtyU360.
            business_day_convention (Union[RollConvention, str], optional): Set of rules defining the adjustment of
                                                                            days to ensure each date being a business
                                                                            day with respect to a given holiday
                                                                            calendar. Defaults to
                                                                            RollConvention.FOLLOWING
            roll_convention (Union[RollRule],str], optional): Roll convention to be applied when building a schedule. Defaults to RollRule.NONE.
            spot_lag (int, optional): Number of days after fixing date when the deposit is actually settled. Defaults to 2.
            calendar (Union[HolidayBase, str], optional): Holiday calendar to be used for business day adjustment. Defaults to ECB calendar.
            issuer (str, optional): Name/id of issuer. Defaults to None.
            securitization_level (_Union[SecuritizationLevel, str], optional): Securitization level. Defaults to None.
            rating (_Union[Rating, str]): Paper rating.
        """

        # check and adjust spot_days for O/N and T/N deposits
        if term == "O/N" or (fixing_date is not None and start_date is not None and fixing_date == start_date):
            spd = 0
            print("Setting spot_lag to 0 for O/N deposit or fixing_date equals start_date.")
        elif term == "T/N" or (fixing_date is not None and start_date is not None and fixing_date + relativedelta(days=1) == start_date):
            spd = 1
            print("Setting spot_lag to 1 for T/N deposit or fixing_date + 1 day equals start_date.")
        else:
            spd = spot_lag

        # set fixing date,  start date, end date, and maturity date
        if fixing_date is not None:
            if not is_business_day(fixing_date, calendar=calendar):
                fd = roll_day(fixing_date, calendar=calendar, business_day_convention=business_day_convention)
                print("Set fixing_date to next business day acc. to business_day_convention.")
            else:
                fd = fixing_date
        elif start_date is not None:
            fd = calc_start_day(
                roll_day(start_date, calendar=calendar, business_day_convention=business_day_convention),
                f"{self._spot_days}D",
                business_day_convention=business_day_convention,
                calendar=calendar,
            )
            print("Set fixing_date to start_date adjusted backwards by spot_days and business_day_convention.")
        else:
            raise ValueError("Either fixing_date or start_date must be provided.")

        if start_date is not None:
            if not is_business_day(start_date, calendar=calendar) and adjust_start_date:
                sd = roll_day(start_date, calendar=calendar, business_day_convention=business_day_convention)
                print("Set start_date to next business day acc. to business_day_convention.")
            else:
                sd = start_date
        elif fixing_date is not None and adjust_start_date:
            sd = calc_end_day(
                fixing_date,
                f"{self._spot_days}D",
                business_day_convention=business_day_convention,
                calendar=calendar,
                roll_convention=roll_convention,
            )
            print("Set start_date to fixing_date adjusted by spot_days and business_day_convention.")
        elif fixing_date is not None:
            sd = fd + timedelta(days=spd)
            print("Set start_date to fixing_date adjusted by spot_days without business day adjustment (i.e. unadjusted).")
        else:
            raise ValueError("Either fixing_date or start_date must be provided.")

        if end_date is not None and (not adjust_end_date or is_business_day(end_date, calendar=calendar)):
            ed = end_date
        elif end_date is not None:
            ed = roll_day(end_date, calendar=calendar, business_day_convention=business_day_convention)
            print("Set end_date to next business day acc. to business_day_convention.")
        elif term is not None and not adjust_end_date:
            ed = calc_end_day(sd, term, calendar=calendar, roll_convention=roll_convention)
            print("set end_date to start_date adjusted by term and roll_convention, not adjusted by business_day_convention.")
        elif term is not None:
            ed = calc_end_day(sd, term, calendar=calendar, roll_convention=roll_convention, business_day_convention=business_day_convention)
            print("set end_date to start_date adjusted by term and roll_convention and business_day_convention.")
        elif maturity_date is not None:
            if not is_business_day(maturity_date, calendar=calendar) and adjust_end_date:
                ed = roll_day(maturity_date, calendar=calendar, business_day_convention=business_day_convention)
            else:
                ed = maturity_date
        else:
            raise ValueError("Either end_date, term and start_date, or maturity_date must be provided.")

        if maturity_date is not None:
            if not is_business_day(maturity_date, calendar=calendar):
                md = roll_day(maturity_date, calendar=calendar, business_day_convention=business_day_convention)
                print("Set maturity_date to next business day acc. to business_day_convention.")
        else:
            md = roll_day(ed, calendar=calendar, business_day_convention=business_day_convention)
            print("Set maturity_date to end_date adjusted by business_day_convention.")

        if term is None:
            t = f"{(ed - sd).days}D"
            print("Set term to the difference between end_date and start_date in days.")
        else:
            t = term

        super().__init__(
            obj_id=obj_id,
            first_fixing_date=fd,
            spot_lag=spd,
            start_date=sd,
            end_date=ed,
            maturity_date=md,
            notional=notional,
            currency=currency,
            coupon=rate,
            frequency=t,
            day_count_convention=day_count_convention,
            business_day_convention=business_day_convention,
            roll_convention=roll_convention,
            calendar=calendar,
            notional_exchange=True,
            settlement_days=settlement_days,
            issuer=issuer,
            securitization_level=securitization_level,
        )

    @staticmethod
    def _create_sample(
        n_samples: int, seed: int = None, ref_date=None, issuers: _List[str] = None, sec_levels: _List[str] = None, currencies: _List[str] = None
    ) -> _List[dict]:
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
        for _ in range(n_samples):
            days = int(15.0 * 365.0 * np.random.beta(2.0, 2.0)) + 1
            start_date = ref_date + timedelta(days=np.random.randint(low=-365, high=0))
            result.append(
                {
                    "fixing_date": start_date + timedelta(days=np.random.randint(low=-2, high=0)),
                    "start_date": start_date,
                    "maturity_date": ref_date + timedelta(days=days),
                    "currency": np.random.choice(currencies),
                    "notional": np.random.choice([100.0, 1000.0, 10_000.0, 100_0000.0]),
                    "rate": np.random.choice([0.01, 0.02, 0.03, 0.04, 0.05]),
                    "issuer": np.random.choice(issuers),
                    "securitization_level": np.random.choice(sec_levels),
                }
            )
        return result

    def _to_dict(self) -> dict:
        result = {
            "obj_id": self.obj_id,
            "fixing_date": self.fixing_date,
            "start_date": self.start_date,
            "maturity_date": self.maturity_date,
            "currency": self.currency,
            "notional": self.notional,
            "rate": self.rate,
            "day_count_convention": self.day_count_convention,
            "roll_convention": self._roll_convention,
            "spot_days": self._spot_days,
            "business_day_convention": self.business_day_convention,
            "issuer": self.issuer,
            "securitization_level": self.securitization_level,
            "securitization_level_str": SecuritizationLevel.to_string(self.securitization_level),
            "settlement_days": self.settlement_days,
        }
        return result

    def ins_type(self):
        """Return instrument type

        Returns:
            Instrument: Forward rate agreement
        """
        return Instrument.Deposit
