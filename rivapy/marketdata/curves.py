import math
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dateutil.relativedelta as relativedelta
import rivapy.tools.interfaces as interfaces
import rivapy.tools._validators as validators
from typing import List, Union, Tuple, Literal, Dict, Optional
from datetime import datetime, date, timedelta
from collections import defaultdict

try:
    import tensorflow as tf

    has_tf = True
except ImportError:
    has_tf = False

from rivapy.tools.enums import DayCounterType, InterpolationType, ExtrapolationType
from rivapy.tools.enums import EnergyTimeGridStructure as ets
from rivapy.tools.datetools import DayCounter
from rivapy.marketdata.factory import create as _create
from rivapy.marketdata_tools.pfc_shaper import PFCShaper
from rivapy.marketdata_tools.pfc_shifter import PFCShifter
from rivapy.tools.scheduler import SimpleSchedule, OffPeakSchedule, PeakSchedule, BaseSchedule
from rivapy.instruments.energy_futures_specifications import EnergyFutureSpecifications

from rivapy import _pyvacon_available

if _pyvacon_available:
    from pyvacon.finance.marketdata import EquityForwardCurve as _EquityForwardCurve
    from pyvacon.finance.marketdata import SurvivalCurve as _SurvivalCurve
    from pyvacon.finance.marketdata import DiscountCurve as _DiscountCurve
    import pyvacon as _pyvacon


class DiscountCurve:

    def __init__(
        self,
        id: str,
        refdate: Union[datetime, date],
        dates: List[Union[datetime, date]],
        df: List[float],
        interpolation: InterpolationType = InterpolationType.HAGAN_DF,
        extrapolation: ExtrapolationType = ExtrapolationType.NONE,
        daycounter: DayCounterType = DayCounterType.Act365Fixed,
    ):
        """Discountcurve

        Args:
            id (str): Identifier of the discount curve.
            refdate (Union[datetime, date]): Reference date of the discount curve.
            dates (List[Union[datetime, date]]): List of dates belonging to the list of discount factors. All dates must be distinct and equal or after the refdate, otherwise an exception will be thrown.
            df (List[float]): List of discount factors. Length of list of discount factors must equal to length of list of dates, otherwise an exception will be thrown.
            interpolation (enums.InterpolationType, optional): Defaults to InterpolationType.HAGAN_DF.
            extrapolation (enums.ExtrapolationType, optional): Defaults to ExtrapolationType.NONE which does not allow to compute a discount factor for a date past all given dates given to this constructor.
            daycounter (enums.DayCounterType, optional): Daycounter used within the interpolation formula to compute a discount factor between two dates from the dates-list above. Defaults to DayCounterType.Act365Fixed.

        """
        if len(dates) < 1:
            raise Exception("Please specify at least one date and discount factor")
        if len(dates) != len(df):
            raise Exception("List of dates and discount factors must have equal length.")
        self.values = sorted(zip(dates, df), key=lambda tup: tup[0])  # zip dates and discount factors and sort by dates
        if isinstance(refdate, datetime):
            self.refdate = refdate
        else:
            self.refdate = datetime(refdate, 0, 0, 0)
        if not isinstance(interpolation, InterpolationType):
            raise TypeError("Interpolation is not of type enums.InterpolationType")
        self.interpolation = interpolation
        if not isinstance(extrapolation, ExtrapolationType):
            raise TypeError("Extrapolation is not of type enums.ExtrapolationType")
        self.extrapolation = extrapolation
        if not isinstance(daycounter, DayCounterType):
            raise TypeError("Daycounter is not of type enums.DaycounterType")
        self.daycounter = daycounter
        self.id = id
        # check if dates are monotonically increasing and if first date is greather then refdate
        if self.values[0][0] < refdate:
            raise Exception("First date must be equal or greater then reference date.")
        if self.values[0][0] > refdate:
            self.values = [(self.refdate, 1.0)] + self.values
        if self.values[0][1] != 1.0:
            raise Exception("Discount factor for today must equal 1.0.")
        for i in range(1, len(self.values)):
            if self.values[i - 1] >= self.values[i]:
                raise Exception("Dates must be given in monotonically increasing order.")
        self._pyvacon_obj = None

    def get_dates(self) -> Tuple[datetime]:
        """Return list of dates of curve

        Returns:
            Tuple[datetime]: List of dates
        """
        x, y = zip(*self.values)
        return x

    def get_df(self) -> Tuple[float]:
        """Return list of discount factors

        Returns:
            Tuple[float]: List of discount factors
        """
        x, y = zip(*self.values)
        return y

    def value(self, refdate: Union[date, datetime], d: Union[date, datetime]) -> float:
        """Return discount factor for a given date

        Args:
            refdate (Union[date, datetime]): The reference date. If the reference date is in the future (compared to the curves reference date), the forward discount factor will be returned.
            d (Union[date, datetime]): The date for which the discount factor will be returned

        Returns:
            float: discount factor
        """
        if not isinstance(refdate, datetime):
            refdate = datetime(refdate, 0, 0, 0)
        if not isinstance(d, datetime):
            d = datetime(d, 0, 0, 0)
        if refdate < self.refdate:
            raise Exception("The given reference date is before the curves reference date.")
        return self._get_pyvacon_obj().value(refdate, d)

    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _DiscountCurve(
                self.id,
                self.refdate,
                [x for x in self.get_dates()],
                [x for x in self.get_df()],
                self.daycounter.value,
                self.interpolation,
                self.extrapolation,
            )
        return self._pyvacon_obj

    def plot(self, days: int = 10, discount_factors: bool = False, **kwargs):
        """Plots the discount curve using matplotlibs plot function.
        The timegrid includes the dates of the discount curve. Here either the discount factors or the zero rates (continuously compounded, ACT365 yearfraction) are plotted.

        Args:
            days (int, optional): The number of days between two plotted rates/discount factors. Defaults to 10.
            discount_factors (bool, optional): If True, discount factors will be plotted, otherwise the rates. Defaults to False.
            **kwargs: optional arguments that will be directly passed to the matplotlib plto function
        """
        dates = self.get_dates()
        dates_new = [dates[0]]
        for i in range(1, len(dates)):
            while dates_new[-1] + timedelta(days=days) < dates[i]:
                dates_new.append(dates_new[-1] + timedelta(days=days))
        dates_new.append(dates[-1])
        values = [self.value(self.refdate, d) for d in dates_new]

        if not discount_factors:
            for i in range(1, len(values)):
                dt = float((dates_new[i] - self.refdate).days) / 365.0
                values[i] = -math.log(values[i]) / dt
        values[0] = values[1]
        plt.plot(dates_new, values, label=self.id, **kwargs)


class NelsonSiegel(interfaces.FactoryObject):
    def __init__(self, beta0: float, beta1: float, beta2: float, tau: float):
        """Nelson-Siegel parametrization for rates and yields, see :footcite:t:`Nelson1987`.

        This parametrization is mostly used to parametrize rate curves and can be used in conjunction with :class:`rivapy.marketdata.DiscountCurveParametrized`. It is defined by

        .. math::

            f(t) = \\beta_0 + (\\beta_1+\\beta_2)\\frac{1-e^{-t/\\tau}}{t/\\tau} -\\beta_2e^{t/\\tau}


        Args:
            beta0 (float): This parameter is the asymptotic (for arbitrary large maturities) rate, see formula above.
            beta1 (float): beta0 + beta1 give the short term rate, see formula above.
            beta2 (float): This parameter controls the size of the hump, see formula above.
            tau (float): This parameter controls the location of the hump, see formula above.

        Examples:
            .. code-block:: python

                >>> from rivapy.marketdata.curves import NelsonSiegel, DiscountCurveParametrized
                >>> ns = NelsonSiegel(beta0=0.05, beta1 = 0.02, beta2=0.1, tau=1.0)
                >>> dc = DiscountCurveParametrized('DC',  refdate = dt.datetime(2023,1,1), rate_parametrization=ns, daycounter = DayCounterType.Act365Fixed)
                >>> dates = [dt.datetime(2023,1,1) + dt.timedelta(days=30*days) for days in range(120)]
                >>> values = [dc.value(refdate = dt.datetime(2023,1,1),d=d) for d in dates]
                >>> plt.plot(dates, values)
        """
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self._multiplier = 1.0

    def _to_dict(self) -> dict:
        return {"beta0": self.beta0, "beta1": self.beta1, "beta2": self.beta2, "tau": self.tau}

    def __call__(self, t: float):
        return self._multiplier * NelsonSiegel.compute(self.beta0, self.beta1, self.beta2, self.tau, t)

    def __mul__(self, x: float):
        result = NelsonSiegel(self.beta0, self.beta1, self.beta2, self.tau)
        result._multiplier = x
        return result

    @staticmethod
    def compute(beta0: float, beta1: float, beta2: float, tau: float, T: float) -> float:
        """_summary_

        Args:
            beta0 (float): longrun
            beta1 (float): beta0 + beta1 = shortrun
            beta2 (float): hump or through
            tau (float):locaton of hump
            T (float): _description_

        Returns:
            float: _description_
        """
        t = np.maximum(T, 1e-4) / tau
        return beta0 + beta1 * (1.0 - np.exp(-t)) / t + beta2 * ((1 - np.exp(-t)) / t - np.exp(-(t)))

    @staticmethod
    def _create_sample(
        n_samples: int,
        seed: int = None,
        min_short_term_rate: float = -0.01,
        max_short_term_rate: float = 0.12,
        min_long_run_rate: float = 0.005,
        max_long_run_rate: float = 0.15,
        min_hump: float = -0.1,
        max_hump: float = 0.1,
        min_tau: float = 0.5,
        max_tau: float = 3.0,
    ):
        if seed is not None:
            np.random.seed(seed)
        result = []
        for i in range(n_samples):
            beta0 = np.random.uniform(min_long_run_rate, max_long_run_rate)
            beta1 = np.random.uniform(min_short_term_rate - beta0, max_short_term_rate - beta0)
            beta2 = np.random.uniform(min_hump, max_hump)
            tau = np.random.uniform(min_tau, max_tau)
            result.append(NelsonSiegel(beta0, beta1, beta2, tau))
        return result

    if has_tf:

        @staticmethod
        def compute_tf(beta0: tf.Tensor, beta1: tf.Tensor, beta2: tf.Tensor, tau: tf.Tensor, T: tf.Tensor) -> tf.Tensor:
            """_summary_

            Args:
                beta0 (float): longrun
                beta1 (float): beta0 + beta1 = shortrun
                beta2 (float): hump or through
                tau (float):locaton of hump
                T (float): _description_

            Returns:
                float: _description_
            """
            t = tf.maximum(T, 1e-4) / tau
            return beta0 + beta1 * (1.0 - tf.exp(-t)) / t + beta2 * ((1 - tf.exp(-t)) / t - tf.exp(-(t)))


class ConstantRate(interfaces.FactoryObject):
    def __init__(self, rate: float):
        """Continuously compounded flat rate object that can be used  in conjunction with :class:`rivapy.marketdata.DiscountCurveParametrized`.

        Args:
            rate (float): The constant rate.

        """
        self.rate = rate

    def _to_dict(self) -> dict:
        return {"rate": self.rate}

    @staticmethod
    def _create_sample(n_samples: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        result = []
        for i in range(n_samples):
            result.append(ConstantRate(rate=np.random.uniform(-0.005, 0.1)))
        return result

    def __call__(self, t: float):
        return self.rate


class LinearRate(interfaces.FactoryObject):
    def __init__(self, shortterm_rate: float, longterm_rate: float, max_maturity: float = 10.0):
        """Continuously compounded linearly interpolated rate object that can be used  in conjunction with :class:`rivapy.marketdata.DiscountCurveParametrized`.

        Args:
            shortterm_rate (float): The short term rate.
            longterm_rate (float): the longterm rate.
            max_maturity (float): AFer this timepoint constant extrapolation is applied.
        """
        self.shortterm_rate = shortterm_rate
        self.longterm_rate = longterm_rate
        self.max_maturity = max_maturity
        self._coeff = (self.longterm_rate - self.shortterm_rate) / (self.max_maturity)

    @staticmethod
    def _create_sample(n_samples: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        result = []
        for i in range(n_samples):
            shortterm_rate = np.random.uniform(-0.005, 0.07)
            longterm_rate = shortterm_rate + np.random.uniform(0.0025, 0.09)
            result.append(LinearRate(shortterm_rate=shortterm_rate, longterm_rate=longterm_rate))
        return result

    def _to_dict(self) -> dict:
        return {"shortterm_rate": self.shortterm_rate, "longterm_rate": self.longterm_rate, "max_maturity": self.max_maturity}

    def __call__(self, t: float):
        if t < self.max_maturity:
            return self.shortterm_rate + self._coeff * t
        return self.longterm_rate


class NelsonSiegelSvensson(NelsonSiegel):
    def __init__(self, beta0: float, beta1: float, beta2: float, beta3: float, tau: float, tau2: float):
        super().__init__(beta0, beta1, beta2, tau)
        self.beta3 = beta3
        self.tau2 = tau2

    def _to_dict(self) -> dict:
        tmp = super()._to_dict()
        tmp.update({"beta3": self.beta3, "tau2": self.tau2})
        return tmp

    def __call__(self, t: float):
        return NelsonSiegelSvensson.compute(self.beta0, self.beta1, self.beta2, self.beta3, self.tau, self.tau2, t)

    @staticmethod
    def compute(beta0, beta1, beta2, beta3, tau, tau2, T):
        t = np.maximum(T, 1e-4) / tau2
        return NelsonSiegel.compute(beta0, beta1, beta2, tau, T) + beta3 * ((1 - np.exp(-t)) / t - np.exp(-(t)))


class DiscountCurveComposition(interfaces.FactoryObject):
    def __init__(self, a, b, c):
        # check if all discount curves have the same daycounter, otherwise exception
        if isinstance(a, dict):
            a = _create(a)
        if isinstance(b, dict):
            b = _create(b)
        if isinstance(c, dict):
            c = _create(c)
        dc = set()
        for k in [a, b, c]:
            if hasattr(k, "daycounter"):
                dc.add(k.daycounter)
        if len(dc) > 1:
            raise Exception("All curves must have same daycounter.")
        if len(dc) > 0:
            self.daycounter = dc.pop()
        else:
            self.daycounter = DayCounterType.Act365Fixed.value
        self._dc = DayCounter(self.daycounter)
        self.a = a
        if not hasattr(a, "value"):
            self.a = DiscountCurveParametrized("", datetime(1980, 1, 1), ConstantRate(a), self.daycounter)
        self.b = b
        if not hasattr(b, "value"):
            self.b = DiscountCurveParametrized("", datetime(1980, 1, 1), ConstantRate(b), self.daycounter)
        self.c = c
        if not hasattr(c, "value"):
            self.c = DiscountCurveParametrized("", datetime(1980, 1, 1), ConstantRate(c), self.daycounter)

    def _to_dict(self) -> dict:
        if hasattr(self.a, "to_dict"):
            a = self.a.to_dict()
        else:
            a = self.a
        if hasattr(self.b, "to_dict"):
            b = self.b.to_dict()
        else:
            b = self.b
        if hasattr(self.c, "to_dict"):
            c = self.c.to_dict()
        else:
            c = self.c
        return {"a": a, "b": b, "c": c}

    @staticmethod
    def _create_sample(n_samples: int, seed: int = None, refdate: Union[datetime, date] = None, parametrization_type=NelsonSiegel) -> list:
        curves = DiscountCurveParametrized._create_sample(n_samples, seed, refdate, parametrization_type)
        results = []
        for c in curves:
            results.append(c + 0.001)
        return results

    def value(self, refdate: Union[date, datetime], d: Union[date, datetime]) -> float:
        r = self.value_rate(refdate, d)
        yf = self._dc.yf(refdate, d)
        return np.exp(-r * yf)

    def value_rate(self, refdate: Union[date, datetime], d: Union[date, datetime]) -> float:
        return self.a.value_rate(refdate, d) * self.b.value_rate(refdate, d) + self.c.value_rate(refdate, d)

    def __mul__(self, other):
        # TODO unittests
        return DiscountCurveComposition(self, other, 0.0)

    def __rmul__(self, other):
        return DiscountCurveComposition(self, other, 0.0)

    def __add__(self, other):
        return DiscountCurveComposition(self, 1.0, other)

    def __radd__(self, other):
        return DiscountCurveComposition(self, 1.0, other)


class DiscountCurveParametrized(interfaces.FactoryObject):
    def __init__(
        self,
        obj_id: str,
        refdate: Union[datetime, date],
        rate_parametrization,  #: Callable[[float], float],
        daycounter: Union[DayCounterType, str] = DayCounterType.Act365Fixed,
    ):
        """_summary_

        Args:
            obj_id (str): _description_
            refdate (Union[datetime, date]): _description_
            rate_parametrization (Callable[[float], float]): _description_
            daycounter (Union[DayCounterType, str], optional): _description_. Defaults to DayCounterType.Act365Fixed.
        """
        if isinstance(refdate, datetime):
            self.refdate = refdate
        else:
            self.refdate = datetime(refdate, 0, 0, 0)

        self.daycounter = DayCounterType.to_string(daycounter)
        self._dc = DayCounter(self.daycounter)
        self.obj_id = obj_id
        if isinstance(rate_parametrization, dict):  # if schedule is a dict we try to create it from factory
            self.rate_parametrization = _create(rate_parametrization)
        else:
            self.rate_parametrization = rate_parametrization

    def _to_dict(self) -> dict:
        try:
            parametrization = self.rate_parametrization.to_dict()
        except Exception as e:
            raise Exception("Missing implementation of to_dict() in parametrization of type " + type(self.rate_parametrization).__name__)
        return {"obj_id": self.obj_id, "refdate": self.refdate, "rate_parametrization": parametrization}

    def value(self, refdate: Union[date, datetime], d: Union[date, datetime]) -> float:
        """Return discount factor for a given date

        Args:
            refdate (Union[date, datetime]): The reference date. If the reference date is in the future (compared to the curves reference date), the forward discount factor will be returned.
            d (Union[date, datetime]): The date for which the discount factor will be returned

        Returns:
            float: discount factor
        """
        if not isinstance(refdate, datetime):
            refdate = datetime(refdate, 0, 0, 0)
        if not isinstance(d, datetime):
            d = datetime(d, 0, 0, 0)
        if refdate < self.refdate:
            raise Exception("The given reference date is before the curves reference date.")
        yf = self._dc.yf(refdate, d)
        return np.exp(-self.rate_parametrization(yf) * yf)

    def value_rate(self, refdate: Union[date, datetime], d: Union[date, datetime]) -> float:
        """Return the continuous rate for a given date

        Args:
            refdate (Union[date, datetime]): The reference date. If the reference date is in the future (compared to the curves reference date), the forward discount factor will be returned.
            d (Union[date, datetime]): The date for which the discount factor will be returned

        Returns:
            float: continuous rate
        """
        if not isinstance(refdate, datetime):
            refdate = datetime(refdate, 0, 0, 0)
        if not isinstance(d, datetime):
            d = datetime(d, 0, 0, 0)
        if refdate < self.refdate:
            raise Exception("The given reference date is before the curves reference date.")
        yf = self._dc.yf(refdate, d)
        return self.rate_parametrization(yf)

    @staticmethod
    def _create_sample(n_samples: int, seed: int = None, refdate: Union[datetime, date] = None, parametrization_type=NelsonSiegel) -> list:
        if seed is not None:
            np.random.seed(seed)
        if refdate is None:
            refdate = datetime.now()
        parametrizations = parametrization_type._create_sample(n_samples)
        result = []
        for i, p in enumerate(parametrizations):
            result.append(DiscountCurveParametrized("DCP_" + str(i), refdate, p))
        return result

    def __mul__(self, other):
        return DiscountCurveComposition(self, other, 0.0)

    def __rmul__(self, other):
        return DiscountCurveComposition(self, other, 0.0)

    def __add__(self, other):
        return DiscountCurveComposition(self, 1.0, other)

    def __radd__(self, other):
        return DiscountCurveComposition(self, 1.0, other)


class EquityForwardCurve:
    def __init__(self, spot: float, funding_curve: DiscountCurve, borrow_curve: DiscountCurve, div_table):
        """Equity Forward Curve

        Args:

            spot (float): Current spot
            discount_curve (DiscountCurve): [description]
            funding_curve (DiscountCurve): [description]
            borrow_curve (DiscountCurve): [description]
            div_table (:class:`rivapy.marketdata.DividendTable`): [description]
        """
        self.spot = spot

        self.bc = borrow_curve
        self.fc = funding_curve
        self.div = div_table
        self._pyvacon_obj = None
        self.refdate = self.fc.refdate
        if self.bc is not None:
            if self.refdate < self.bc.refdate:
                self.refdate = self.bc.refdate

        if self.div is not None:
            if hasattr(self.div, "refdate"):
                if self.refdate < self.div.refdate:
                    self.refdate = self.div.refdate

    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            if hasattr(self.fc, "_get_pyvacon_obj"):
                fc = self.fc._get_pyvacon_obj()
            else:
                fc = self.fc

            if hasattr(self.bc, "_get_pyvacon_obj"):
                bc = self.bc._get_pyvacon_obj()
            else:
                bc = self.bc

            if hasattr(self.div, "_get_pyvacon_obj"):
                div = self.div._get_pyvacon_obj()
            else:
                div = self.div
            self._pyvacon_obj = _EquityForwardCurve(self.refdate, self.spot, fc, bc, div)

        return self._pyvacon_obj

    def value(self, refdate, expiry):
        return self._get_pyvacon_obj().value(refdate, expiry)

    def plot(self, days: int = 10, days_end: int = 10 * 365, **kwargs):
        """Plots the forward curve using matplotlibs plot function.

        Args:
            days (int, optional): The number of days between two plotted rates/discount factors. Defaults to 10.
            days_end (int. optional): Number of days when plotting will end. Defaults to 10*365 (10yr)
            **kwargs: optional arguments that will be directly passed to the matplotlib plto function
        """
        dates = [self.refdate + timedelta(days=i) for i in range(0, days_end, days)]
        values = [self.value(self.refdate, d) for d in dates]
        plt.plot(dates, values, **kwargs)
        plt.xlabel("expiry")
        plt.ylabel("forward value")


class BootstrapHazardCurve:
    def __init__(
        self, ref_date: datetime, trade_date: datetime, dc: DiscountCurve, RR: float, payment_dates: List[datetime], market_spreads: List[float]
    ):
        """[summary]

        Args:
            ref_date (datetime): [description]
            trade_date (datetime): [description]
            dc (DiscountCurve): [description]
            RR (float): [description]
            payment_dates (List[datetime]): [description]
            market_spreads (List[float]): [description]
        """

        self.ref_date = ref_date
        self.trade_date = trade_date
        self.dc = dc
        self.RR = RR
        self.payment_dates_bootstrapp = payment_dates
        self.market_spreads = market_spreads
        self._pyvacon_obj = None

    def par_spread(self, dc_survival, maturity_date, payment_dates: List[datetime]):
        integration_step = relativedelta.relativedelta(days=365)
        premium_period_start = self.ref_date
        prev_date = self.ref_date
        current_date = min(prev_date + integration_step, maturity_date)
        dc_valuation_date = self.dc.value(self.ref_date, maturity_date)
        risk_adj_factor_protection = 0
        risk_adj_factor_premium = 0
        risk_adj_factor_accrued = 0

        while current_date <= maturity_date:
            default_prob = dc_survival.value(self.ref_date, prev_date) - dc_survival.value(self.ref_date, current_date)
            risk_adj_factor_protection += self.dc.value(self.ref_date, current_date) * default_prob
            prev_date = current_date
            current_date += integration_step

        if prev_date < maturity_date and current_date > maturity_date:
            default_prob = dc_survival.value(self.ref_date, prev_date) - dc_survival.value(self.ref_date, maturity_date)
            risk_adj_factor_protection += self.dc.value(self.ref_date, maturity_date) * default_prob

        for premium_payment in payment_dates:
            if premium_payment >= self.ref_date:
                period_length = ((premium_payment - premium_period_start).days) / 360
                survival_prob = (dc_survival.value(self.ref_date, premium_period_start) + dc_survival.value(self.ref_date, premium_payment)) / 2
                df = self.dc.value(self.ref_date, premium_payment)
                risk_adj_factor_premium += period_length * survival_prob * df
                default_prob = dc_survival.value(self.ref_date, premium_period_start) - dc_survival.value(self.ref_date, premium_payment)
                risk_adj_factor_accrued += period_length * default_prob * df
                premium_period_start = premium_payment

        PV_accrued = (1 / 2) * risk_adj_factor_accrued
        PV_premium = (1) * risk_adj_factor_premium
        PV_protection = ((1 - self.RR)) * risk_adj_factor_protection

        par_spread_i = (PV_protection) / ((PV_premium + PV_accrued))
        return par_spread_i

    def create_survival(self, dates: List[datetime], hazard_rates: List[float]):
        return _SurvivalCurve("survival_curve", self.refdate, dates, hazard_rates)

    def calibration_error(x, self, mkt_par_spread, ref_date, payment_dates, dates, hazard_rates):
        hazard_rates[-1] = x
        maturity_date = dates[-1]
        dc_surv = self.create_survival(ref_date, dates, hazard_rates)
        return mkt_par_spread - self.par_spread(dc_surv, maturity_date, payment_dates)

    def calibrate_hazard_rate(self):
        sc_dates = [self.ref_date]
        hazard_rates = [0.0]
        for i in range(len(self.payment_dates_bootstrapp)):
            payment_dates_iter = self.payment_dates_bootstrapp[i]
            mkt_par_spread_iter = self.market_spreads[i]
            sc_dates.append(payment_dates_iter[-1])
            hazard_rates.append(hazard_rates[-1])
            sol = scipy.optimize.root_scalar(
                self.calibration_error,
                args=(mkt_par_spread_iter, self.ref_date, payment_dates_iter, sc_dates, hazard_rates),
                method="brentq",
                bracket=[0, 3],
                xtol=1e-8,
                rtol=1e-8,
            )
            hazard_rates[-1] = sol.root
        return hazard_rates, sc_dates  # self.create_survival(self.ref_date, sc_dates, hazard_rates)#.value, hazard_rates

    # def hazard_rates(self):
    #     #hazard_rates_value=[]
    #     hazard_rates_value=self.calibrate_hazard_rate()
    #     return self.hazard_rates_value

    # def value(self, refdate: Union[date, datetime], d: Union[date, datetime])->float:
    #     """Return discount factor for a given date

    #     Args:
    #         refdate (Union[date, datetime]): The reference date. If the reference date is in the future (compared to the curves reference date), the forward discount factor will be returned.
    #         d (Union[date, datetime]): The date for which the discount factor will be returned

    #     Returns:
    #         float: discount factor
    #     """
    #     #if not isinstance(refdate, datetime):
    #     #    refdate = datetime(refdate,0,0,0)
    #     #if not isinstance(d, datetime):
    #     #    d = datetime(d,0,0,0)
    #     #if refdate < self.refdate:
    #     #    raise Exception('The given reference date is before the curves reference date.')
    #     return self._get_pyvacon_obj().value(refdate, d)

    # def _get_pyvacon_obj(self):
    #     if self._pyvacon_obj is None:
    #         self._pyvacon_obj = _SurvivalCurve('survival_curve', self.refdate,
    #                                         self.calibrate_hazard_rate[1], self.calibrate_hazard_rate[0])
    #     return self._pyvacon_obj


class PowerPriceForwardCurve:
    def __init__(
        self, refdate: Union[datetime, date], start: datetime, end: datetime, values: np.ndarray, freq: str = "1H", tz: str = None, id: str = None
    ):
        """Simple forward curve for power.

        Args:
            refdate (Union[datetime, date]): Reference date of curve
            start (dt.datetime): Start of forward curve datetimepoints (including this timepoint).
                        end (dt.datetime): End of forad curve datetimepoints (excluding this timepoint).
            values (np.ndarray): One dimensional array holding the price for each datetimepint in the curve. The method value will raise an exception if the number of values is not equal to the number of datetimepoints.
                        freq (str, optional): Frequency of timepoints. Defaults to '1H'. See documentation for pandas.date_range for further details on freq.
                        tz (str or tzinfo): Time zone name for returning localized datetime points, for example ‘Asia/Hong_Kong’.
                                                                By default, the resulting datetime points are timezone-naive. See documentation for pandas.date_range for further details on tz.
            id (str): Identifier for the curve. It has no impact on the valuation functionality. If None, a uuid will be generated. Defaults to None.
        """
        self.id = id
        if id is None:
            self.id = "PFC/" + str(datetime.now())
        self.refdate = refdate
        self.start = start
        self.end = end
        self.freq = freq
        self.tz = tz
        self.values = values
        # timegrid used to compute prices for a certain schedule
        self._tg = None
        self._df = (
            pd.DataFrame(
                {"dates": pd.date_range(self.start, self.end, freq=self.freq, tz=self.tz, inclusive="left").to_pydatetime(), "values": self.values}
            )
            .set_index(["dates"])
            .sort_index()
        )

    def value(self, refdate: Union[date, datetime], schedule) -> np.ndarray:
        if self._tg is None:
            self._tg = pd.DataFrame(
                {"dates": pd.date_range(self.start, self.end, freq=self.freq, tz=self.tz, inclusive="left").to_pydatetime(), "values": self.values}
            ).reset_index()
            if self._tg.shape[0] != self.values.shape[0]:
                raise Exception(
                    "The number of dates ("
                    + str(self._tg.shape[0])
                    + ") does not equal number of values ("
                    + str(self.values.shape[0])
                    + ") in forward curve."
                )
        tg = self._tg[(self._tg.dates >= schedule.start) & (self._tg.dates < schedule.end)].set_index("dates")
        _schedule = pd.DataFrame({"dates": schedule.get_schedule(refdate)})
        tg = _schedule.join(tg, on="dates")
        # tg = tg[tg['dates']>=refdate]
        if tg["index"].isna().sum() > 0:
            raise Exception("There are " + str(tg["index"].isna().sum()) + " dates in the schedule not covered by the forward curve.")
        return self.values[tg["index"].values]

    def get_df(self) -> pd.DataFrame:
        return self._df


class EnergyPriceForwardCurve:
    """Energy Price Forward Curve object.
    It is recommended to initialze this object via the class methods ``from_existing_pfc``, ``from_existing_shape`` or ``from_scratch``.

    Args:
        id (_type_): ID for the PFC object
        refdate (Union[datetime, date]): Reference date
        pfc (pd.DataFrame, optional): This object can be initialized with an existing pfc. Defaults to None.
    """

    def __init__(self, id, refdate: Union[datetime, date], pfc: pd.DataFrame = None, **kwargs):
        self.id = id
        if id is None:
            self.id = "PFC/" + str(datetime.now())
        self.refdate = refdate

        self._pfc = pfc

        self._pfc_shape: pd.DataFrame = kwargs.get("pfc_shape", None)

        self._apply_schedule: SimpleSchedule = kwargs.get("apply_schedule", None)
        self._pfc_shaper: PFCShaper = kwargs.get("pfc_shaper", None)

        list(map(lambda x: EnergyPriceForwardCurve._validate_dataframes(x), [self._pfc, self._pfc_shape]))

        self._future_contracts: List[EnergyFutureSpecifications] = kwargs.get("future_contracts", None)

        if self._pfc is None and self._pfc_shape is None and self._pfc_shaper is None:
            raise ValueError("No values provided for the arguments pfc, pfc_shape and pfc_shaper!")

    @staticmethod
    def _validate_dataframes(dataframe: Optional[pd.DataFrame]):
        if dataframe is not None:
            validators._check_pandas_index_for_datetime(dataframe)

    @classmethod
    def from_existing_pfc(cls, id, refdate: Union[datetime, date], pfc: pd.DataFrame) -> "EnergyPriceForwardCurve":
        """Initialization of the ``EnergyPriceForwardCurve`` given an existing PFC.

        Args:
            id (_type_): ID for the PFC object
            refdate (Union[datetime, date]): Reference Date
            pfc (pd.DataFrame): Existing Pfc

        Returns:
            EnergyPriceForwardCurve: ``EnergyPriceForwardCurve`` object
        """
        instance = cls(id=id, refdate=refdate, pfc=pfc)
        return instance

    @classmethod
    def from_existing_shape(
        cls, id, refdate: Union[datetime, date], pfc_shape: pd.DataFrame, future_contracts: List[EnergyFutureSpecifications]
    ) -> "EnergyPriceForwardCurve":
        """Initialization of the ``EnergyPriceForwardCurve`` given an existing PFC shape. The shape is then shifted in order to match the future contracts defined in the ``future_contracts`` list.


        Args:
            id (_type_): ID for the PFC object
            refdate (Union[datetime, date]): Reference Date
            pfc_shape (pd.DataFrame): Existing PFC shape
            future_contracts (List[EnergyFutureSpecifications]): List of future contracts (``EnergyFutureSpecifications`` objects)

        Returns:
            EnergyPriceForwardCurve: ``EnergyPriceForwardCurve`` object
        """
        instance = cls(id=id, refdate=refdate, pfc_shape=pfc_shape, future_contracts=future_contracts)
        instance._shift_shape()
        return instance

    @classmethod
    def from_scratch(
        cls,
        id,
        refdate: Union[datetime, date],
        apply_schedule: SimpleSchedule,
        pfc_shaper: PFCShaper,
        future_contracts: List[EnergyFutureSpecifications],
    ) -> "EnergyPriceForwardCurve":
        """Initialization of the ``EnergyPriceForwardCurve`` from scratch. First a shape is created using the ``pfc_shaper``. Afterwards, shape is shifted in order to match the future contracts defined in the ``future_contracts`` list.

        Args:
            id (_type_): ID for the PFC object
            refdate (Union[datetime, date]): Reference Date
            apply_schedule (SimpleSchedule): Schedule to apply the ``pfc_shaper`` on, in order to obtain shape values for future time points
            pfc_shaper (PFCShaper): PFC shaper
            future_contracts (List[EnergyFutureSpecifications]): List of future contracts (``EnergyFutureSpecifications`` objects)

        Returns:
            EnergyPriceForwardCurve: ``EnergyPriceForwardCurve`` object
        """
        instance = cls(id=id, refdate=refdate, pfc_shaper=pfc_shaper, future_contracts=future_contracts, apply_schedule=apply_schedule)
        instance._create_shape()
        instance._shift_shape()
        return instance

    def __validate_contracts_frequency(self):
        """Checks if all contracts in ``self._future_contracts`` have the sample schedule frequency."""
        frequencies_contracts = defaultdict(list)
        for future_contracts in self._future_contracts:
            frequencies_contracts[future_contracts.schedule.freq].append((future_contracts.schedule.__class__.__name__, future_contracts.name))

        if len(list(frequencies_contracts.keys())) > 1:
            raise ValueError(
                f"Found different contract frequencies: {frequencies_contracts}.\n Please provide uniform frequencies for the elements in the `future_contract` dictionary!"
            )

    def __get_offpeak_contracts(
        self, base_contracts: List[EnergyFutureSpecifications], peak_contracts: List[EnergyFutureSpecifications]
    ) -> List[EnergyFutureSpecifications]:
        """In cases where base and peak contracts are part of the ``self._future_contracts``, offpeak contracts need to be decuted from these two in order to shift the shape properly.

        Args:
            base_contracts (List[EnergyFutureSpecifications]): List of base contracts
            peak_contracts (List[EnergyFutureSpecifications]): List of peak contracts

        Returns:
            List[EnergyFutureSpecifications]: List of offpeak contracts
        """
        offpeak_contracts = []

        # iterate over each combination of base and peak contracts
        for base_contract_spec in base_contracts:
            n_base = len(base_contract_spec.get_schedule())
            for peak_contract_spec in peak_contracts:
                # match both by the start and end dates of their respective schedule
                if base_contract_spec.get_start_end() == peak_contract_spec.get_start_end():
                    # if both match, an offpeak contract can be created from these two
                    offpeak_name = f"offpeak_{base_contract_spec.name}&{peak_contract_spec.name}"
                    n_peak = len(peak_contract_spec.get_schedule())
                    offpeak_price = (
                        n_base / (n_base - n_peak) * base_contract_spec.get_price() - n_peak / (n_base - n_peak) * peak_contract_spec.get_price()
                    )
                    offpeak_contracts.append(
                        EnergyFutureSpecifications(
                            schedule=OffPeakSchedule(start=base_contract_spec.get_start(), end=base_contract_spec.get_end()),
                            price=offpeak_price,
                            name=offpeak_name,
                        )
                    )
                    break

        return offpeak_contracts

    def _shift_shape(self):
        """Shifts the shape to match the future contracts defined in the ``self._future_contracts`` list."""
        self.__validate_contracts_frequency()

        base_contracts, peak_contracts = [
            [fc for fc in self._future_contracts if fc.schedule.__class__._name == schedule_type] for schedule_type in (ets.BASE, ets.PEAK)
        ]

        # if base and peak contracts both exist, offpeak contracts are computed
        if (len(base_contracts) > 0) and (len(peak_contracts) > 0):
            shifted_pfc = []
            offpeak_contracts = self.__get_offpeak_contracts(base_contracts=base_contracts, peak_contracts=peak_contracts)

            # shift offpeak and peak separately
            for contracts in [offpeak_contracts, peak_contracts]:
                shifting_datetimes = np.sort(np.unique(np.concatenate([contract.get_schedule() for contract in contracts])))
                _pfc_shape = self._pfc_shape.loc[shifting_datetimes, :]
                pfc_shifter = PFCShifter(shape=_pfc_shape, contracts=contracts)
                shifted_pfc.append(pfc_shifter.compute())

            # combine offpeak and peak shifts
            shifted_pfc = pd.concat(shifted_pfc, axis=0)
            self._pfc = shifted_pfc.sort_index(ascending=True)

        else:
            # if either base of peak exists, the shift can be directly performed
            pfc_shifter = PFCShifter(shape=self._pfc_shape, contracts=self._future_contracts)
            self._pfc = pfc_shifter.compute()

    def _create_shape(self):
        """Creates a shape using the ``self._pfc_shaper`` model"""
        self._pfc_shaper.calibrate()
        self._pfc_shape = self._pfc_shaper.apply(self._apply_schedule)

    def get_pfc(self) -> pd.DataFrame:
        """Returns the PFC

        Returns:
            pd.DataFrame: PFC
        """
        return self._pfc
