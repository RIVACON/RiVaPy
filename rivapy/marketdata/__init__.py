# from pyvacon.marketdata.analytics_classes import *  # TODO: Clarify why this is necessary for imports in pricing_data.
# from pyvacon.marketdata import analytics_classes
# __all__ = ['analytics_classes', 'bootstrapping']
import abc
import numpy as np

# from pyvacon.pyvacon_swig import EquityOptionQuoteTable
from rivapy import enums
from typing import List, Union, Tuple
from rivapy import _pyvacon_available
from scipy.optimize import least_squares
from rivapy.marketdata.curves import *
from rivapy.marketdata.factory import _factory

if _pyvacon_available:
    import pyvacon.finance.marketdata as _mkt_data

    InflationIndexForwardCurve = _mkt_data.InflationIndexForwardCurve
    SurvivalCurve = _mkt_data.SurvivalCurve
    DatedCurve = _mkt_data.DatedCurve
    EquityOptionQuoteTable = _mkt_data.EquityOptionQuoteTable
    import pyvacon.finance.marketdata as _mkt_data
    import pyvacon.finance.utils as _utils
    import pyvacon.finance.pricing as _pricing

    # DividendTable = _mkt_data.DividendTable
else:

    class SurvivalCurve:
        def __init__(self):
            raise Exception("Up to now only implemented in pyvacon that has not been installed.")


class DividendTable:
    def __init__(
        self,
        id: str,
        refdate: datetime,
        ex_dates: List[datetime],
        pay_dates: List[datetime],
        div_yield: List[float],
        div_cash: List[float],
        tax_factors: List[float],
    ):
        """[summary]

        Args:
            id (str): [description]
            refdate (datetime): [description]
            ex_dates (List[datetime]): [description]
            pay_dates (List[datetime]): [description]
            div_yield (List[float]): [description]
            div_cash (List[float]): [description]
            tax_factors (List[float]): [description]

        Yields:
            [type]: [description]
        """
        self.id = id
        self.refdate = refdate
        self.ex_dates = ex_dates
        self.pay_dates = pay_dates
        self.div_yield = div_yield
        self.div_cash = div_cash
        self.tax_factors = tax_factors
        self._pyvacon_obj = None

    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _mkt_data.DividendTable(
                self.id, self.refdate, self.ex_dates, self.div_yield, self.div_cash, self.tax_factors, self.pay_dates
            )
        return self._pyvacon_obj


class _VolatilityParametrizationExpiry:
    def __init__(self, expiries: List[float], params_at_expiry: List[Tuple]):
        self.n_params = len(params_at_expiry[0])
        self.expiries = np.array(expiries)
        self._x = self._get_x(params_at_expiry)

    def get_params_at_expiry(self, expiry: int) -> np.array:
        """Get parameters for given expiry.

        Args:
            expiry (int): Position in expiry list.

        Returns:
            np.array: Parameter Tuple for given expiry.
        """
        return self._x[self.n_params * expiry : self.n_params * (expiry + 1)]

    def calc_implied_vol(self, ttm, strike):
        """Calculate implied volatility for given expiry and strike

        Args:
            ttm ([float]): Expiry.
            strike ([float]): Strike.

        Returns:
            [float]: Implied volatility.
        """
        i = np.searchsorted(self.expiries, ttm)
        if i == 0 or i == self.expiries.shape[0]:
            if i == self.expiries.shape[0]:
                i -= 1
            return np.sqrt(self._calc_implied_vol_at_expiry(self.get_params_at_expiry(i), ttm, strike))
        w0 = self._calc_implied_vol_at_expiry(self.get_params_at_expiry(i - 1), self.expiries[i - 1], strike)
        w1 = self._calc_implied_vol_at_expiry(self.get_params_at_expiry(i), self.expiries[i], strike)
        # linear n total variance
        delta_t = self.expiries[i] - self.expiries[i - 1]
        w = ((self.expiries[i] - ttm) * w0 + (ttm - self.expiries[i - 1]) * w1) / delta_t
        return np.sqrt(w / ttm)

    @abc.abstractmethod
    def _calc_implied_vol_at_expiry(self, params, ttm: float, strike: float):
        pass

    def _get_x(self, params) -> np.array:
        x = np.empty(len(params) * self.n_params)
        j = 0
        for i in range(len(params)):
            for k in range(self.n_params):
                x[j] = params[i][k]
                j += 1
        return x

    def _set_param(self, x) -> np.array:
        self._x = x

    def calibrate_params(self, quotes: pd.DataFrame, **kwargs):
        """Calibrate parameters to given implied volatility quotes.

        Args:
            quotes (pd.DataFrame): pd.DataFrame with columns EXPIRY as year fraction, STRIKE asm moneyness, BID_IV, ASK_IV.
        """

        def cost_function(x):
            self._set_param(x)
            quotes["VOLS"] = [self.calc_implied_vol(expiry, strike) for expiry, strike in zip(quotes["EXPIRY"], quotes["STRIKE"])]
            quotes["DIST_ASK"] = [max(vol - ask, 0) for ask, vol in zip(quotes["ASK_IV"], quotes["VOLS"])]
            quotes["DIST_BID"] = [max(bid - vol, 0) for bid, vol in zip(quotes["BID_IV"], quotes["VOLS"])]
            quotes["DIST_TOTAL"] = quotes["DIST_ASK"] + quotes["DIST_BID"]
            return np.copy(quotes["DIST_TOTAL"].values)

        if kwargs is None:
            kwargs = {"method": "lm"}
        result = least_squares(cost_function, self._x, **kwargs)

        return result.x


class VolatilityParametrizationFlat:
    def __init__(self, vol: float):
        """Flat volatility parametrization

        Args:
            vol (float): Constant volatility.
        """
        self.vol = vol
        self._pyvacon_obj = None

    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _mkt_data.VolatilityParametrizationFlat(self.vol)
        return self._pyvacon_obj


class VolatilityParametrizationTerm:
    def __init__(self, expiries: List[float], fwd_atm_vols: List[float]):
        """Term volatility parametrization

        Args:
            expiries (List[float]): List of expiries (sorted from nearest to farest).
            fwd_atm_vols (List[float]): List of at-the-money volatilities.
        """
        self.expiries = expiries
        self.fwd_atm_vols = fwd_atm_vols
        self._pyvacon_obj = None

    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _mkt_data.VolatilityParametrizationTerm(self.expiries, self.fwd_atm_vols)
        return self._pyvacon_obj


class VolatilityParametrizationSVI(_VolatilityParametrizationExpiry):
    def __init__(self, expiries: List[float], svi_params: List[Tuple]):
        """Raw SVI parametrization (definition 3.1 in  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2033323)

        .. math::
            w(k) = a + b(\\rho (k-m) + \\sqrt{(k-m)^2+\\sigma^2 })

        Args:
            expiries (List[float]): List of expiries (sorted from nearest to farest).
            svi_params (List[Tuple]): List of SVI parameters (one Tuple for each expiry). Tuple in the order (a, b, rho, m, sigma).

        """
        super().__init__(expiries, svi_params)

    def _calc_implied_vol_at_expiry(self, params: List[float], ttm: float, k: float):
        return params[0] + params[1] * (params[2] * (np.log(k) - params[3]) + np.sqrt((np.log(k) - params[3]) ** 2 + params[4] ** 2))


class VolatilityParametrizationSSVI:
    def __init__(self, expiries: List[float], fwd_atm_vols: List[float], rho: float, eta: float, gamma: float):
        """SSVI volatility parametrization
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2033323

        Args:
            expiries (List[float]): List of expiries (sorted from nearest to farest).
            fwd_atm_vols (List[float]): List of at-the-money volatilities.
            rho (float): Responsible for the skewness of the volatility surface.
            eta (float): Responsible for the curvature.
            gamma (float): Responsible for the "rate of decay".

        """
        self.expiries = expiries
        # self.fwd_atm_vols = fwd_atm_vols
        # self.rho = rho
        # self.eta = eta
        # self.gamma = gamma
        self._pyvacon_obj = None

        self._x = self._get_x(fwd_atm_vols, rho, eta, gamma)
        self.n_fwd_atm_vols = len(fwd_atm_vols)

    def calc_implied_vol(self, ttm, strike):
        """Calculate implied volatility for given expiry and strike

        Args:
            ttm ([float]): Expiry.
            strike ([float]): Strike.

        Returns:
            [float]: Implied volatility.
        """
        return self._get_pyvacon_obj().calcImpliedVol(ttm, strike)

    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            # self._pyvacon_obj = _mkt_data.VolatilityParametrizationSSVI(self.expiries, self.fwd_atm_vols, self.rho, self.eta, self.gamma)
            self._pyvacon_obj = _mkt_data.VolatilityParametrizationSSVI(
                self.expiries, self._x[: self.n_fwd_atm_vols], self._x[-3], self._x[-2], self._x[-1]
            )
        return self._pyvacon_obj

    def _get_x(self, fwd_atm_vols, rho, eta, gamma) -> np.array:
        x = np.empty(len(fwd_atm_vols) + 3)
        j = 0
        for i in range(len(fwd_atm_vols)):
            x[i] = fwd_atm_vols[i]
        x[i + 1] = rho
        x[i + 2] = eta
        x[i + 3] = gamma

        return x

    def _set_param(self, x) -> np.array:
        self._x = x
        self._pyvacon_obj = None

    def get_rho(self):
        return self._x[-3]

    def get_eta(self):
        return self._x[-2]

    def get_gamma(self):
        return self._x[-1]

    def get_fwd_atm_vols(self):
        return self._x[: self.n_fwd_atm_vols]


class VolatilityParametrizationSABR(_VolatilityParametrizationExpiry):
    def __init__(self, expiries: List[float], sabr_params: List[Tuple]):
        """SABR parametrization
        https://bsic.it/sabr-stochastic-volatility-model-volatility-smile/

        The SABR model assumes that the forward rate and the instantaneous volatility are driven by two correlated Brownian motions:

        .. math::
            df_t = \\alpha_t f_t^\\beta dW_t^1
        .. math::
            d\\alpha_t = \\nu\\alpha_t dW_t^2
        .. math::
            E\\bigl[dW_t^1 dW_T^2\\bigr] = \\rho dt

        The expression that the implied volatility must satisfy is

        .. math::
            \\sigma_B(K,f) = \\frac{\\alpha\\biggl\{1+\\biggl[\\frac {(1-\\beta)^2}{24}\\frac {\\alpha^2}{(fK)^{1-\\beta}}+\\frac {1}{4}\\frac {\\rho\\beta\\nu\\alpha}{(FK)^{(1-\\beta)/2}}+\\frac {2-3\\rho^2}{24}\\nu^2\\biggr ]T\\biggr \\}}{(fK)^{(1-\\beta)/2}\\biggl[1+\\frac {(1-\\beta)^2}{24}{ln}^2\\frac {f}{K}+\\frac {(1-\\beta)^4}{1920}{ln}^4\\frac {f}{K}\\biggr]} \\frac {z}{\\chi(z)}

        .. math::
            z = \\frac {\\nu }{\\alpha }(fK)^{(1-\\beta )/2} ln \\frac {f}{K}

        .. math::
            \\chi(z) = ln \\bigl[ \\frac {\\sqrt{1-2 \\rho z+z^2}+z-\\rho }{1- \\rho} \\bigr]

        When :math:`f = K` (for ATM options), the above formula for implied volatility simplifies to:

        .. math::
            \\sigma_{ATM} = \\sigma_B(f,f)=\\frac{\\alpha\\biggl\{1+\\biggl[\\frac{(1-\\beta)^2}{24}\\frac{\\alpha^2}{f^{2-2\\beta}}+\\frac{1}{4}\\frac{\\rho\\beta\\nu\\alpha}{f^{1-\\beta}}\\frac{2-3\\rho^2}{24}\\nu^2\\biggr]T\\biggr\}}{f^{1-\\beta}}

        where

        > :math:`\\alpha` is the instantaneous vol;

        > :math:`\\nu` is the vol of vol;

        > :math:`\\rho` is the correlation between the Brownian motions driving the forward rate and the instantaneous vol;

        > :math:`\\beta` is the CEV component for forward rate (determines shape of forward rates, leverage effect and backbond of ATM vol).

        Args:
            expiries (List[float]): List of expiries (sorted from nearest to farest).
            sabr_params (List[Tuple]): List of SABR parameters (one Tuple for each expiry). Tuple in the order (alpha, nu, beta, rho).
        """

        super().__init__(expiries, sabr_params)

    def _calc_implied_vol_at_expiry(self, params: List[float], ttm: float, strike: float):
        K = strike
        alpha = params[0]
        nu = params[1]
        beta = params[2]
        rho = params[3]
        f = 1

        zeta = nu / alpha * (f * K) ** ((1 - beta) / 2) * np.log(f / K)
        chi_zeta = np.log((np.sqrt(1 - 2 * rho * zeta + zeta**2) + zeta - rho) / (1 - rho))

        if f == K:
            sigma = (
                alpha
                * (
                    1
                    + (
                        (1 - beta) ** 2 / 24 * alpha**2 / f ** (2 - 2 * beta)
                        + 1 / 4 * rho * beta * nu * alpha / f ** (1 - beta)
                        + (2 - 3 * rho**2) / 24 * nu**2
                    )
                    * ttm
                )
                / f ** (1 - beta)
            )

        else:
            sigma = (
                alpha
                * (
                    1
                    + (
                        (1 - beta) ** 2 / 24 * alpha**2 / (f * K) ** (1 - beta)
                        + 1 / 4 * rho * beta * nu * alpha / (f * K) ** ((1 - beta) / 2)
                        + (2 - 3 * rho**2) / 24 * nu**2
                    )
                    * ttm
                )
                / (f * K) ** ((1 - beta) / 2)
                * (1 + (1 - beta) ** 2 / 24 * np.log(f / K) ** 2 + (1 - beta) ** 4 / 1920 * np.log(f / K) ** 4)
                * zeta
                / chi_zeta
            )

        return sigma**2


class VolatilityGridParametrization:
    def __init__(self, expiries: np.array, strikes: np.ndarray, vols: np.ndarray):
        """Grid parametrization
        This parametrization stores a set of strike-vol grids for a given list of expiries and computes a volatility by
        - search for the neighboring expiries
        - apply a splien interpolation in each expiry to get the respective volatility
        - apply a linear interpolation (in total variance)

        Args:
            expiries (np.array): An array of the expiries.
            strikes (np.ndarray):
            vols (np.ndarray): Two dimensional array of volatilities where each row i contains the values for expiry i
        """
        self.expiries = expiries
        if len(strikes.shape) == 1:
            strikes = [strikes] * expiries.shape[0]
        self.strikes = strikes
        self.vols = vols
        self._pyvacon_obj = None

    def calc_implied_vol(self, ttm: float, strike: float):
        """Calculate implied volatility for given expiry and strike

        Args:
            ttm ([float]): Expiry.
            strike ([float]): Strike.

        Returns:
            [float]: Implied volatility.
        """
        return self._get_pyvacon_obj().calcImpliedVol(ttm, strike)

    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            vol_params = []
            self._pyvacon_obj = _mkt_data.VolatilityParametrizationTimeSlice(self.expiries, self.strikes, self.vols)
        return self._pyvacon_obj


class VolatilitySurface:
    @staticmethod
    def load(filename: str):
        return _mkt_data.VolatilitySurface.load(filename)

    @staticmethod
    def _create_param_pyvacon_obj(vol_param):
        if hasattr(vol_param, "_get_pyvacon_obj"):
            return vol_param._get_pyvacon_obj()
        if hasattr(vol_param, "expiries"):
            expiries = vol_param.expiries
        else:
            expiries = np.linspace(0.0, 4.0, 13, endpoint=True)
        strikes = np.linspace(0.4, 1.6, num=100)
        vols = np.empty((expiries.size, strikes.size))
        for i in range(expiries.size):
            for j in range(strikes.size):
                vols[i, j] = vol_param.calc_implied_vol(expiries[i], strikes[j])
        return VolatilityGridParametrization(expiries, strikes, vols)._get_pyvacon_obj()

    def __init__(self, id: str, refdate: datetime, forward_curve, daycounter, vol_param):
        """Volatility surface

        Args:
            id (str): Identifier (name) of the volatility surface.
            refdate (datetime): Valuation date.
            forward_curve (rivapy.market_data.EquityForwardCurve): Forward curve.
            daycounter (enums.DayCounterType): [description]
            vol_param ([VolatilityParametrizationFlat,VolatilityParametrizationTerm,VolatilityParametrizationSSVI]): Volatility parametrization.
        """
        self.id = id
        self.refdate = refdate
        self.forward_curve = forward_curve
        self.daycounter = daycounter
        self.vol_param = vol_param
        self._pyvacon_obj = None

    def _get_pyvacon_obj(self, fwd_curve=None):
        if self._pyvacon_obj is None:
            if fwd_curve is None:
                fwd_curve = self.forward_curve
            try:
                _py_fwd_curve = fwd_curve._get_pyvacon_obj()
            except:
                _py_fwd_curve = fwd_curve
            self._pyvacon_obj = _mkt_data.VolatilitySurface(
                self.id, self.refdate, _py_fwd_curve, self.daycounter.name, VolatilitySurface._create_param_pyvacon_obj(self.vol_param)
            )
        return self._pyvacon_obj

    def calc_implied_vol(self, expiry: datetime, strike: float, refdate: datetime = None, forward_curve=None) -> float:
        """Calculate implied volatility

        Args:
            refdate (datetime): Valuation date.
            expiry (datetime): Expiration date.
            strike (float): Strike price.

        Raises:
            Exception: [description]

        Returns:
            float: Implied volatility.
        """
        # convert strike into x_strike
        if refdate is None:
            refdate = self.forward_curve.refdate
        if forward_curve is None and self.forward_curve is None:
            raise Exception("Please specify a forward curve")
        vol = self._get_pyvacon_obj()
        if forward_curve is None:
            forward_curve = self.forward_curve
        elif self.forward_curve is not None:
            vol = _mkt_data.VolatilitySurface.createVolatilitySurfaceShiftedFwd(vol, forward_curve._get_pyvacon_obj())
        forward_curve_obj = forward_curve._get_pyvacon_obj()
        x_strike = _utils.computeXStrike(
            strike, forward_curve_obj.value(refdate, expiry), forward_curve_obj.discountedFutureCashDivs(refdate, expiry)
        )
        if x_strike < 0:
            raise Exception(
                f"The given strike value seems implausible compared to the discounted future cash dividends\
                ({forward_curve_obj.discountedFutureCashDivs(refdate, expiry)})."
            )
        return vol.calcImpliedVol(refdate, expiry, x_strike)

    @staticmethod
    def set_stickyness(vol_stickyness: enums.VolatilityStickyness):
        if vol_stickyness is enums.VolatilityStickyness.StickyXStrike:
            _pricing.GlobalSettings.setVolatilitySurfaceFwdStickyness(_pricing.VolatilitySurfaceFwdStickyness.Type.StickyXStrike)
        elif vol_stickyness is enums.VolatilityStickyness.StickyStrike:
            _pricing.GlobalSettings.setVolatilitySurfaceFwdStickyness(vol_stickyness)
        elif vol_stickyness is enums.VolatilityStickyness.StickyFwdMoneyness:
            _pricing.GlobalSettings.setVolatilitySurfaceFwdStickyness(_pricing.VolatilitySurfaceFwdStickyness.Type.StickyFwdMoneyness)
        elif vol_stickyness is enums.VolatilityStickyness.NONE:
            _pricing.GlobalSettings.setVolatilitySurfaceFwdStickyness(_pricing.VolatilitySurfaceFwdStickyness.Type.NONE)
        else:
            raise Exception("Error")


def _add_to_factory(cls):
    factory_entries = _factory()
    factory_entries[cls.__name__] = cls


_add_to_factory(NelsonSiegel)
_add_to_factory(NelsonSiegelSvensson)
_add_to_factory(ConstantRate)
_add_to_factory(LinearRate)
_add_to_factory(DiscountCurveParametrized)
_add_to_factory(DiscountCurveComposition)

if __name__ == "__main__":
    svi = VolatilityParametrizationSVI(
        expiries=np.array([1.0 / 365.0, 1.0]),
        svi_params=[
            (0.0001, 0.1, -0.5, 0.0, 0.0001),
            (0.2, 0.1, -0.5, 0.0, 0.4),
        ],
    )
    expiry = 1.0 / 365.0
    x_strike = 1.0
    svi.calc_implied_vol(expiry, x_strike)
