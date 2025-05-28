# -*- coding: utf-8 -*-


from datetime import date, datetime
from typing import  List as _List, Union as _Union
from rivapy.tools.interfaces import FactoryObject
from rivapy.tools.datetools import _date_to_datetime as datetime_to_date, _datetime_to_date_list as datetime_to_date_list


class PricingRequestBase(FactoryObject):

    _keys = ['theo_val', 'paths_udl', 'cf_expected', 'cf_paths']
    
    def __init__(self, **kwargs):
        #validate the given keys
        for key in kwargs.keys():
            if key not in PricingRequestBase._keys:
                raise ValueError("Unknown key '{}'".format(key))
            #set the attributes
            setattr(self, key, kwargs[key])

    def _to_dict(self) -> dict:
        return self.__dict__

class GreenPPAPricingRequest(PricingRequestBase):
    def __init__(self, theo_val: bool, cf_paths: bool=False, cf_expected: bool=False):
        """PricingRequest for Green PPA pricing.

        Args:
            price (bool): If True, the :term:`Theoretical Value` is calculated.
            cf_expected (bool, optional): If True, the paths of the simulated :term:`Cash flow` is returned. Defaults to False.
        """
        super().__init__(theo_val=theo_val, cf_expected=cf_expected, cf_paths=cf_paths)
   

class PricingRequest:
    # def __init__(self, calc_delta_gamma: bool = False, calc_cross_gamma: bool = True, calc_clean_price: bool = False,
    #              calc_rho: bool = False, rho_scale: float = 0.0001, calc_vega: bool = False, vega_scale: float = 0.01,
    #              calc_cross_volga: bool = False, calc_vanna: bool = False, calc_theta: bool = False,
    #              theta_scale: float = 1.0, calc_spline: bool = False, calc_grid_sizes: bool = False,
    #              calc_implied_volatility: bool = False, management_delta_limit: float = 0.01,
    #              calc_pricing_data: bool = False, calc_expected_cashflows: bool = False,
    #              calc_simulation_data: bool = False, calc_additional_information: bool = False,
    #              calc_z_spread: bool = False, calc_yield_to_maturity: bool = False, calc_convexity: bool = False,
    #              max_expected_cashflow_date: _Union[date, datetime] = None,
    #              cashflow_times: _List[_Union[date, datetime]] = None, calc_macaulay_duration: bool = False):
    #     """
    #     Defines the set of information, e.g. sensitivities, cashflow information, etc., to be calculated together with
    #     the instrument's price in the context of pricing. For some sensitivities the shift parameter for calculating
    #     the difference quotient (rather than the sensitivity in closed formula) is also stored in the pricing request
    #     object.
    #
    #     Args: TODO: Read carefully and double-check if these guessed descriptions are valid!
    #         calc_delta_gamma (bool, optional): Flag determining if the change of the instrument's price associated with
    #                                            small changes of the underlying's price shall be calculated to first (and
    #                                            second) order within the pricing routine. Defaults to False.
    #                                            TODO: also second order? delta_scale=?
    #         calc_cross_gamma (bool, optional): Flag determining if the change of the instrument's price associated with
    #                                            small changes of two underlyings' prices (both at fist order) shall be
    #                                            calculated within the pricing routine. Defaults to True.
    #                                            TODO: why defaults to True?
    #         calc_clean_price (bool, optional): Flag determining if the instrument's clean price shall be calculated
    #                                            additional to its dirty price within the pricing routine.
    #                                            Defaults to False.
    #         calc_rho (bool, optional): Flag determining if the change of the instrument's price associated with small
    #                                    changes of the zero rate curve shall be calculated to first order within the
    #                                    pricing routine. Defaults to False.
    #         rho_scale (float, optional): Size of zero rate shift in the calculation of the change of the instrument's
    #                                      price associated with zero rate curve shifts as difference quotient.
    #                                      Defaults to 0.0001 (= 1 basis point).
    #         calc_vega (bool, optional): Flag determining if the change of the instrument's price associated with small
    #                                     changes of the implied volatility surface shall be calculated to first order
    #                                     within the pricing routine. Defaults to False.
    #         vega_scale (float, optional): Size of implied volatility shift in the calculation of the change of the
    #                                       instrument's price associated with implied volatility surface shifts as
    #                                       difference quotient. Defaults to 0.01.
    #         calc_cross_volga (bool, optional): Flag determining if the change of the instrument's price associated with
    #                                            small changes of two implied volatility surfaces (both at first order)
    #                                            shall be calculated within the pricing routine. Defaults to True.
    #                                            TODO: why defaults to True?
    #         calc_vanna (bool, optional): Flag determining if the change of the instrument's price associated with small
    #                                      changes of the underlying's price and the implied volatility surface (both at
    #                                      first order) shall be calculated within the pricing routine. Defaults to False.
    #                                      TODO: why defaults to False in contrast to cross Gamma/Volga?
    #         calc_theta (bool, optional): Flag determining if the change of the instrument's price associated with small
    #                                      changes of the instrument's maturity shall be calculated to first order within
    #                                      the pricing routine. Defaults to False.
    #         theta_scale (float, optional): Size of maturity shift in the calculation of the change of the instrument's
    #                                        price associated with maturity shifts as difference quotient. Defaults to 1.0
    #         calc_spline (bool, optional):
    #         calc_grid_sizes (bool, optional):
    #         calc_implied_volatility (bool, optional):
    #         management_delta_limit (float, optional):
    #         calc_pricing_data (bool, optional):
    #         calc_expected_cashflows (bool, optional):
    #         calc_simulation_data (bool, optional):
    #         calc_additional_information (bool, optional):
    #         calc_z_spread (bool, optional): Flag determining if the instrument's z-spread shall be calculated within the
    #                                         pricing routine. Defaults to False.
    #         calc_yield_to_maturity (bool, optional): Flag determining if the instrument's yield-to-maturity shall be
    #                                                  calculated within the pricing routine. Defaults to False.
    #         calc_convexity (bool, optional): Flag determining if the change of the instrument's price associated with
    #                                          small changes of the zero rate curve shall be calculated to second order
    #                                          within the pricing routine. Defaults to False.
    #         max_expected_cashflow_date (_Union[date, datetime], optional):
    #         cashflow_times (_List[_Union[date, datetime]], optional):
    #         calc_macaulay_duration (bool, optional): Flag determining if the instrument's Macaulay duration shall be
    #                                                  calculated within the pricing routine. Defaults to False.
    #     """
    def __init__(self, calc_delta_gamma: bool = None, calc_cross_gamma: bool = None, calc_clean_price: bool = None,
                 calc_rho: bool = None, rho_scale: float = None, calc_vega: bool = None, vega_scale: float = None,
                 calc_cross_volga: bool = None, calc_vanna: bool = None, calc_theta: bool = None,
                 theta_scale: float = None, calc_spline: bool = None, calc_grid_sizes: bool = None,
                 calc_implied_volatility: bool = None, management_delta_limit: float = None,
                 calc_pricing_data: bool = None, calc_expected_cashflows: bool = None,
                 calc_simulation_data: bool = None, calc_additional_information: bool = None,
                 calc_z_spread: bool = None, calc_yield_to_maturity: bool = None, calc_convexity: bool = None,
                 max_expected_cashflow_date: _Union[date, datetime] = None,
                 cashflow_times: _List[_Union[date, datetime]] = None, calc_macaulay_duration: bool = None):
        """
        Defines the set of information, e.g. sensitivities, cashflow information, etc., to be calculated together with
        the instrument's price in the context of pricing. For some sensitivities the shift parameter for calculating
        the difference quotient (rather than the sensitivity in closed formula) is also stored in the pricing request
        object.

        Args: TODO: Read carefully and double-check if these guessed descriptions are valid!
            calc_delta_gamma (bool, optional): Flag determining if the change of the instrument's price associated with
                                               small changes of the underlying's price shall be calculated to first (and
                                               second) order within the pricing routine. Defaults to None.
                                               TODO: also second order? delta_scale=?
            calc_cross_gamma (bool, optional): Flag determining if the change of the instrument's price associated with
                                               small changes of two underlyings' prices (both at fist order) shall be
                                               calculated within the pricing routine. Defaults to None.
            calc_clean_price (bool, optional): Flag determining if the instrument's clean price shall be calculated
                                               additional to its dirty price within the pricing routine.
                                               Defaults to None.
            calc_rho (bool, optional): Flag determining if the change of the instrument's price associated with small
                                       changes of the zero rate curve shall be calculated to first order within the
                                       pricing routine. Defaults to None.
            rho_scale (float, optional): Size of zero rate shift in the calculation of the change of the instrument's
                                         price associated with zero rate curve shifts as difference quotient.
                                         Defaults to None.
            calc_vega (bool, optional): Flag determining if the change of the instrument's price associated with small
                                        changes of the implied volatility surface shall be calculated to first order
                                        within the pricing routine. Defaults to None.
            vega_scale (float, optional): Size of implied volatility shift in the calculation of the change of the
                                          instrument's price associated with implied volatility surface shifts as
                                          difference quotient. Defaults to None.
            calc_cross_volga (bool, optional): Flag determining if the change of the instrument's price associated with
                                               small changes of two implied volatility surfaces (both at first order)
                                               shall be calculated within the pricing routine. Defaults to None.
            calc_vanna (bool, optional): Flag determining if the change of the instrument's price associated with small
                                         changes of the underlying's price and the implied volatility surface (both at
                                         first order) shall be calculated within the pricing routine. Defaults to None.
            calc_theta (bool, optional): Flag determining if the change of the instrument's price associated with small
                                         changes of the instrument's maturity shall be calculated to first order within
                                         the pricing routine. Defaults to None.
            theta_scale (float, optional): Size of maturity shift in the calculation of the change of the instrument's
                                           price associated with maturity shifts as difference quotient.
                                           Defaults to None
            calc_spline (bool, optional):
            calc_grid_sizes (bool, optional):
            calc_implied_volatility (bool, optional):
            management_delta_limit (float, optional):
            calc_pricing_data (bool, optional):
            calc_expected_cashflows (bool, optional):
            calc_simulation_data (bool, optional):
            calc_additional_information (bool, optional):
            calc_z_spread (bool, optional): Flag determining if the instrument's z-spread shall be calculated within the
                                            pricing routine. Defaults to None.
            calc_yield_to_maturity (bool, optional): Flag determining if the instrument's yield-to-maturity shall be
                                                     calculated within the pricing routine. Defaults to None.
            calc_convexity (bool, optional): Flag determining if the change of the instrument's price associated with
                                             small changes of the zero rate curve shall be calculated to second order
                                             within the pricing routine. Defaults to None.
            max_expected_cashflow_date (_Union[date, datetime], optional):
            cashflow_times (_List[_Union[date, datetime]], optional):
            calc_macaulay_duration (bool, optional): Flag determining if the instrument's Macaulay duration shall be
                                                     calculated within the pricing routine. Defaults to None.
        """
        self._calc_delta_gamma = calc_delta_gamma
        self._calc_cross_gamma = calc_cross_gamma
        self._calc_clean_price = calc_clean_price
        self._calc_rho = calc_rho
        self._rho_scale = rho_scale
        self._calc_vega = calc_vega
        self._vega_scale = vega_scale
        self._calc_cross_volga = calc_cross_volga
        self._calc_vanna = calc_vanna
        self._calc_theta = calc_theta
        self._theta_scale = theta_scale
        self._calc_spline = calc_spline
        self._calc_grid_sizes = calc_grid_sizes
        self._calc_implied_volatility = calc_implied_volatility
        self._management_delta_limit = management_delta_limit
        self._calc_pricing_data = calc_pricing_data
        self._calc_expected_cashflows = calc_expected_cashflows
        self._calc_simulation_data = calc_simulation_data
        self._calc_additional_information = calc_additional_information
        self._calc_z_spread = calc_z_spread
        self._calc_yield_to_maturity = calc_yield_to_maturity
        self._calc_convexity = calc_convexity
        self._max_expected_cashflow_date = max_expected_cashflow_date
        self._cashflow_times = cashflow_times
        self._calc_macaulay_duration = calc_macaulay_duration

    @property
    def _calc_delta_gamma(self):
        return self.__calc_delta_gamma

    @_calc_delta_gamma.setter
    def _calc_delta_gamma(self, calc_delta_gamma: bool):
        if calc_delta_gamma is not None:
            if isinstance(calc_delta_gamma, bool):
                self.__calc_delta_gamma = calc_delta_gamma
            else:
                raise TypeError("'" + str(calc_delta_gamma) + "' must be of type bool!")

    @property
    def _calc_cross_gamma(self):
        return self.__calc_cross_gamma

    @_calc_cross_gamma.setter
    def _calc_cross_gamma(self, calc_cross_gamma: bool):
        if calc_cross_gamma is not None:
            if isinstance(calc_cross_gamma, bool):
                self.__calc_cross_gamma = calc_cross_gamma
                if self._calc_cross_gamma:
                    self.calc_delta_gamma = True
            else:
                raise TypeError("'" + str(calc_cross_gamma) + "' must be of type bool!")

    @property
    def _calc_clean_price(self):
        return self.__calc_clean_price

    @_calc_clean_price.setter
    def _calc_clean_price(self, calc_clean_price: bool):
        if calc_clean_price is not None:
            if isinstance(calc_clean_price, bool):
                self.__calc_clean_price = calc_clean_price
            else:
                raise TypeError("'" + str(calc_clean_price) + "' must be of type bool!")

    @property
    def _calc_rho(self):
        return self.__calc_rho

    @_calc_rho.setter
    def _calc_rho(self, calc_rho: bool):
        if calc_rho is not None:
            if isinstance(calc_rho, bool):
                self.__calc_rho = calc_rho
                if self._calc_rho & ~hasattr(self, 'rho_scale'):
                    self.rho_scale = 0.0001
            else:
                raise TypeError("'" + str(calc_rho) + "' must be of type bool!")

    @property
    def _rho_scale(self):
        return self.__rho_scale

    @_rho_scale.setter
    def _rho_scale(self, rho_scale: float):
        if rho_scale is not None:
            if isinstance(rho_scale, float):
                self.__rho_scale = rho_scale
                self.calc_rho = True
            else:
                raise TypeError("'" + str(rho_scale) + "' must be of type float!")

    @property
    def _calc_vega(self):
        return self.__calc_vega

    @_calc_vega.setter
    def _calc_vega(self, calc_vega: bool):
        if calc_vega is not None:
            if isinstance(calc_vega, bool):
                self.__calc_vega = calc_vega
                if self._calc_vega & ~hasattr(self, 'vega_scale'):
                    self.vega_scale = 0.01
            else:
                raise TypeError("'" + str(calc_vega) + "' must be of type bool!")

    @property
    def _vega_scale(self):
        return self.__vega_scale

    @_vega_scale.setter
    def _vega_scale(self, vega_scale: float):
        if vega_scale is not None:
            if isinstance(vega_scale, float):
                self.__vega_scale = vega_scale
                self.calc_vega = True
            else:
                raise TypeError("'" + str(vega_scale) + "' must be of type float!")

    @property
    def _calc_cross_volga(self):
        return self.__calc_cross_volga

    @_calc_cross_volga.setter
    def _calc_cross_volga(self, calc_cross_volga: bool):
        if calc_cross_volga is not None:
            if isinstance(calc_cross_volga, bool):
                self.__calc_cross_volga = calc_cross_volga
                if self._calc_cross_volga:
                    self.calc_vega = True
                    if not hasattr(self, 'vega_scale'):
                        self.vega_scale = 0.01
            else:
                raise TypeError("'" + str(calc_cross_volga) + "' must be of type bool!")

    @property
    def _calc_vanna(self):
        return self.__calc_vanna

    @_calc_vanna.setter
    def _calc_vanna(self, calc_vanna: bool):
        if calc_vanna is not None:
            if isinstance(calc_vanna, bool):
                self.__calc_vanna = calc_vanna
                if self._calc_vanna:
                    self.calc_vega = True
                    if not hasattr(self, 'vega_scale'):
                        self.vega_scale = 0.01
            else:
                raise TypeError("'" + str(calc_vanna) + "' must be of type bool!")

    @property
    def _calc_theta(self):
        return self.__calc_theta

    @_calc_theta.setter
    def _calc_theta(self, calc_theta: bool):
        if calc_theta is not None:
            if isinstance(calc_theta, bool):
                self.__calc_theta = calc_theta
                if self._calc_theta & ~hasattr(self, 'theta_scale'):
                    self.theta_scale = 1.0
            else:
                raise TypeError("'" + str(calc_theta) + "' must be of type bool!")

    @property
    def _theta_scale(self):
        return self.__theta_scale

    @_theta_scale.setter
    def _theta_scale(self, theta_scale: float):
        if theta_scale is not None:
            if isinstance(theta_scale, float):
                self.__theta_scale = theta_scale
                self.calc_theta = True
            else:
                raise TypeError("'" + str(theta_scale) + "' must be of type float!")

    @property
    def _calc_spline(self):
        return self.__calc_spline

    @_calc_spline.setter
    def _calc_spline(self, calc_spline: bool):
        if calc_spline is not None:
            if isinstance(calc_spline, bool):
                self.__calc_spline = calc_spline
            else:
                raise TypeError("'" + str(calc_spline) + "' must be of type bool!")

    @property
    def _calc_grid_sizes(self):
        return self.__calc_grid_sizes

    @_calc_grid_sizes.setter
    def _calc_grid_sizes(self, calc_grid_sizes: bool):
        if calc_grid_sizes is not None:
            if isinstance(calc_grid_sizes, bool):
                self.__calc_grid_sizes = calc_grid_sizes
            else:
                raise TypeError("'" + str(calc_grid_sizes) + "' must be of type bool!")

    @property
    def _calc_implied_volatility(self):
        return self.__calc_implied_volatility

    @_calc_implied_volatility.setter
    def _calc_implied_volatility(self, calc_implied_volatility: bool):
        if calc_implied_volatility is not None:
            if isinstance(calc_implied_volatility, bool):
                self.__calc_implied_volatility = calc_implied_volatility
            else:
                raise TypeError("'" + str(calc_implied_volatility) + "' must be of type bool!")

    @property
    def _management_delta_limit(self):
        return self.__management_delta_limit

    @_management_delta_limit.setter
    def _management_delta_limit(self, management_delta_limit: float):
        self.__management_delta_limit = management_delta_limit

    @property
    def _calc_pricing_data(self):
        return self.__calc_pricing_data

    @_calc_pricing_data.setter
    def _calc_pricing_data(self, calc_pricing_data: bool):
        if calc_pricing_data is not None:
            if isinstance(calc_pricing_data, float):
                self.__calc_pricing_data = calc_pricing_data
            else:
                raise TypeError("'" + str(calc_pricing_data) + "' must be of type float!")

    @property
    def _calc_expected_cashflows(self):
        return self.__calc_expected_cashflows

    @_calc_expected_cashflows.setter
    def _calc_expected_cashflows(self, calc_expected_cashflows: bool):
        if calc_expected_cashflows is not None:
            if isinstance(calc_expected_cashflows, bool):
                self.__calc_expected_cashflows = calc_expected_cashflows
            else:
                raise TypeError("'" + str(calc_expected_cashflows) + "' must be of type bool!")

    @property
    def _calc_simulation_data(self):
        return self.__calc_simulation_data

    @_calc_simulation_data.setter
    def _calc_simulation_data(self, calc_simulation_data: bool):
        if calc_simulation_data is not None:
            if isinstance(calc_simulation_data, bool):
                self.__calc_simulation_data = calc_simulation_data
            else:
                raise TypeError("'" + str(calc_simulation_data) + "' must be of type bool!")

    @property
    def _calc_additional_information(self):
        return self.__calc_additional_information

    @_calc_additional_information.setter
    def _calc_additional_information(self, calc_additional_information: bool):
        if calc_additional_information is not None:
            if isinstance(calc_additional_information, bool):
                self.__calc_additional_information = calc_additional_information
            else:
                raise TypeError("'" + str(calc_additional_information) + "' must be of type bool!")

    @property
    def _calc_z_spread(self):
        return self.__calc_z_spread

    @_calc_z_spread.setter
    def _calc_z_spread(self, calc_z_spread: bool):
        if calc_z_spread is not None:
            if isinstance(calc_z_spread, bool):
                self.__calc_z_spread = calc_z_spread
            else:
                raise TypeError("'" + str(calc_z_spread) + "' must be of type bool!")

    @property
    def _calc_yield_to_maturity(self):
        return self.__calc_yield_to_maturity

    @_calc_yield_to_maturity.setter
    def _calc_yield_to_maturity(self, calc_yield_to_maturity: bool):
        if calc_yield_to_maturity is not None:
            if isinstance(calc_yield_to_maturity, bool):
                self.__calc_yield_to_maturity = calc_yield_to_maturity
            else:
                raise TypeError("'" + str(calc_yield_to_maturity) + "' must be of type bool!")

    @property
    def _calc_convexity(self):
        return self.__calc_convexity

    @_calc_convexity.setter
    def _calc_convexity(self, calc_convexity: bool):
        if calc_convexity is not None:
            if isinstance(calc_convexity, bool):
                self.__calc_convexity = calc_convexity
                if self._calc_convexity:
                    self.calc_rho = True
                    if not hasattr(self, 'rho_scale'):
                        self.rho_scale = 0.0001
            else:
                raise TypeError("'" + str(calc_convexity) + "' must be of type bool!")

    @property
    def _max_expected_cashflow_date(self):
        return self.__max_expected_cashflow_date

    @_max_expected_cashflow_date.setter
    def _max_expected_cashflow_date(self, max_expected_cashflow_date: _Union[date, datetime]):
        if max_expected_cashflow_date is not None:
            if isinstance(max_expected_cashflow_date, datetime) | isinstance(max_expected_cashflow_date, date):
                self.__max_expected_cashflow_date = datetime_to_date(max_expected_cashflow_date)
            else:
                raise TypeError("'" + str(max_expected_cashflow_date) + "' must be of type datetime or date!")

    @property
    def _cashflow_times(self):
        return self.__cashflow_times

    @_cashflow_times.setter
    def _cashflow_times(self, cashflow_times: _List[_Union[date, datetime]]):
        if cashflow_times is not None:
            if isinstance(cashflow_times, list) & (
                    isinstance(cashflow_times[0], datetime) | isinstance(cashflow_times[0], date)
            ):
                self.__cashflow_times = datetime_to_date_list(cashflow_times)
            else:
                raise TypeError("'" + str(cashflow_times) + "' must be of type list of datetime or date!")

    @property
    def _calc_macaulay_duration(self):
        return self.__calc_macaulay_duration

    @_calc_macaulay_duration.setter
    def _calc_macaulay_duration(self, calc_macaulay_duration: bool):
        if calc_macaulay_duration is not None:
            if isinstance(calc_macaulay_duration, bool):
                self.__calc_macaulay_duration = calc_macaulay_duration
            else:
                raise TypeError("'" + str(calc_macaulay_duration) + "' must be of type bool!")


class BondPricingRequest(PricingRequest):
    def __init__(self, calc_clean_price: bool = False, calc_rho: bool = False, rho_scale: float = None,
                 calc_theta: bool = False, theta_scale: float = None, calc_yield_to_maturity: bool = False,
                 calc_convexity: bool = False, calc_macaulay_duration: bool = False):  # TODO: clarify why no z-spread
        """
        Configuration of set of information to be calculated together with the bond's dirty price. In restricts the
        general PricingRequest to the sub-set relevant for the pricing of bonds.
        """
        PricingRequest.__init__(self, calc_clean_price=calc_clean_price, calc_rho=calc_rho, rho_scale=rho_scale,
                                calc_theta=calc_theta, theta_scale=theta_scale,
                                calc_yield_to_maturity=calc_yield_to_maturity, calc_convexity=calc_convexity,
                                calc_macaulay_duration=calc_macaulay_duration)


if __name__ == '__main__':
    bond_pricing_request = BondPricingRequest()
