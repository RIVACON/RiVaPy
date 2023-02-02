import numpy as np
from rivapy.tools.enums import Currency, ESGRating, Rating, Sector, Country
from rivapy.marketdata import DiscountCurveParametrized, NelsonSiegel, LinearRate
from rivapy.instruments.components import Issuer
from rivapy.instruments import PlainVanillaCouponBondSpecification

class SpreadCurveSampler:
    def __init__(self):
        """This class samples spreadcurves used to price bonds. It creates different curves according to
            
            * issuer rating (for all ratings defined in :class:`rivapy.tools.enums.Rating`)
            * currency (for all currencies defined in :class:`rivapy.tools.enums.Currency`)
            * esg rating (for all ratings defined in :class:`rivapy.tools.enums.ESGRating`)
            * sector (for all sectors defined in :class:`rivapy.tools.enums.Sector`)
        """
        pass
    
    def sample(self, ref_date):
        min_params = {'min_short_term_rate': -0.01, 
                          'max_short_term_rate': 0.02, 
                          'min_long_run_rate': 0.0,
                          'max_long_run_rate': 0.03,
                          'min_hump': -0.02,
                          'max_hump': 0.05,
                          'min_tau': 0.5,
                          'max_tau': 3.0}
        
        max_params = {'min_short_term_rate': 0.02, 
                          'max_short_term_rate': 0.15, 
                          'min_long_run_rate': 0.03,
                          'max_long_run_rate': 0.25,
                          'min_hump': 0.0,
                          'max_hump': 0.2,
                          'min_tau': 0.5,
                          'max_tau': 5.0}
        
        curve_best_rating = DiscountCurveParametrized('', ref_date, 
                                                           NelsonSiegel._create_sample(n_samples=1, 
                                                                                       seed=None,**min_params)[0])
        curve_worst_rating = curve_best_rating + DiscountCurveParametrized('', ref_date, 
                                                           NelsonSiegel._create_sample(n_samples=1, 
                                                                                     seed=None,**max_params)[0])
        self.rating_curve = (curve_best_rating, curve_worst_rating)
        self._sample_currency_spread()
        self._sample_esg_rating_spreads()
        self._sample_rating_weights()
        self._sample_sector_spreads()
        self._sample_country_curves()
        
        
    def _sample_currency_spread(self):
        self.currency_spread = {}
        low = np.random.uniform(0.005,0.01)
        high = low + np.random.uniform(0.0,0.1)
        for c in Currency:
            self.currency_spread[c.value] = (low, high)
        for c in [Currency.EUR, Currency.USD, Currency.GBP, Currency.JPY]:
            low = np.random.uniform(0.0,0.01)
            high = low + np.random.uniform(0.0,0.1)
            self.currency_spread[c.value] = (low, high)
        
    def _sample_esg_rating_spreads(self):
        self.esg_rating_spread = {}
        low = 0.0
        for i,s in enumerate(ESGRating):
            high = low + np.random.uniform(low=0.001, high=0.0025)
            self.esg_rating_spread[s.value] = (low, high)
            low = high
        
    def _sample_rating_weights(self):
        rating_weights = np.random.uniform(low=1.0, high=4.0, size=len(Rating)).cumsum()
        rating_weights[0] = 0.0
        rating_weights[-1] = 4.0
        rating_weights = rating_weights/rating_weights.max()
        self.rating_weights = {}
        for i,k in enumerate(Rating):
            self.rating_weights[k.value] = rating_weights[i]
        
    def _sample_sector_spreads(self):
        result = {}
        for s in Sector:
            s_low = np.random.uniform(low=0.001, high=0.0025)
            result[s.value] = (s_low, s_low+np.random.uniform(low=0.001, high=0.0025))
        self.sector_spreads = result
        
    def _sample_country_curves(self):
        self.country_curves = {}
        for c in Country:
            shortterm_rate = np.random.uniform(low=0.0, high=0.02)
            longterm_rate = shortterm_rate + np.random.uniform(low=-0.005, high=0.005)
            lower_curve = DiscountCurveParametrized('', ref_date, LinearRate(shortterm_rate, longterm_rate))
            self.country_curves[c.value] = (lower_curve, lower_curve + DiscountCurveParametrized('', ref_date,
                                                                    ConstantRate(np.random.uniform(0.05, 0.15))
                                                                                       )
                                  )
    
    def get_curve(self, issuer: Issuer, bond: PlainVanillaCouponBondSpecification):
        rating_weight = self.rating_weights[issuer.rating]
        w1 = 1.0-rating_weight
        w2 = rating_weight
        rating_curve = w1*self.rating_curve[0] + w2*self.rating_curve[1]
        country_spread = w1*self.country_curves[issuer.country][0] + w2*self.country_curves[issuer.country][1]
        esg_spread = w1*self.esg_rating_spread[issuer.esg_rating][0] +  w2*self.esg_rating_spread[issuer.esg_rating][1]
        sector_spread = w1*self.sector_spreads[issuer.sector][0] + w2*self.sector_spreads[issuer.sector][1]
        currency_spread = w1*self.currency_spread[bond.currency][0] + w2*self.currency_spread[bond.currency][1]
        curve = 0.5*rating_curve + 0.5*(0.3*country_spread + 0.4*esg_spread + 0.2*sector_spread+0.1*currency_spread)
        return curve