
from typing import Tuple, Iterable
from datetime import datetime 
from dateutil.relativedelta import relativedelta
from enum import IntEnum as _IntEnum
import pyvacon as _pyvacon
from rivapy.instruments import CDSSpecification

class ResultType(_IntEnum):
    PRICE = 0
    DELTA = 1
    GAMMA = 2
    THETA = 3
    RHO = 4
    VEGA = 5
    VANNA = 6

class PricingResults:
    def set_price(self, price: float):
        self._price = price

    def getPrice(self):
        return self._price

def _create_pricing_request(pr_dict : Iterable[ResultType]):
    result = _pyvacon.finance.pricing.PricingRequest()
    for d in pr_dict:
        if d is ResultType.DELTA or d is ResultType.GAMMA:
            result.setDeltaGamma(True)
        elif d is ResultType.THETA:
            result.setTheta(True)
        elif d is ResultType.RHO:
            result.setRho(True)
        elif d is ResultType.VEGA:
            result.setVega(True)
        elif d is ResultType.VANNA:
            result.setVanna(True)
    return result

class Black76PricingData:
    def __init__(self, val_date, spec, discount_curve, vol_surface, pricing_request : Iterable[ResultType]):
        self.spec = spec
        self.val_date = val_date
        self.discount_curve = discount_curve
        self.vol_surface = vol_surface
        self.pricing_request = pricing_request
        self._pyvacon_obj = None

    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _pyvacon.finance.pricing.Black76PricingData()
            self._pyvacon_obj.valDate = self.val_date
            self._pyvacon_obj.spec = self.spec._get_pyvacon_obj()
            self._pyvacon_obj.dsc = self.discount_curve._get_pyvacon_obj()
            self._pyvacon_obj.param = _pyvacon.finance.pricing.PricingParameter()
            self._pyvacon_obj.vol = self.vol_surface._get_pyvacon_obj()
            self._pyvacon_obj.pricingRequest = _create_pricing_request(self.pricing_request)
        return self._pyvacon_obj

    def price(self):
        return _pyvacon.finance.pricing.BasePricer.price(self._get_pyvacon_obj())
class CDSPricingData:
    def __init__(self, spec: CDSSpecification, val_date, discount_curve, survival_curve, 
                recovery_curve=None, integration_step = relativedelta(days=30)):
        self.spec = spec
        self.val_date = val_date
        self.discount_curve = discount_curve
        self.survival_curve = survival_curve
        self.recovery_curve = recovery_curve
        self._pricer_type = 'ISDA'
        self.integration_step = integration_step
        
    def _pv_protection_leg(self, valuation_date: datetime, integration_stepsize: relativedelta)->float:
        prev_date = max(self.val_date, self.spec.protection_start)
        current_date = min(prev_date + self.integration_step, self.spec.expiry)
        pv_protection = 0.0
        
        while current_date <= self.spec.expiry:
            default_prob = self.survival_curve.value(valuation_date, prev_date)-self.survival_curve.value(valuation_date, current_date)
            recovery = self.spec.recovery
            if recovery is None and self.recovery_curve is not None:
                recovery = self.recovery_curve.value(valuation_date, current_date) 
            pv_protection += self.discount_curve.value(valuation_date, current_date) * (1.0-recovery) * default_prob
            prev_date = current_date
            current_date += self.integration_step
            
        if prev_date < self.spec.expiry and current_date > self.spec.expiry:
            default_prob = self.survival_curve.value(valuation_date, prev_date)-self.survival_curve.value(valuation_date, self.spec.expiry)
            recovery = self.spec.recovery
            if recovery is None and self.recovery_curve is not None:
                recovery = self.recovery_curve.value(valuation_date, self.spec.expiry) 
            pv_protection += self.discount_curve.value(valuation_date, self.spec.expiry) * (1.0-recovery) * default_prob
            
        return pv_protection

    def _pv_premium_leg(self, valuation_date: datetime)->Tuple[float, float]:
        premium_period_start = self.spec.protection_start
        risk_adj_factor_premium=0  
        accrued = 0      
        #TODO include daycounter into CDSSpecification
        dc = _pyvacon.finance.definition.DayCounter(_pyvacon.finance.definition.DayCounter.Type.Act365Fixed)
        for premium_payment in self.spec.premium_pay_dates:
            if premium_payment >= valuation_date:
                period_length = dc.yf(premium_period_start, premium_payment)
                survival_prob = self.survival_curve.value(valuation_date, premium_payment)
                df = self.discount_curve.value(valuation_date, premium_payment)
                risk_adj_factor_premium += period_length*survival_prob*df
                default_prob = self.survival_curve.value(valuation_date, premium_period_start)-self.survival_curve.value(valuation_date, premium_payment)
                accrued += period_length*default_prob*df
                premium_period_start = premium_payment
        return risk_adj_factor_premium, accrued

    def par_spread(self, valuation_date: datetime, integration_stepsize: relativedelta)->float:
        prev_date = max(self.val_date, self.spec.protection_start)
        current_date = min(prev_date + self.integration_step, self.spec.expiry)
        pv_protection = 0.0
        premium_period_start = self.spec.protection_start
        risk_adj_factor_premium=0  
        accrued = 0 

        while current_date <= self.spec.expiry:
            default_prob = self.survival_curve.value(valuation_date, prev_date)-self.survival_curve.value(valuation_date, current_date)
            recovery = self.spec.recovery
            if recovery is None and self.recovery_curve is not None:
                recovery = self.recovery_curve.value(valuation_date, current_date) 
            pv_protection += self.discount_curve.value(valuation_date, current_date) * (1.0-recovery) * default_prob
            prev_date = current_date
            current_date += self.integration_step
            
        if prev_date < self.spec.expiry and current_date > self.spec.expiry:
            default_prob = self.survival_curve.value(valuation_date, prev_date)-self.survival_curve.value(valuation_date, self.spec.expiry)
            recovery = self.spec.recovery
            if recovery is None and self.recovery_curve is not None:
                recovery = self.recovery_curve.value(valuation_date, self.spec.expiry) 
            pv_protection += self.discount_curve.value(valuation_date, self.spec.expiry) * (1.0-recovery) * default_prob

        dc = _pyvacon.finance.definition.DayCounter(_pyvacon.finance.definition.DayCounter.Type.Act365Fixed)
        for premium_payment in self.spec.premium_pay_dates:
            if premium_payment >= valuation_date:
                period_length = dc.yf(premium_period_start, premium_payment)
                survival_prob = self.survival_curve.value(valuation_date, premium_payment)
                df = self.discount_curve.value(valuation_date, premium_payment)
                risk_adj_factor_premium += period_length*survival_prob*df
                default_prob = self.survival_curve.value(valuation_date, premium_period_start)-self.survival_curve.value(valuation_date, premium_payment)
                accrued += period_length*default_prob*df
                premium_period_start = premium_payment

        PV_accrued=((1/2)*accrued)
        PV_premium=(1)*risk_adj_factor_premium
        PV_protection=(((1-recovery))*pv_protection)
        
        par_spread_i=(PV_protection)/((PV_premium+PV_accrued))
        return par_spread_i

    def price(self):
        pv_protection = self._pv_protection_leg(self.val_date, self.integration_step)
        pr_results = PricingResults()
        pr_results.pv_protection = self.spec.notional*pv_protection
        premium_leg, accrued = self._pv_premium_leg(self.val_date)
        pr_results.premium_leg = self.spec.premium*self.spec.notional*premium_leg
        pr_results.accrued = 0.5*self.spec.premium*self.spec.notional*accrued
        pr_results.par_spread=self.par_spread(self.val_date, self.integration_step)
        pr_results.set_price(pr_results.pv_protection-pr_results.premium_leg-pr_results.accrued)
        return pr_results