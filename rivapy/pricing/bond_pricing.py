from datetime import datetime
from scipy.optimize import brentq
from rivapy.tools.interfaces import BaseDatedCurve, HasExpectedCashflows
from rivapy.marketdata import DiscountCurveParametrized, ConstantRate
from rivapy.pricing.pricing_request import PricingRequest
from rivapy.pricing._logger import logger

class SimpleCashflowPricer:
    @staticmethod
    def pv_cashflows(val_date: datetime, specification: HasExpectedCashflows, discount_curve: BaseDatedCurve)->float:
        logger.info('Start computing pv cashflows for bond ' + specification.obj_id)

        cashflows =  specification.expected_cashflows()
        pv_cashflows = 0.0
        for c in cashflows:
            if c[0]>= val_date:
                df =  discount_curve.value(val_date, c[0])
                logger.debug('Cashflow ' + str(c[1]) + ', date: ' + str(c[0]) + ', df: ' + str(df))
                pv_cashflows += df*c[1]
        logger.info('Finished computing pv cashflows for bond ' + specification.obj_id + ', pv_cashflows: '+ str(pv_cashflows) )
        return pv_cashflows
    
    # TODO: add accrued interest 
    @staticmethod
    def compute_yield(target_dirty_price: float, val_date: datetime, 
                    specification: HasExpectedCashflows)->float:
        logger.info('Start computing bond yield for bond ' + specification.obj_id + ', dirty price: ' + str(target_dirty_price))
        def target_function(r: float)->float:
            dc = DiscountCurveParametrized('', val_date, ConstantRate(r))
            price = SimpleCashflowPricer.pv_cashflows(val_date, specification, dc)
            logger.debug('Target function called with r: ' + str(r) + ', price: ' + str(price) + ', target_dirty_price: ' + str(target_dirty_price))
            return price - target_dirty_price
        result =  brentq(target_function, -0.2, 1.5, full_output = False)
        logger.info('Finished computing bond yield')
        return result
        