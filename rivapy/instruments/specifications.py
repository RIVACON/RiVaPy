
from datetime import datetime 
from rivapy.tools.enums import SecuritizationLevel, Currency
import rivapy.tools.interfaces as interfaces
#from rivapy.enums import Currency
from rivapy import _pyvacon_available
if _pyvacon_available:
    import pyvacon.finance.specification as _spec
    
    ComboSpecification = _spec.ComboSpecification
    #Equity/FX
    PayoffStructure = _spec.PayoffStructure
    ExerciseSchedule = _spec.ExerciseSchedule
    BarrierDefinition = _spec.BarrierDefinition
    BarrierSchedule = _spec.BarrierSchedule
    BarrierPayoff = _spec.BarrierPayoff
    BarrierSpecification = _spec.BarrierSpecification
    # EuropeanVanillaSpecification = _spec.EuropeanVanillaSpecification
    # AmericanVanillaSpecification = _spec.AmericanVanillaSpecification
    #RainbowUnderlyingSpec = _spec.RainbowUnderlyingSpec
    #RainbowBarrierSpec = _spec.RainbowBarrierSpec
    LocalVolMonteCarloSpecification = _spec.LocalVolMonteCarloSpecification
    RainbowSpecification = _spec.RainbowSpecification
    MultiMemoryExpressSpecification = _spec.MultiMemoryExpressSpecification
    MemoryExpressSpecification = _spec.MemoryExpressSpecification
    ExpressPlusSpecification = _spec.ExpressPlusSpecification
    AsianVanillaSpecification = _spec.AsianVanillaSpecification
    RiskControlStrategy = _spec.RiskControlStrategy
    AsianRiskControlSpecification = _spec.AsianRiskControlSpecification


    #Interest Rates
    IrSwapLegSpecification = _spec.IrSwapLegSpecification
    IrFixedLegSpecification = _spec.IrFixedLegSpecification
    IrFloatLegSpecification = _spec.IrFloatLegSpecification
    InterestRateSwapSpecification = _spec.InterestRateSwapSpecification
    InterestRateBasisSwapSpecification = _spec.InterestRateBasisSwapSpecification
    DepositSpecification = _spec.DepositSpecification
    InterestRateFutureSpecification = _spec.InterestRateFutureSpecification
        
    InflationLinkedBondSpecification = _spec.InflationLinkedBondSpecification
    CallableBondSpecification = _spec.CallableBondSpecification

    #GasStorageSpecification = _spec.GasStorageSpecification

    #ScheduleSpecification = _spec.ScheduleSpecification

    #SpecificationManager = _spec.SpecificationManager

    #Bonds/Credit
    CouponDescription = _spec.CouponDescription
    BondSpecification = _spec.BondSpecification
else:
    #empty placeholder...
    class BondSpecification:
        pass
    class ComboSpecification:
        pass
    class BarrierSpecification:
        pass
    class RainbowSpecification:
        pass
    class MemoryExpressSpecification:
        pass

class EuropeanVanillaSpecification(interfaces.FactoryObject):
    def __init__(self, 
                 id: str,
                 type: str,
                 expiry: datetime,
                 strike: float,
                 issuer: str = '',
                 sec_lvl: str = SecuritizationLevel.COLLATERALIZED,
                 curr: str = Currency.EUR,
                 udl_id: str = '',
                 share_ratio: float = 1.0,
                #  holidays: str = '',
                #  ex_settle: int = 0, not implemented
                #  trade_settle: int = 0 not implemented
                 ):
        
        """Constructor for european vanilla option

        Args:
            id (str): Identifier (name) of the european vanilla specification.
            type (str): Type of the european vanilla option ('PUT','CALL').
            expiry (datetime): Expiration date.
            strike (float): Strike price.
            issuer (str, optional): Issuer Id. Must not be set if pricing data is manually defined. Defaults to ''.
            sec_lvl (str, optional): Securitization level. Can be selected from rivapy.enums.SecuritizationLevel. Defaults to SecuritizationLevel.COLLATERALIZED.
            curr (str, optional): Currency (ISO-4217 Code). Must not be set if pricing data is manually defined. Can be selected from rivapy.enums.Currency. Defaults to Currency.EUR.
            udl_id (str, optional): Underlying Id. Must not be set if pricing data is manually defined. Defaults to ''.
            share_ratio (float, optional): Ratio of covered shares of the underlying by a single option contract. Defaults to 1.0.
        """
        
        self.id = id
        self.issuer = issuer
        self.sec_lvl = sec_lvl
        self.curr =  curr
        self.udl_id = udl_id
        self.type = type
        self.expiry = expiry
        self.strike = strike
        self.share_ratio = share_ratio
        # self.holidays = holidays
        # self.ex_settle = ex_settle
        # self.trade_settle = trade_settle
        
        self._pyvacon_obj = None

    def _to_dict(self)->dict:
        return {'id': self.id, 'issuer':self.issuer, 'sec_lvl': self.sec_lvl, 'curr': self.curr, 'udl_id': self.udl_id, 'type': self.type,'expiry':self.expiry, 'strike':self.strike,'share_ratio': self.share_ratio}

        
    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _spec.EuropeanVanillaSpecification(self.id, 
                                            self.issuer, 
                                            self.sec_lvl, 
                                            self.curr, 
                                            self.udl_id, 
                                            self.type,
                                            self.expiry,
                                            self.strike,
                                            self.share_ratio,
                                            '',
                                            0,
                                            0)
                                            
        return self._pyvacon_obj
    
class AmericanVanillaSpecification:
    def __init__(self
                 ,id: str
                 ,type: str
                 ,expiry: datetime
                 ,strike: float
                 ,issuer: str = ''
                 ,sec_lvl: str = SecuritizationLevel.COLLATERALIZED
                 ,curr: str = Currency.EUR
                 ,udl_id: str = ''
                 ,share_ratio: float = 1.0
                 ,exercise_before_ex_date: bool = False
                #  ,holidays: str
                #  ,ex_settle: str
                #  ,trade_settle: str
                 ):
        """Constructor for american vanilla option

        Args:
            id (str): Identifier (name) of the american vanilla specification.
            type (str): Type of the american vanilla option ('PUT','CALL').
            expiry (datetime): Expiration date.
            strike (float): Strike price.
            issuer (str, optional): Issuer Id. Must not be set if pricing data is manually defined. Defaults to ''.
            sec_lvl (str, optional): Securitization level. Can be selected from rivapy.enums.SecuritizationLevel. Defaults to SecuritizationLevel.COLLATERALIZED.
            curr (str, optional): Currency (ISO-4217 Code). Must not be set if pricing data is manually defined. Can be selected from rivapy.enums.Currency. Defaults to Currency.EUR.
            udl_id (str, optional): Underlying Id. Must not be set if pricing data is manually defined. Defaults to ''.
            share_ratio (float, optional): Ratio of covered shares of the underlying by a single option contract. Defaults to 1.0.
            exercise_before_ex_date (bool, optional): Indicates if option can be exercised within two days before dividend ex-date. Defaults to False.
        """
     
        self.id = id
        self.type = type
        self.expiry = expiry
        self.strike = strike
        self.issuer = issuer
        self.sec_lvl = sec_lvl
        self.curr =  curr
        self.udl_id = udl_id
        self.share_ratio = share_ratio
        self.exercise_before_ex_date = exercise_before_ex_date
        # self.holidays = holidays
        # self.ex_settle = ex_settle
        # self.trade_settle = trade_settle
            
        self._pyvacon_obj = None
        
    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _spec.AmericanVanillaSpecification(self.id
                                            ,self.issuer
                                            ,self.sec_lvl
                                            ,self.curr
                                            ,self.udl_id
                                            ,self.type
                                            ,self.expiry
                                            ,self.strike
                                            ,self.share_ratio
                                            ,self.exercise_before_ex_date
                                            ,''
                                            ,0
                                            ,0)
                                            
        return self._pyvacon_obj   


