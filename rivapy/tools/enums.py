# -*- coding: utf-8 -*-
from enum import Enum as _Enum, unique as _unique
from rivapy import _pyvacon_available
"""

The following Enum sub-classes replace to corresponding former classes one-on-one. The main reason for this replacement
is the more comfortable iterations over the enumeration class members. Moreover, the Enum class provides potentially
useful functionalities like comparisons, pickling, ... Finally, the decorator @unique ensures unique enumeration values.
"""
class _MyEnum(_Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def to_string(cls, value)->str:
        """Checks if given enum class contains the value and raises exception if not. If value is str 

        Args:
            enum (_type_): _description_
            value (_type_): _description_

        Returns:
            str: _description_
        """
        def has_value(cls, value):
            return value in cls._value2member_map_
        if isinstance(value, str):
            if not cls.has_value(value):
                raise Exception('Unknown  ' + cls.__name__ +': ' + value)
            return value
        if isinstance(value, cls):
            return value.value
        raise Exception('Given value ' + str(value) + ' does not belong to enum ' + cls.__name__)
    
class _MyIntEnum(_Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def to_string(cls, value)->str:
        """Checks if given enum class contains the value and raises exception if not. If value is str 

        Args:
            enum (_type_): _description_
            value (_type_): _description_

        Returns:
            str: _description_
        """
        def has_value(cls, value):
            return value in cls._value2member_map_
        if isinstance(value, str):
            cls[value] #check if the string exists as key
            return value
        elif isinstance(value, int):
            return cls._value2member_map_[value].name
        if isinstance(value, cls):
            return value.name
        raise Exception('Given value ' + str(value) + ' does not belong to enum ' + cls.__name__)

if _pyvacon_available:
    from pyvacon.finance.definition import DayCounter as _DayCounter
    DayCounterType = _DayCounter.Type

    from pyvacon.numerics.interpolation import InterpolationType
    from pyvacon.numerics.extrapolation import ExtrapolationType
else:
    @_unique
    class InterpolationType(_MyEnum):
        CONSTANT = 'CONSTANT'
        LINEAR = 'LINEAR'
        LINEAR_LOG = 'LINEARLOG'
        CONSTRAINED_SPLINE = 'CONSTRAINED_SPLINE'
        HAGAN = 'HAGAN'
        HAGAN_DF = 'HAGAN_DF'


    @_unique
    class ExtrapolationType(_MyEnum):
        NONE = 'NONE'
        CONSTANT = 'CONSTANT'
        LINEAR = 'LINEAR'
        LINEAR_LOG = 'LINEARLOG'
        
@_unique
class SecuritizationLevel(_MyEnum):
    NONE = 'NONE'
    COLLATERALIZED = 'COLLATERALIZED'
    SENIOR_SECURED = 'SENIOR_SECURED'
    SENIOR_UNSECURED = 'SENIOR_UNSECURED'
    SUBORDINATED = 'SUBORDINATED'
    MEZZANINE = 'MEZZANINE'
    EQUITY = 'EQUITY'
    PREFERRED_SENIOR = 'PREFERRED_SENIOR'
    NON_PREFERRED_SENIOR = 'NON_PREFERRED_SENIOR'

# class SecuritizationLevel:
#     NONE = 'NONE'
#     COLLATERALIZED = 'COLLATERALIZED' #,,,'','SUBORDINATED','MEZZANINE','EQUITY']
#     SENIOR_SECURED = 'SENIOR_SECURED'
#     SENIOR_UNSECURED = 'SENIOR_UNSECURED'
#     SUBORDINATED = 'SUBORDINATED'
#     MEZZANINE = 'MEZZANINE'
#     EQUITY = 'EQUITY'

# @_unique
# class ProductType(_MyEnum):
#     BOND = 'BOND'
#     CALLABLE_BOND = 'CALLABLE_BOND'


# @_unique
# class PricerType(_MyEnum):
#     ANALYTIC = 'ANALYTIC'
#     PDE = 'PDE'
#     MONTE_CARLO = 'MONTE_CARLO'
#     COMBO = 'COMBO'


@_unique
class Model(_MyEnum):
    BLACK76 = 'BLACK76'
    CIR = 'CIR'
    HULL_WHITE = 'HULL_WHITE'
    HESTON = 'HESTON'
    LV = 'LV'
    GBM = 'GBM'
    G2PP = 'G2PP'
    VASICEK = 'VASICEK'

# class Model:
#     BLACK76 = 'BLACK76'
#     CIR ='CIR'
#     HULL_WHITE = 'HULL_WHITE'
#     HESTON = 'HESTON'
#     LV = 'LV'
#     GBM = 'GBM'
#     G2PP = 'G2PP'
#     VASICEK = 'VASICEK'

@_unique
class Period(_MyEnum):
    A = 'A'
    SA = 'SA'
    Q = 'Q'
    M = 'M'
    D = 'D'

    
# class Period:
#     A = 'A'
#     SA = 'SA'
#     Q = 'Q'
#     M = 'M'
#     D = 'D'

@_unique
class RollConvention(_MyEnum):
    FOLLOWING = 'Following'
    MODIFIED_FOLLOWING = 'ModifiedFollowing'
    MODIFIED_FOLLOWING_EOM = 'ModifiedFollowingEOM'
    MODIFIED_FOLLOWING_BIMONTHLY = 'ModifiedFollowingBimonthly'
    PRECEDING = 'Preceding'
    MODIFIED_PRECEDING = 'ModifiedPreceding'
    NEAREST = 'Nearest'
    UNADJUSTED = 'Unadjusted'

# class RollConvention:
#     FOLLOWING = 'Following'
#     MODIFIED_FOLLOWING = 'ModifiedFollowing'
#     MODIFIED_FOLLOWING_EOM = 'ModifiedFollowingEOM'
#     PRECEDING = 'Preceding'
#     MODIFIED_PRECEDING = 'ModifiedPreceding'
#     UNADJUSTED = 'Unadjusted'
@_unique
class DayCounterType(_MyEnum):
    ACT_ACT = 'ActAct'
    Act365Fixed = 'Act365Fixed'
    ACT360 = 'Act360'
    ThirtyU360 = '30U360'
    ThirtyE360 = '30E360'
    ACT252 = 'Act252'
@_unique
class InflationInterpolation(_MyEnum):
    UNDEFINED = 'UNDEFINED'
    GERMAN = 'GERMAN'
    JAPAN = 'JAPAN'
    CONSTANT = 'CONSTANT'

@_unique
class Sector(_MyEnum):
    UNDEFINED = 'UNDEFINED'
    # BASIC_MATERIALS = 'BasicMaterials'
    CONGLOMERATES = 'Conglomerates'
    CONSUMER_GOODS = 'ConsumerGoods'
    # FINANCIAL = 'Financial'
    # HEALTHCARE = 'Healthcare'
    # INDUSTRIAL_GOODS = 'IndustrialGoods'
    SERVICES = 'Services'
    # TECHNOLOGY = 'Technology'
    # UTILITIES = 'Utilities'

    COMMUNICATION_SERVICES = 'CommunicationServices'
    CONSUMER_STAPLES = 'ConsumerStaples'
    CONSUMER_DISCRETIONARY = 'ConsumerDiscretionary'
    ENERGY = 'Energy'
    FINANCIAL = 'Financial'
    HEALTH_CARE = 'HealthCare'
    INDUSTRIALS = 'Industrials'
    INFORMATION_TECHNOLOGY = 'InformationTechnology'
    MATERIALS = 'Materials'
    REAL_ESTATE = 'RealEstate'
    UTILITIES = 'Utilities'

@_unique
class ESGRating(_MyEnum): # see MSCI ESG ratings
    AAA = 'AAA'
    AA = 'AA'
    A = 'A'
    BBB = 'BBB'
    BB = 'BB'
    B = 'B'
    CCC = 'CCC'

@_unique
class Rating(_MyEnum):
    # cf. https://www.moneyland.ch/de/vergleich-rating-agenturen
    AAA = 'AAA'
    AA_PLUS = 'AA+'
    AA = 'AA'
    AA_MINUS = 'AA-'
    A_PLUS = 'A+'
    A = 'A'
    A_MINUS = 'A-'
    BBB_PLUS = 'BBB+'
    BBB = 'BBB'
    BBB_MINUS = 'BBB-'
    BB_PLUS = 'BB+'
    BB = 'BB'
    BB_MINUS = 'BB-'
    B_PLUS = 'B+'
    B = 'B'
    B_MINUS = 'B-'
    CCC_PLUS = 'CCC+'
    CCC = 'CCC'
    CCC_MINUS = 'CCC-'
    CC = 'CC'
    C = 'C'
    D = 'D'
    NONE = 'NONE' # not rated

# class ProductType:
#        BOND = 'BOND'
#        CALLABLE_BOND = 'CALLABLE_BOND'   

class PricerType:
    ANALYTIC = 'ANALYTIC'
    PDE = 'PDE'
    MONTE_CARLO = 'MONTE_CARLO'
    COMBO = 'COMBO'
    
       
@_unique
class VolatilityStickyness(_MyEnum):
    NONE = 'NONE'
    StickyStrike = 'StickyStrike'
    StickyXStrike = 'StickyXStrike'
    StickyFwdMoneyness = 'StickyFwdMoneyness'

@_unique
class Currency(_MyEnum):
    AED =  'AED'
    AFN =  'AFN'
    ALL =  'ALL'
    AMD =  'AMD'
    ANG =  'ANG'
    AOA =  'AOA'
    ARS =  'ARS'
    AUD =  'AUD'
    AWG =  'AWG'
    AZN =  'AZN'
    BAM =  'BAM'
    BBD =  'BBD'
    BDT =  'BDT'
    BGN =  'BGN'
    BHD =  'BHD'
    BIF =  'BIF'
    BMD =  'BMD'
    BND =  'BND'
    BOB =  'BOB'
    BRL =  'BRL'
    BSD =  'BSD'
    BTN =  'BTN'
    BWP =  'BWP'
    BYR =  'BYR'
    BZD =  'BZD'
    CAD =  'CAD'
    CDF =  'CDF'
    CHF =  'CHF'
    CLP =  'CLP'
    CNH =  'CNH'
    CNY =  'CNY'
    COP =  'COP'
    CRC =  'CRC'
    CUC =  'CUC'
    CUP =  'CUP'
    CVE =  'CVE'
    CZK =  'CZK'
    DJF =  'DJF'
    DKK =  'DKK'
    DOP =  'DOP'
    DZD =  'DZD'
    EGP =  'EGP'
    ERN =  'ERN'
    ETB =  'ETB'
    EUR =  'EUR'
    FJD =  'FJD'
    FKP =  'FKP'
    GBP =  'GBP'
    GEL =  'GEL'
    GGP =  'GGP'
    GHS =  'GHS'
    GIP =  'GIP'
    GMD =  'GMD'
    GNF =  'GNF'
    GTQ =  'GTQ'
    GYD =  'GYD'
    HKD =  'HKD'
    HNL =  'HNL'
    HRK =  'HRK'
    HTG =  'HTG'
    HUF =  'HUF'
    IDR =  'IDR'
    ILS =  'ILS'
    IMP =  'IMP'
    INR =  'INR'
    IQD =  'IQD'
    IRR =  'IRR'
    ISK =  'ISK'
    JEP =  'JEP'
    JMD =  'JMD'
    JOD =  'JOD'
    JPY =  'JPY'
    KES =  'KES'
    KGS =  'KGS'
    KHR =  'KHR'
    KMF =  'KMF'
    KPW =  'KPW'
    KRW =  'KRW'
    KWD =  'KWD'
    KYD =  'KYD'
    KZT =  'KZT'
    LAK =  'LAK'
    LBP =  'LBP'
    LKR =  'LKR'
    LRD =  'LRD'
    LSL =  'LSL'
    LTL =  'LTL'
    LVL =  'LVL'
    LYD =  'LYD'
    MAD =  'MAD'
    MDL =  'MDL'
    MGA =  'MGA'
    MKD =  'MKD'
    MMK =  'MMK'
    MNT =  'MNT'
    MOP =  'MOP'
    MRO =  'MRO'
    MUR =  'MUR'
    MVR =  'MVR'
    MWK =  'MWK'
    MXN =  'MXN'
    MYR =  'MYR'
    MZN =  'MZN'
    NAD =  'NAD'
    NGN =  'NGN'
    NIO =  'NIO'
    NOK =  'NOK'
    NPR =  'NPR'
    NZD =  'NZD'
    OMR =  'OMR'
    PAB =  'PAB'
    PEN =  'PEN'
    PGK =  'PGK'
    PHP =  'PHP'
    PKR =  'PKR'
    PLN =  'PLN'
    PYG =  'PYG'
    QAR =  'QAR'
    RON =  'RON'
    RSD =  'RSD'
    RUB =  'RUB'
    RWF =  'RWF'
    SAR =  'SAR'
    SBD =  'SBD'
    SCR =  'SCR'
    SDG =  'SDG'
    SEK =  'SEK'
    SGD =  'SGD'
    SHP =  'SHP'
    SLL =  'SLL'
    SOS =  'SOS'
    SPL =  'SPL'
    SRD =  'SRD'
    STD =  'STD'
    SVC =  'SVC'
    SYP =  'SYP'
    SZL =  'SZL'
    THB =  'THB'
    TJS =  'TJS'
    TMT =  'TMT'
    TND =  'TND'
    TOP =  'TOP'
    TRY =  'TRY'
    TTD =  'TTD'
    TVD =  'TVD'
    TWD =  'TWD'
    TZS =  'TZS'
    UAH =  'UAH'
    UGX =  'UGX'
    USD =  'USD'
    UYU =  'UYU'
    UZS =  'UZS'
    VEF =  'VEF'
    VND =  'VND'
    VUV =  'VUV'
    WST =  'WST'
    XAF =  'XAF'
    XAG =  'XAG'
    XAU =  'XAU'
    XPD =  'XPD'
    XPT =  'XPT'
    XCD =  'XCD'
    XDR =  'XDR'
    XOF =  'XOF'
    XPF =  'XPF'
    YER =  'YER'
    ZAR =  'ZAR'
    ZMW =  'ZMW'
    ZWD =  'ZWD'


class Country(_MyEnum):
    DE = 'DE'
    FR = 'FR'
    CA = 'CA'
    US = 'US'
    GB = 'GB'
    JP = 'JP'
    CN = 'CN'
       

    
