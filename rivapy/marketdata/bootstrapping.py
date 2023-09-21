import pyvacon

from dateutil.relativedelta import relativedelta
from .curves import DiscountCurve
from ..tools.enums import DayCounterType


def bootstrap_curve(ref_date, curve_id, dc, instruments, quotes, discount_curve=None, basis_curve=None):
    _dc = discount_curve._get_pyvacon_obj() if discount_curve is not None else None
    _bc = basis_curve._get_pyvacon_obj() if basis_curve is not None else None
    _curve = pyvacon.finance.calibration.YieldCurveBootstrapper.compute(
            ref_date, curve_id, dc, instruments, quotes, _dc, _bc)
    curve = DiscountCurve(
        id=_curve.getObjectId(),
        refdate = pyvacon.converter.create_datetime(_curve.getRefDate()),
        dates=[pyvacon.converter.create_datetime(d) for d in _curve.getNodeDates()],
        df=[_curve.value(_curve.getRefDate(), d) for d in _curve.getNodeDates()],
        interpolation=_curve.getInterpolationType(),
        extrapolation=_curve.getExtrapolationType(),
        daycounter=DayCounterType[_curve.getDayCounterType()]
    )
    return curve


def bootstrap_curve_from_quote_table(input_data, output_spec):
    ref_date = output_spec['refDate']
    n = len(input_data)
    instruments = pyvacon.finance.specification.VectorBaseSpecification(n)
    quotes = []
    holidays = output_spec['calendar']
    for i in range(0, n):
        ins = InstrumentSpec(ref_date, input_data.iloc[i, :], holidays)
        instruments[i] = ins.get_instrument()
        quotes.append(ins.parRate)
    df = output_spec.get('discountCurve')
    if df is not None:
        df = df._get_pyvacon_obj()
    basis_curve = output_spec.get('basisCurve')
    if basis_curve is not None:
        basis_curve = basis_curve._get_pyvacon_obj()
    _curve = pyvacon.finance.calibration.YieldCurveBootstrapper.compute(
        ref_date, output_spec['curveName'], output_spec['dayCount'], instruments, quotes, df, basis_curve)
    curve = DiscountCurve(
        id=_curve.getObjectId(),
        refdate=pyvacon.converter.create_datetime(_curve.getRefDate()),
        dates=[pyvacon.converter.create_datetime(d) for d in _curve.getNodeDates()],
        df=[_curve.value(_curve.getRefDate(), d) for d in _curve.getNodeDates()],
        interpolation=_curve.getInterpolationType(),
        extrapolation=_curve.getExtrapolationType(),
        daycounter=DayCounterType[_curve.getDayCounterType()]
    )
    return curve



class InstrumentSpec:
    """
    Definition of input instruments for IR boostrapping
    """
    def __init__(self, ref_date, input_data, holidays):
        
        self.refDate = ref_date
        self.instr = input_data['Instrument']
        self.fixDayCount = input_data['DayCountFixed']
        self.floatDayCount = input_data['DayCountFloat']
        self.basisDayCount = input_data['DayCountBasis']
        self.maturity = input_data['Maturity']
        self.tenor = input_data['UnderlyingTenor']
        self.underlyingPayFreq = input_data['UnderlyingPaymentFrequency']
        self.basisTenor = input_data['BasisTenor']
        self.basisPayFreq = input_data['BasisPaymentFrequency']
        self.fixPayFreq = input_data['PaymentFrequencyFixed']
        self.rollConvFloat = input_data['RollConventionFloat']
        self.rollConvFix = input_data['RollConventionFixed']
        self.rollConvBasis = input_data['RollConventionBasis']
        self.spotLag = input_data['SpotLag']
        self.label = self.instr + '_' + self.maturity
        self.currency = input_data['Currency']
        self.holidays = holidays
        self.parRate = input_data['Quote']
        
    def get_instrument(self):
        """
        Instrument specification based on the "Instrument" field the input data
        """
        if self.instr.upper() == Instrument.IRS:
            instrument = self.get_irs_spec()  
        elif self.instr.upper() == Instrument.OIS:
            instrument = self.get_irs_spec()
        elif self.instr.upper() == Instrument.TBS:
            instrument = self.get_tbs_spec() 
        elif self.instr.upper() == Instrument.Deposit:
            instrument = self.get_deposit_spec() 
        elif self.instr.upper() == Instrument.FRA:
            instrument = self.get_fra_spec() 
        else:
            raise ValueError('Unknown instrument type')
        return instrument
    

    def get_irs_spec(self):
        """
        Specification for interest rate swaps
        """
        # get floating leg schedule
        floatleg = self.get_float_leg(self.underlyingPayFreq, self.tenor, self.rollConvFloat, self.spotLag)
        # get fix leg schedule
        fixedleg = self.get_fix_leg(self.fixPayFreq, self.rollConvFix, self.spotLag)
        
        # get expiry of swap (cannot be before last paydate of legs)
        spot_date = get_end_date(self.refDate, self.spotLag)
        expiry = get_end_date(spot_date, self.maturity)

        # SecuritizationLevel is not used in the bootstrapping algorithm
        ir_swap = pyvacon.finance.specification.InterestRateSwapSpecification(
            self.label, 'dummy_issuer', 'COLLATERALIZED', self.currency, expiry, fixedleg, floatleg)
        return ir_swap
    
    def get_tbs_spec(self):
        """
        Specification for tenor basis swaps
        """
        # get floating leg schedule
        floatleg = self.get_float_leg(self.underlyingPayFreq, self.tenor, self.rollConvFloat, self.spotLag)
        floatleg_basis = self.get_float_leg(self.basisPayFreq, self.basisTenor, self.rollConvBasis, self.spotLag)
        
        # get fix leg schedule
        fixedleg = self.get_fix_leg(self.fixPayFreq, self.rollConvFix, self.spotLag)
        
        # get expiry of swap (cannot be before last paydate of legs)
        spot_date = get_end_date(self.refDate, self.spotLag)
        expiry = get_end_date(spot_date, self.maturity)

        # the basis leg should be the pay leg
        basis_swap = pyvacon.finance.specification.InterestRateBasisSwapSpecification(
            self.label, 'dummy_issuer', 'COLLATERALIZED', self.currency, expiry, floatleg_basis, floatleg, fixedleg)
        return basis_swap
    
    
    def get_deposit_spec(self):
        """
        Specification for deposits
        """
        
        # get spot date
        spot_date = get_end_date(self.refDate, self.spotLag)
        # end date of the accrual period
        end_date = get_end_date(spot_date, self.maturity)
        
        # start date of FRA is endDate - tenor
        start_date = get_start_date(end_date, self.tenor)
        
        # specification of the deposit
        deposit = pyvacon.finance.specification.DepositSpecification(
            self.label, 'dummy_issuer', 'NONE', self.currency, self.refDate, start_date, end_date, 100, self.parRate, self.floatDayCount)
        return deposit

    def get_fra_spec(self):
        """
        Specification for FRAs/Futures
        """
        # get spot date
        spot_date = get_end_date(self.refDate,  self.spotLag)
        
        # end date of the accrual period
        end_date = get_end_date(spot_date, self.maturity)

        # start date of FRA is endDate - tenor
        start_date = get_start_date(end_date, self.tenor)

        # expiry of FRA is the fixing date 
        expiry_date = get_start_date(start_date, self.spotLag)

        # specification of the deposit
        fra = pyvacon.finance.specification.InterestRateFutureSpecification(
            self.label, 'dummy_issuer', 'NONE', self.currency, 'dummy_udlId', expiry_date, 100, start_date, end_date, self.floatDayCount)
     
        return fra
    
    def get_float_leg(self, pay_freq, reset_freq, roll_conv, spot_lag ='0D'):
        
        # get swap leg schedule
        flt_schedule = get_schedule(self.refDate, self.maturity, pay_freq, roll_conv, self.holidays, spot_lag)
        
        # get start dates
        flt_start_dates = flt_schedule[:-1]
        
        # get end dates
        flt_end_dates = flt_schedule[1:]
        flt_pay_dates = flt_end_dates
       
        # get reset dates
        flt_reset_schedule = get_schedule(self.refDate, self.maturity, reset_freq, roll_conv, self.holidays, spot_lag)
        flt_reset_dates = flt_reset_schedule[:-1]
         
        flt_notionals = [1.0 for _ in range(len(flt_start_dates))]
        floatleg = pyvacon.finance.specification.IrFloatLegSpecification(
            flt_notionals, flt_reset_dates, flt_start_dates, flt_end_dates, flt_pay_dates, self.currency, 'dummy_undrl', self.floatDayCount, 0.0)
        return floatleg

    def get_fix_leg(self, pay_freq, roll_conv, spot_lag ='0D'):
        # get fix leg schedule
        fix_schedule = get_schedule(self.refDate, self.maturity, pay_freq, roll_conv, self.holidays, spot_lag)
        fix_start_dates = fix_schedule[:-1]
        fix_end_dates = fix_schedule[1:]
        fix_pay_dates = fix_end_dates
        fix_notionals = [1.0 for _ in range(len(fix_start_dates))]
        fixedleg = pyvacon.finance.specification.IrFixedLegSpecification(
            self.parRate, fix_notionals, fix_start_dates, fix_end_dates, fix_pay_dates, self.currency, self.fixDayCount)
        return fixedleg


class Instrument:
    IRS = 'IRS'
    TBS = 'TBS'
    Deposit = 'DEPOSIT'
    OIS = 'OIS'
    FRA = 'FRA'

def get_schedule(ref_date, term, tenor, roll_conv, holidays, spot_lag ='0D', stub_period = False):
    """
    Generates a schedule starting with refDate + spotLag
    """
    # calc schedule start & end dates
    start = get_end_date(ref_date, spot_lag)
    end = get_end_date(start, term)

    # calc schedule period
    period = get_period(tenor)
    schedule_spec = pyvacon.finance.utils.ScheduleSpecification(
        start, end, period, stub_period, roll_conv, holidays)
    schedule_p = schedule_spec.generate()
    schedule = pyvacon.converter.create_datetime_list(schedule_p)
    return schedule


def get_period(tenor):
    t = tenor[-1].upper()
    p = int(tenor[:-1])
    
    if t == 'D':
        result = pyvacon.utilities.Period(0, 0, p)
    elif t == 'M':
        result = pyvacon.utilities.Period(0, p, 0)
    elif t == 'Y':
        result = pyvacon.utilities.Period(p, 0, 0)
    else:
        raise ValueError('Unknown tenor')
    return result

def get_end_date(start_date, term):
    t = term[-1].upper()
    p = int(term[:-1])
    
    if t == 'D':
        result = start_date + relativedelta(days=+p)
    elif t == 'M':
        result = start_date + relativedelta(months=+p)
    elif t == 'Y':
        result = start_date + relativedelta(years=+p)
    else:
        raise ValueError('Unknown term')
    return result

def get_start_date(end_date, term):
    t = term[-1].upper()
    p = int(term[:-1])
    
    if t == 'D':
        result = end_date + relativedelta(days=-p)
    elif t == 'M':
        result = end_date + relativedelta(months=-p)
    elif t == 'Y':
        result = end_date + relativedelta(years=-p)
    else:
        raise ValueError('Unknown term')
    return result
