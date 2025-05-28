# -*- coding: utf-8 -*-


from datetime import datetime, date
from pandas import DataFrame, Series
from typing import Union as _Union
from rivapy.tools.datetools import \
    Period, \
    Schedule, \
    calc_end_day, \
    calc_start_day, \
    term_to_period, \
    datetime_to_date
from rivapy.tools.enums import SecuritizationLevel
from rivapy.tools._converter import \
    converter as _converter, \
    create_ptime as _create_ptime
from pyvacon.instruments.analytics_classes import \
    DepositSpecification, \
    InterestRateBasisSwapSpecification, \
    InterestRateFutureSpecification, \
    InterestRateSwapSpecification, \
    IrFloatLegSpecification, \
    IrFixedLegSpecification
from pyvacon.analytics import \
    ScheduleSpecification, \
    vectorBaseSpecification, \
    vectorDouble, \
    YieldCurveBootstrapper  # do not use method YieldCurveBootstrapper_compute


class Par_IR_Data:
    def __init__(self, data_file: DataFrame, ref_date: _Union[date, datetime] = None):
        """
        Definition of input instruments for IR bootstrapping
        """
        headings = data_file.columns
        if ref_date is not None:
            self.ref_date = datetime_to_date(ref_date)  # mal länge
        else:
            self.ref_date = data_file[intersect(headings, ('Reference Date', 'Ref Date', 'refDate'))]
        self.instrument_type = data_file[intersect(headings, ('Instrument', 'Instrument Type', 'InstrumentType'))]
        self.fix_day_count = data_file[intersect(headings, ('Day Count Fixed', 'DayCountFixed'))]
        self.float_day_count = data_file[intersect(headings, ('Day Count Float', 'DayCountFloat'))]
        self.basis_day_count = data_file[intersect(headings, ('Day Count Basis', 'DayCountBasis'))]
        self.maturity = data_file[intersect(headings, ('Maturity', 'Tenor'))]
        self.fix_pay_freq = data_file[intersect(headings, ('Payment Frequency Fixed', 'PaymentFrequencyFixed'))]
        self.underlying_tenor = data_file[intersect(headings, ('Underlying Tenor', 'UnderlyingTenor'))]
        self.underlying_pay_freq = data_file[intersect(headings, ('Underlying Payment Frequency',
                                                                  'UnderlyingPaymentFrequency'))]
        self.basis_tenor = data_file[intersect(headings, ('Basis Tenor', 'BasisTenor'))]
        self.basis_pay_freq = data_file[intersect(headings, ('Basis Payment Frequency', 'BasisPaymentFrequency'))]
        self.roll_conv_fix = data_file[intersect(headings, ('Roll Convention Fixed', 'RollConventionFixed',
                                                            'Business Day Convention Fixed'))]
        self.roll_conv_fix = data_file[intersect(headings, ('Roll Convention Float', 'RollConventionFloat',
                                                            'Business Day Convention Float'))]
        self.roll_conv_fix = data_file[intersect(headings, ('Roll Convention Basis', 'RollConventionBasis',
                                                            'Business Day Convention Basis'))]
        self.spot_lag = data_file[intersect(headings, ('Spot Lag', 'spotLag'))]
        self.label = self.instrument_type + '_' + self.maturity
        self.currency = data_file[intersect(headings, ('Currency', 'CCY'))]
        self.holidays = data_file[intersect(headings, ('Calendar', 'Holiday Calendar', 'Holidays'))]
        self.par_rate = data_file[intersect(headings, ('Quote', 'Par Rate'))]

    def specify_instrument(self):
        if self.instrument_type.upper() == 'IRS':
            # Interest Rate Swap
            instrument = self.specify_irs()
        elif self.instrument_type.upper() == 'OIS':
            # Overnight Index Swap
            instrument = self.get_irs_spec()
        elif self.instrument_type.upper() == 'TBS':
            # Traded Basis Swap
            instrument = self.get_tbs_spec()
        elif self.instrument_type.upper() == 'DEPOSIT':
            # Deposit
            instrument = self.get_deposit_spec()
        elif self.instrument_type.upper() == 'FRA':
            # Forward Rate Agreement
            instrument = self.get_fra_spec()
        else:
            raise Exception('Unknown instrument type')
        return instrument

    def specify_irs(self):
        # get floating leg schedule
        float_leg_schedule = Schedule()
        float_leg = self.get_float_leg(self.underlying_pay_freq, self.tenor, self.roll_conv_float, self.spot_lag)

        # get fix leg schedule
        fixed_leg = self.get_fix_leg(self.fix_pay_freq, self.roll_conv_fix, self.spot_lag)

        # get expiry of swap (cannot be before last pay date of legs)
        spot_date = calc_end_day(self.ref_date, self.spot_lag)
        expiry_date = calc_end_day(spot_date, self.maturity)
        # expiryDate = _create_ptime(expiry_date)

        # SecuritizationLevel is not used in the bootstrapping algorithm
        ir_swap = InterestRateSwapSpecification(self.label, 'dummy_issuer', SecuritizationLevel.COLLATERALIZED,
                                                self.currency, expiry_date, fixed_leg, float_leg)
        return ir_swap

    def get_tbs_spec(self):
        """
        Specification for tenor basis swaps
        """
        # get floating leg schedule
        float_leg = self.get_float_leg(self.underlying_pay_freq, self.tenor, self.roll_conv_float, self.spot_lag)
        float_leg_basis = self.get_float_leg(self.basis_pay_freq, self.basis_tenor, self.roll_conv_basis, self.spot_lag)

        # get fix leg schedule
        fixed_leg = self.get_fix_leg(self.fix_pay_freq, self.roll_conv_fix, self.spot_lag)

        # get expiry of swap (cannot be before last paydate of legs)
        spot_date = calc_end_day(self.ref_date, self.spot_lag)
        expiry_date = calc_end_day(spot_date, self.maturity)
        # expiryDate = _create_ptime(expiry)

        # the basis leg should be the pay leg
        basis_swap = InterestRateBasisSwapSpecification(self.label, 'dummy_issuer', SecuritizationLevel.COLLATERALIZED,
                                                        self.currency, expiry_date, float_leg_basis, float_leg,
                                                        fixed_leg)
        return basis_swap

    def get_deposit_spec(self):
        """
        Specification for deposits
        """
        # get start date
        start_date = calc_end_day(self.ref_date, self.spot_lag)
        # startDate = _create_ptime(start_date)

        # get end date
        end_date = calc_end_day(start_date, self.maturity)
        # endDate = _create_ptime(end_date)

        # specification of the deposit
        deposit = DepositSpecification(self.label, 'dummy_issuer', SecuritizationLevel.NONE, self.currency,
                                       self.ref_date, start_date, end_date, 100, self.par_rate, self.float_day_count)

        return deposit

    def get_fra_spec(self):
        """
        Specification for FRAs/Futures
        """
        # get spot date
        spot_date = calc_end_day(self.ref_date, self.spot_lag)

        # end date of the accrual period
        end_date = calc_end_day(spot_date, self.maturity)
        # endDate = _create_ptime(end_date)

        # start date of FRA is endDate - tenor
        start_date = calc_start_day(end_date, self.tenor)
        # startDate = _create_ptime(start_date)

        # expiry of FRA is the fixing date
        expiry_date = calc_start_day(start_date, self.spot_lag)
        # expiryDate = _create_ptime(expiry_date)

        # specification of the deposit
        fra = InterestRateFutureSpecification(self.label, 'dummy_issuer', SecuritizationLevel.NONE, self.currency,
                                              'dummy_udlId', expiry_date, 100, start_date, end_date,
                                              self.float_day_count)

        return fra

    def get_float_leg(self, pay_freq, reset_freq, roll_conv, spot_lag='0D'):

        # get swap leg schedule
        flt_schedule = get_schedule(self.ref_date, self.maturity, pay_freq, roll_conv, self.holidays, spot_lag)

        # get start dates
        flt_start_dates = _converter.createPTimeList(self.ref_date, flt_schedule[:-1])

        # get end dates
        flt_end_dates = _converter.createPTimeList(self.ref_date, flt_schedule[1:])
        flt_pay_dates = flt_end_dates

        # get reset dates
        flt_reset_schedule = get_schedule(self.ref_date, self.maturity, reset_freq, roll_conv, self.holidays, spot_lag)
        flt_reset_dates = _converter.createPTimeList(self.ref_date, flt_reset_schedule[:-1])

        flt_notionals = [1.0] * len(flt_start_dates)
        float_leg = IrFloatLegSpecification(flt_notionals, flt_reset_dates, flt_start_dates, flt_end_dates,
                                            flt_pay_dates, self.currency, 'dummy_undrl', self.float_day_count, 0.0)
        return float_leg

    def get_fix_leg(self, pay_freq, roll_conv, spot_lag='0D'):
        # get fix leg schedule
        fix_schedule = get_schedule(self.ref_date, self.maturity, pay_freq, roll_conv, self.holidays, spot_lag)

        # get start dates
        fix_start_dates = _converter.createPTimeList(self.ref_date, fix_schedule[:-1])

        # get end dates
        fix_end_dates = _converter.createPTimeList(self.ref_date, fix_schedule[1:])
        fix_pay_dates = fix_end_dates

        fix_notionals = [1.0] * len(fix_start_dates)
        fixed_leg = IrFixedLegSpecification(self.par_rate, fix_notionals, fix_start_dates, fix_end_dates,
                                            fix_pay_dates, self.currency, self.fix_day_count)
        return fixed_leg


def intersect(header: Series, good_headings: tuple):
    heading = header & good_headings
    if len(heading) != 1:
        raise Exception('The header must include (exactly) one key of ' + str(good_headings) + '!')
    return heading.pop()


def bootstrap_curve(data_file: DataFrame, output_spec: dict) -> DataFrame:
    # get reference date for bootstrapped curve
    # list of accepted keys for reference date
    ref_date_keys = ('ref_date', 'refDate')
    ref_date = ref_date_keys & output_spec.keys()
    if len(ref_date) != 1:
        raise Exception('The output specification must include (exactly) one key '
                        + str(ref_date_keys) + ' defining the reference date!')
    del ref_date_keys
    ref_date = ref_date.pop()
    # refDate = _create_ptime(ref_date)

    # get holiday calendar for bootstrapped curve
    if 'calendar' in output_spec.keys():
        holidays = output_spec['calendar']
    else:
        holidays = None

    # get quotes and instrument specification from input data
    n = len(data_file.index)
    instruments = vectorBaseSpecification(n)
    quotes = vectorDouble(n)

    for i in range(0, n):
        ins = InstrumentSpec(ref_date, data_file.iloc[i, :], holidays)
        instruments[i] = ins.get_instrument()
        quotes[i] = ins.par_rate  # data_file.iloc[i, :]['Quote']
        
    # get discount curve to be used in bootstrapping algorithm
    # list of accepted keys for discount curve
    discount_curve_keys = ('discount_curve', 'discountCurve')
    discount_curve = discount_curve_keys & output_spec.keys()
    if len(discount_curve) == 1:
        discount_curve = discount_curve.pop()
    elif len(discount_curve) == 0:
        discount_curve = None
    else:
        raise Exception('The output specification must not include more than one key '
                        + str(discount_curve_keys) + ' defining the discount curve!')
    del discount_curve_keys

    # get basis curve to be used in bootstrapping algorithm
    # list of accepted keys for basis curve
    basis_curve_keys = ('basis_curve', 'basisCurve')
    basis_curve = basis_curve_keys & output_spec.keys()
    if len(basis_curve) == 1:
        basis_curve = basis_curve.pop()
    elif len(basis_curve) == 0:
        basis_curve = None
    else:
        raise Exception('The output specification must not include more than one key '
                        + str(basis_curve_keys) + ' defining the basis curve!')
    del basis_curve_keys

    curve = YieldCurveBootstrapper.compute(ref_date, output_spec['curveName'], output_spec['dayCount'],  instruments,
                                           quotes, discount_curve, basis_curve)
    return curve


def get_schedule(ref_date, term, tenor, roll_conv, holidays, spot_lag='0D', stub_period=False):
    """
    Generates a schedule starting with refDate + spot_lag
    """
    # calc schedule start & end dates
    start_date = calc_end_day(ref_date, spot_lag)
    end_date = calc_end_day(start_date, term)
    # startDate = _create_ptime(start_date)
    # endDate = _create_ptime(end_date)
    
    # calc schedule period
    period = term_to_period(tenor)
    schedule_spec = ScheduleSpecification(start_date, end_date, period, stub_period, roll_conv, holidays)
    schedule_p = schedule_spec.generate()
    # schedule = _converter.create_datetime_list(schedule_p)
    return schedule_p
