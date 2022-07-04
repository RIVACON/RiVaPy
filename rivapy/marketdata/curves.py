from typing import List, Union, Tuple
from enum import Enum
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import math
import dateutil.relativedelta as relativedelta
from typing import List
import scipy.optimize
import pandas as pd
import numpy as np

from rivapy.enums import DayCounterType, InterpolationType, ExtrapolationType

from rivapy import _pyvacon_available
if _pyvacon_available:
    from pyvacon.finance.marketdata import EquityForwardCurve as _EquityForwardCurve
    from pyvacon.finance.marketdata import SurvivalCurve as _SurvivalCurve
    from pyvacon.finance.marketdata import DiscountCurve as _DiscountCurve
    import pyvacon as _pyvacon

class DiscountCurve:

    def __init__(self, 
                id: str,
                refdate: Union[datetime, date], 
                dates: List[Union[datetime, date]], 
                df: List[float],
                interpolation: InterpolationType = InterpolationType.HAGAN_DF,
                extrapolation: ExtrapolationType = ExtrapolationType.NONE,
                daycounter: DayCounterType = DayCounterType.Act365Fixed):
        """Discountcurve

        Args:
            id (str): Identifier of the discount curve.
            refdate (Union[datetime, date]): Reference date of the discount curve.
            dates (List[Union[datetime, date]]): List of dates belonging to the list of discount factors. All dates must be distinct and equal or after the refdate, otherwise an exception will be thrown.
            df (List[float]): List of discount factors. Length of list of discount factors must equal to length of list of dates, otherwise an exception will be thrown.
            interpolation (enums.InterpolationType, optional): Defaults to InterpolationType.HAGAN_DF.
            extrapolation (enums.ExtrapolationType, optional): Defaults to ExtrapolationType.NONE which does not allow to compute a discount factor for a date past all given dates given to this constructor.
            daycounter (enums.DayCounterType, optional): Daycounter used within the interpolation formula to compute a discount factor between two dates from the dates-list above. Defaults to DayCounterType.Act365Fixed.

        Raises:
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
        """
        if len(dates) < 1:
            raise Exception('Please specify at least one date and discount factor')
        if len(dates) != len(df):
            raise Exception('List of dates and discount factors must have equal length.')
        self.values = sorted(zip(dates,df), key=lambda tup: tup[0]) # zip dates and discount factors and sort by dates
        if isinstance(refdate, datetime):
            self.refdate = refdate
        else:
            self.refdate = datetime(refdate,0,0,0)
        if not isinstance(interpolation, InterpolationType):
            raise TypeError('Interpolation is not of type enums.InterpolationType')
        self.interpolation = interpolation
        if not isinstance(extrapolation, ExtrapolationType):
            raise TypeError('Extrapolation is not of type enums.ExtrapolationType')
        self.extrapolation = extrapolation
        if not isinstance(daycounter, DayCounterType):
            raise TypeError('Daycounter is not of type enums.DaycounterType')
        self.daycounter = daycounter
        self.id = id
        #check if dates are monotonically increasing and if first date is greather then refdate
        if self.values[0][0] < refdate:
            raise Exception('First date must be equal or greater then reference date.')
        if self.values[0][0] > refdate:
            self.values = [(self.refdate, 1.0)] + self.values
        if self.values[0][1] != 1.0:
            raise Exception('Discount factor for today must equal 1.0.')
        for i in range(1,len(self.values)):
            if self.values[i-1]>= self.values[i]:
                raise Exception('Dates must be given in monotonically increasing order.')
        self._pyvacon_obj = None

    def get_dates(self)->Tuple[datetime]:
        """Return list of dates of curve

        Returns:
            Tuple[datetime]: List of dates
        """
        x,y = zip(*self.values)
        return x

    def get_df(self)->Tuple[float]:
        """Return list of discount factors

        Returns:
            Tuple[float]: List of discount factors
        """
        x,y = zip(*self.values)
        return y

    def value(self, refdate: Union[date, datetime], d: Union[date, datetime])->float:
        """Return discount factor for a given date

        Args:
            refdate (Union[date, datetime]): The reference date. If the reference date is in the future (compared to the curves reference date), the forward discount factor will be returned.
            d (Union[date, datetime]): The date for which the discount factor will be returned

        Returns:
            float: discount factor
        """
        if not isinstance(refdate, datetime):
            refdate = datetime(refdate,0,0,0)
        if not isinstance(d, datetime):
            d = datetime(d,0,0,0)
        if refdate < self.refdate:
            raise Exception('The given reference date is before the curves reference date.')
        return self._get_pyvacon_obj().value(refdate, d)

    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _DiscountCurve(self.id, self.refdate, 
                                            [x for x in self.get_dates()], [x for x in self.get_df()], 
                                            self.daycounter, 
                                            self.interpolation,
                                            self.extrapolation)
        return self._pyvacon_obj

    def plot(self, days:int = 10, discount_factors: bool = False, **kwargs):
        """Plots the discount curve using matplotlibs plot function.
        The timegrid includes the dates of the discount curve. Here either the discount factors or the zero rates (continuously compounded, ACT365 yearfraction) are plotted.

        Args:
            days (int, optional): The number of days between two plotted rates/discount factors. Defaults to 10.
            discount_factors (bool, optional): If True, discount factors will be plotted, otherwise the rates. Defaults to False.
            **kwargs: optional arguments that will be directly passed to the matplotlib plto function
        """
        dates = self.get_dates()
        dates_new = [dates[0]]
        for i in range(1,len(dates)):
            while dates_new[-1] + timedelta(days=days) < dates[i]:
                dates_new.append(dates_new[-1]+ timedelta(days=days))
        dates_new.append(dates[-1])
        values = [self.value(self.refdate, d) for d in dates_new]

        if not discount_factors:
            for i in range(1,len(values)):
                dt = float((dates_new[i]-self.refdate).days)/365.0
                values[i] = -math.log(values[i])/dt
        values[0] = values[1]
        plt.plot(dates_new, values, label=self.id, **kwargs)
            

class EquityForwardCurve:
    def __init__(self, 
                    spot: float, 
                    funding_curve: DiscountCurve, 
                    borrow_curve: DiscountCurve, 
                    div_table):
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
            if hasattr(self.div, 'refdate'):
                if self.refdate < self.div.refdate:
                    self.refdate = self.div.refdate

    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            if hasattr(self.fc, '_get_pyvacon_obj'):
                fc = self.fc._get_pyvacon_obj()
            else:
                fc = self.fc
            
            if hasattr(self.bc, '_get_pyvacon_obj'):
                bc = self.bc._get_pyvacon_obj()
            else:
                bc = self.bc

            if hasattr(self.div, '_get_pyvacon_obj'):
                div = self.div._get_pyvacon_obj()
            else:
                div = self.div
            self._pyvacon_obj = _EquityForwardCurve(self.refdate, self.spot, fc, bc, div)

        return self._pyvacon_obj
           
    def value(self, refdate, expiry):
        return self._get_pyvacon_obj().value(refdate, expiry)

    def plot(self, days:int = 10, days_end: int = 10*365, **kwargs):
        """Plots the forward curve using matplotlibs plot function.
        
        Args:
            days (int, optional): The number of days between two plotted rates/discount factors. Defaults to 10.
            days_end (int. optional): Number of days when plotting will end. Defaults to 10*365 (10yr)
            **kwargs: optional arguments that will be directly passed to the matplotlib plto function
        """
        dates = [self.refdate + timedelta(days=i) for i in range(0, days_end, days)]
        values = [self.value(self.refdate, d) for d in dates]
        plt.plot(dates, values, **kwargs)
        plt.xlabel('expiry')
        plt.ylabel('forward value')

class BootstrapHazardCurve:
    def __init__(self, 
                    ref_date: datetime, 
                    trade_date: datetime,
                    dc: DiscountCurve,
                    RR: float,
                    payment_dates: List[datetime],
                    market_spreads: List[float] ):
        """[summary]

        Args:
            ref_date (datetime): [description]
            trade_date (datetime): [description]
            dc (DiscountCurve): [description]
            RR (float): [description]
            payment_dates (List[datetime]): [description]
            market_spreads (List[float]): [description]
        """                      

        self.ref_date=ref_date
        self.trade_date=trade_date
        self.dc=dc
        self.RR=RR
        self.payment_dates_bootstrapp=payment_dates
        self.market_spreads=market_spreads
        self._pyvacon_obj = None

    def par_spread(self, dc_survival, maturity_date, payment_dates: List[datetime]):
        integration_step= relativedelta.relativedelta(days=365)
        premium_period_start = self.ref_date
        prev_date=self.ref_date
        current_date=min(prev_date+integration_step, maturity_date)
        dc_valuation_date=self.dc.value(self.ref_date, maturity_date)
        risk_adj_factor_protection=0
        risk_adj_factor_premium=0
        risk_adj_factor_accrued=0

        while current_date <= maturity_date:
            default_prob = dc_survival.value(self.ref_date, prev_date)-dc_survival.value(self.ref_date, current_date)
            risk_adj_factor_protection += self.dc.value(self.ref_date, current_date) * default_prob
            prev_date = current_date
            current_date += integration_step
        
        if prev_date < maturity_date and current_date > maturity_date:
            default_prob = dc_survival.value(self.ref_date, prev_date)-dc_survival.value(self.ref_date, maturity_date)
            risk_adj_factor_protection += self.dc.value(self.ref_date, maturity_date)  * default_prob

        for premium_payment in payment_dates:
            if premium_payment >= self.ref_date:
                period_length = ((premium_payment-premium_period_start).days)/360
                survival_prob = (dc_survival.value(self.ref_date, premium_period_start)+dc_survival.value(self.ref_date, premium_payment))/2
                df = self.dc.value(self.ref_date, premium_payment)
                risk_adj_factor_premium += period_length*survival_prob*df
                default_prob = dc_survival.value(self.ref_date, premium_period_start)-dc_survival.value(self.ref_date, premium_payment)
                risk_adj_factor_accrued += period_length*default_prob*df
                premium_period_start = premium_payment

        PV_accrued=((1/2)*risk_adj_factor_accrued)
        PV_premium=(1)*risk_adj_factor_premium
        PV_protection=(((1-self.RR))*risk_adj_factor_protection)
        
        par_spread_i=(PV_protection)/((PV_premium+PV_accrued))
        return par_spread_i

    def create_survival(self, dates: List[datetime], hazard_rates: List[float]):
        return _SurvivalCurve('survival_curve', self.refdate, dates, hazard_rates)
    
    def calibration_error(x, self, mkt_par_spread, ref_date, payment_dates, dates, hazard_rates):
        hazard_rates[-1] = x
        maturity_date = dates[-1]
        dc_surv = self.create_survival(ref_date, dates, hazard_rates)
        return  mkt_par_spread - self.par_spread(dc_surv, maturity_date, payment_dates)


    def calibrate_hazard_rate(self):
        sc_dates=[self.ref_date]
        hazard_rates=[0.0]
        for i in range(len(self.payment_dates_bootstrapp)):
            payment_dates_iter = self.payment_dates_bootstrapp[i]
            mkt_par_spread_iter = self.market_spreads[i]
            sc_dates.append(payment_dates_iter[-1])
            hazard_rates.append(hazard_rates[-1])
            sol=scipy.optimize.root_scalar(self.calibration_error,args=(mkt_par_spread_iter, self.ref_date, 
                            payment_dates_iter, sc_dates, hazard_rates),method='brentq',bracket=[0,3],xtol=1e-8,rtol=1e-8)
            hazard_rates[-1] = sol.root
        return  hazard_rates, sc_dates #self.create_survival(self.ref_date, sc_dates, hazard_rates)#.value, hazard_rates

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
    def __init__(self, id: str,
                refdate: Union[datetime, date], 
                start: datetime, 
                end: datetime, 
                values: np.ndarray,
                freq: str='1H',
                tz: str=None):
        """Simple forward curve for power.

        Args:
            id (str): Identifier for the curve. It has no impact on the valuation functionality.
            refdate (Union[datetime, date]): Reference date of curve
            start (dt.datetime): Start of forward curve datetimepoints (including this timepoint).
			end (dt.datetime): End of forad curve datetimepoints (excluding this timepoint).
            values (np.ndarray): One dimensional array holding the price for each datetimepint in the curve. The method value will raise an exception if the number of values is not equal to the number of datetimepoints.
			freq (str, optional): Frequency of timepoints. Defaults to '1H'. See documentation for pandas.date_range for further details on freq.
			tz (str or tzinfo): Time zone name for returning localized datetime points, for example ‘Asia/Hong_Kong’. 
								By default, the resulting datetime points are timezone-naive. See documentation for pandas.date_range for further details on tz.
        """
        self.id = id
        self.refdate = refdate
        self.start = start
        self.end = end
        self.freq = freq
        self.tz = tz
        self.values = values
        # timegrid used to compute prices for a certain schedule
        self._tg = None
        
    def value(self, refdate: Union[date, datetime], schedule)->np.ndarray:
        if self._tg is None:
            self._tg = pd.DataFrame({'dates': pd.date_range(self.start, self.end, freq=self.freq, tz=self.tz, closed='left').to_pydatetime()}).reset_index()
            if self._tg.shape[0] != self.values.shape[0]:
                raise Exception('The number of dates (' + str(self._tg.shape[0])+') does not equal number of values (' + str(self.values.shape[0]) + ') in forward curve.')
        tg = self._tg[(self._tg.dates>=schedule.start)&(self._tg.dates<schedule.end)].set_index('dates')
        _schedule = pd.DataFrame({'dates': schedule.get_schedule(refdate)})
        tg = _schedule.join(tg, on='dates')
        #tg = tg[tg['dates']>=refdate]
        if tg['index'].isna().sum()>0:
            raise Exception('There are ' + str(tg['index'].isna().sum()) + ' dates in the schedule not covered by the forward curve.')
        return self.values[tg['index'].values]
        

        
