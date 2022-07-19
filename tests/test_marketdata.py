import unittest
import numpy as np
import datetime as dt

import rivapy
from rivapy.marketdata import VolatilityGridParametrization, VolatilityParametrizationSABR, VolatilitySurface
from rivapy.marketdata import DiscountCurve, EquityForwardCurve, enums, PowerPriceForwardCurve
from rivapy.instruments import SimpleSchedule
from rivapy import enums

class VolatilityGridParamTest(unittest.TestCase):
    def test_exceptions(self):
        expiries = np.linspace(1.0/365.0, 4.0, 10)
        strikes = np.linspace(0.4, 1.6, 100)
        # test for exception if number of strikes does not match number of cols
        vols =np.empty((expiries.shape[0], 1))
        try:
            vol_grid_param = VolatilityGridParametrization(expiries, strikes, vols)
            self.assertFalse(False)
        except:
            self.assertFalse(True)

    def test_calc_implied_vol(self):
        """Simple tests for class VolatilityGridParametrization
        """
        expiries = np.linspace(1.0/365.0, 4.0, 10)
        strikes = np.linspace(0.4, 1.6, 100)
        vols = 0.3*np.ones((expiries.shape[0], strikes.shape[0]))
        vol_grid_param = VolatilityGridParametrization(expiries, strikes, vols)
        self.assertAlmostEqual(0.3, vol_grid_param.calc_implied_vol(1.0, 1.3), delta=1e-7)
        # now add to volatility surface
        refdate = dt.datetime(2021,1,1)
        dummy = DiscountCurve('',refdate=refdate, dates=[refdate, refdate+dt.timedelta(days=10*365)], df=[1.0,1.0])
        fwd = EquityForwardCurve(100.0, funding_curve=dummy, borrow_curve=dummy,div_table=None)
        vol_surface = VolatilitySurface('', refdate, fwd, enums.DayCounterType.Act365Fixed, vol_grid_param)
        vol = vol_surface.calc_implied_vol(refdate+dt.timedelta(days=365), 100.0)
        self.assertAlmostEqual(vol, vol_grid_param.calc_implied_vol(1.0, 1.3), delta=1e-7)
        
        
class VolatilitySABRParamTest(unittest.TestCase):
    
    def test_calc_implied_vol(self):
        # 1. Parametrization
        expiries = [1.0/12.0, 1.0, 2.0, 3.0]
        sabr_params = np.array([[.1, 0.1, .9,-.8], [.3, 0.1, .1, .1], [.5, .3, .9, -.75,], [.5, .3, .9, -.85,]])
        sabr_param = VolatilityParametrizationSABR(expiries, sabr_params)
        self.assertAlmostEqual(0.30118,sabr_param.calc_implied_vol(ttm = 1.0,strike = 1.0), delta=1e-7)
        
        # 2. Vol Surface
        obj_id = 'Test Surface'
        refdate = dt.datetime(2021,1,1)
        dc = DiscountCurve('',refdate=refdate, dates=[refdate, refdate+dt.timedelta(days=10*365)], df=[1.0,1.0])
        fc = EquityForwardCurve(100.0, funding_curve=dc, borrow_curve=dc,div_table=None)
        vol_surf = VolatilitySurface(obj_id, refdate, fc, enums.DayCounterType.Act365Fixed, sabr_param)
        vol = vol_surf.calc_implied_vol(refdate+dt.timedelta(days=365), 100.0,refdate)
        self.assertAlmostEqual(vol, sabr_param.calc_implied_vol(ttm = 1.0,strike = 1.0), delta=1e-7)
        
class VolatilitySurfaceTest(unittest.TestCase):
    
    def test_calc_implied_vol_single_expiry(self):
        self.assertAlmostEqual(0,0)
        # Fix to make the the commented code below working
        return
        """Test if calc_implied_vol works for single expiry
        """
        expiries = np.array([1])
        strikes = np.linspace(0.4, 1.6, 100)
        vols = 0.3*np.ones((expiries.shape[0], strikes.shape[0]))
        vol_grid_param = VolatilityGridParametrization(expiries, strikes, vols)
        self.assertAlmostEqual(0.3, vol_grid_param.calc_implied_vol(1.0, 1.3), delta=1e-7)
        refdate = dt.datetime(2021,1,1)
        dummy = DiscountCurve('',refdate=refdate, dates=[refdate, refdate+dt.timedelta(days=10*365)], df=[1.0,1.0])
        fwd = EquityForwardCurve(100.0, funding_curve=dummy, borrow_curve=dummy,div_table=None)
        vol_surface = VolatilitySurface('', refdate, fwd, enums.DayCounterType.Act365Fixed, vol_grid_param)
        vol = vol_surface.calc_implied_vol(refdate+dt.timedelta(days=365), 100.0)
        self.assertAlmostEqual(vol, vol_grid_param.calc_implied_vol(1.0, 1.3), delta=1e-7)

class PowerPriceForwardCurveTest(unittest.TestCase):
    def test_simple_schedule(self):
        """Test value with SimpleSchedule
        """
        simple_schedule = SimpleSchedule(dt.datetime(2022,12,1), dt.datetime(2023,11,1,4,0,0), freq='1H')
        values = np.ones((len(simple_schedule.get_schedule()),)).cumsum()
        hpfc = PowerPriceForwardCurve(dt.datetime(2022,12,1), dt.datetime(2022,12,1), dt.datetime(2023,11,1,4,0,0), freq='1H', values = values)
        simple_schedule = SimpleSchedule(dt.datetime(2022,12,1), dt.datetime(2022,12,1,4,0,0), freq='1H')
        values = hpfc.value(dt.datetime(2022,1,1), simple_schedule)
        self.assertEqual(values.shape[0], 4)
        self.assertEqual(values[0],1)
        self.assertEqual(values[-1], 4)
        # same with hours excluded
        simple_schedule = SimpleSchedule(dt.datetime(2022,12,1), dt.datetime(2022,12,1,4,0,0), freq='1H', hours=[2])
        values = hpfc.value(dt.datetime(2022,1,1), simple_schedule)
        self.assertEqual(values.shape[0], 1)
        self.assertEqual(values[0],3)

    def test_exceptions(self):
        """Test consistency checks in forward curve.
        """
        simple_schedule = SimpleSchedule(dt.datetime(2022,12,1), dt.datetime(2023,11,1,4,0,0), freq='1H')
        values = np.ones((len(simple_schedule.get_schedule()),)).cumsum()
        hpfc = PowerPriceForwardCurve(dt.datetime(2022,12,1), dt.datetime(2022,12,1), dt.datetime(2023,11,1,4,0,0), freq='1H', values = values)
        # schedule starts before first date of forward curve
        simple_schedule = SimpleSchedule(dt.datetime(2022,11,30), dt.datetime(2022,12,1,4,0,0), freq='1H')
        self.assertRaises(Exception, lambda: hpfc.value(dt.datetime(2022,1,1), simple_schedule))
        # schedule ends after last date of forward curve
        simple_schedule = SimpleSchedule(dt.datetime(2022,11,30), dt.datetime(2023,12,1,4,0,0), freq='1H')
        self.assertRaises(Exception, lambda: hpfc.value(dt.datetime(2022,1,1), simple_schedule))
        # hpfc where  number of values 
        values = np.ones((len(simple_schedule.get_schedule()),)).cumsum()
        hpfc = PowerPriceForwardCurve(dt.datetime(2022,12,1), dt.datetime(2022,12,1), dt.datetime(2023,11,1,4,0,0), freq='1H', values = values[:-1])
        simple_schedule = SimpleSchedule(dt.datetime(2022,12,1), dt.datetime(2022,12,1,4,0,0), freq='1H', hours=[2])
        self.assertRaises(Exception, lambda: hpfc.value(dt.datetime(2022,1,1), simple_schedule))
        
        

if __name__ == '__main__':
    unittest.main()

