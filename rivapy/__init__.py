import warnings
import pyvacon.version as version
#from pyvacon.analytics import setLogLevel as set_log_level
#from pyvacon.analytics import registerSerialization as _register_serialization
#import pyvacon.analytics as _analytics

#_register_serialization('depp')
_pyvacon_available = True
if version.is_beta:
    warnings.warn('Imported pyvacon is just beta version.')


from rivapy import enums
import rivapy.instruments as instruments
import rivapy.pricing as pricing
import rivapy.marketdata as marketdata
#import rivapy


