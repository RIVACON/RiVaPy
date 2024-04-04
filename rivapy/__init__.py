import warnings

_pyvacon_available = False
try:
    import pyvacon

    _pyvacon_available = True
    import pyvacon.version as version

    if version.is_beta:
        warnings.warn("Imported pyvacon is just beta version.")
except Exception as e:
    warnings.warn(
        "The pyvacon module is not available. You may not use all functionality without this module. Consider installing pyvacon."
    )


from rivapy.tools import enums
import rivapy.instruments as instruments
import rivapy.pricing as pricing
import rivapy.marketdata as marketdata
import rivapy.models as models

# import rivapy
