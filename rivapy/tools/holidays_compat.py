# Compatibility wrapper for different versions of the `holidays` package.
# It exposes a small, stable surface used across the codebase: HolidayBase, ECB
# (European Central Bank calendar), EuropeanCentralBank (alias), country_holidays
# and common country classes where available.

try:
    # Newer/standard API
    from holidays import HolidayBase, country_holidays

    # Many installations provide ECB; some versions provide EuropeanCentralBank
    try:
        from holidays import ECB
    except Exception:
        ECB = None
    try:
        from holidays import EuropeanCentralBank
    except Exception:
        # alias ECB if available
        EuropeanCentralBank = ECB
    # optional country classes
    try:
        from holidays import UnitedStates, Germany
    except Exception:
        UnitedStates = None
        Germany = None
    # provide short aliases expected by tests
    DE = Germany
    US = UnitedStates
except Exception:
    # If the holidays package is not available, expose minimal placeholders so
    # imports do not fail at module import time. Attempting to use these
    # placeholders will raise informative ImportErrors later.
    class HolidayBase:  # type: ignore
        """Placeholder base class when `holidays` is not installed."""

    def country_holidays(country):
        raise ImportError("`holidays` package not installed or does not provide country_holidays")

    ECB = None
    EuropeanCentralBank = None
    UnitedStates = None
    Germany = None
    DE = None
    US = None
