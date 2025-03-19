from rivapy.marketdata_tools.factory import _factory
from rivapy.marketdata_tools.pfc_shifter import PFCShifter


def _add_to_factory(cls):
    factory_entries = _factory()
    factory_entries[cls.__name__] = cls


_add_to_factory(PFCShifter)


if __name__ == "__main__":
    pass
