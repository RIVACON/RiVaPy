from rivapy.marketdata_tools.factory import _factory
from rivapy.marketdata_tools.pfc_shifter import PFCShifter
from rivapy.marketdata_tools.pfc_shaper import PFCShaper, CategoricalRegression, SimpleCategoricalRegression


def _add_to_factory(cls):
    factory_entries = _factory()
    factory_entries[cls.__name__] = cls


_add_to_factory(PFCShifter)
_add_to_factory(PFCShaper)
_add_to_factory(CategoricalRegression)
_add_to_factory(SimpleCategoricalRegression)


if __name__ == "__main__":
    pass
