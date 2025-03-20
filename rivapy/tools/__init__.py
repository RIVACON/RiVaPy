# -*- coding: utf-8 -*-
from rivapy.tools.factory import _factory
from rivapy.tools.datetools import Period, Schedule
from rivapy.tools.datetime_grid import DateTimeGrid
from rivapy.tools.scheduler import SimpleSchedule, PeakSchedule, OffPeakSchedule, GasSchedule

# __all__ = ['_converter', '_validators', 'datetools', 'enums']


def _add_to_factory(cls):
    factory_entries = _factory()
    factory_entries[cls.__name__] = cls


_add_to_factory(SimpleSchedule)
_add_to_factory(PeakSchedule)
_add_to_factory(OffPeakSchedule)
_add_to_factory(GasSchedule)
