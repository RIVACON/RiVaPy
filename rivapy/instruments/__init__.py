from rivapy.instruments.factory import _factory
from rivapy.instruments.specifications import *
from rivapy.instruments.components import Issuer
from rivapy.instruments.specifications import HasExpectedCashflows
from rivapy.instruments.cds_specification import CDSSpecification
from rivapy.instruments.ppa_specification import PPASpecification, GreenPPASpecification

# from rivapy.instruments.bond_specifications import (
#     ZeroCouponBondSpecification,
#     FixedRateBondSpecification,
#     PlainVanillaCouponBondSpecification,
#     FloatingRateNoteSpecification,
#     FixedToFloatingRateNoteSpecification,
# )
from rivapy.instruments.cashflow import CashFlow
from rivapy.instruments.energy_futures_specifications import EnergyFutureSpecifications
from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.instruments.fra_specifications import ForwardRateAgreementSpecification
from rivapy.instruments.ir_swap_specification import (
    IrSwapLegSpecification,
    IrFixedLegSpecification,
    IrFloatLegSpecification,
    IrOISLegSpecification,
    InterestRateSwapSpecification,
)


def _add_to_factory(cls):
    factory_entries = _factory()
    factory_entries[cls.__name__] = cls


_add_to_factory(Issuer)
_add_to_factory(PPASpecification)
_add_to_factory(GreenPPASpecification)
# _add_to_factory(ZeroCouponBondSpecification)
# _add_to_factory(FixedRateBondSpecification)
# _add_to_factory(PlainVanillaCouponBondSpecification)
_add_to_factory(EnergyFutureSpecifications)
_add_to_factory(DepositSpecification)
_add_to_factory(ForwardRateAgreementSpecification)
_add_to_factory(IrSwapLegSpecification)
_add_to_factory(IrFixedLegSpecification)
_add_to_factory(IrFloatLegSpecification)
_add_to_factory(IrOISLegSpecification)
_add_to_factory(InterestRateSwapSpecification)
_add_to_factory(CashFlow)
