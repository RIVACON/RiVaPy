# from pyvacon.pricing import *
# del price

# __all__ = ['pricer', 'pricing_data', 'pricing_request']
from rivapy.pricing.bond_pricing import *
from rivapy import _pyvacon_available

# if _pyvacon_available:
# 	from rivapy.pricing.pricing_data import CDSPricingData
# 	from rivapy.pricing.pricing_data import Black76PricingData, ResultType, AmericanPdePricingData

# from rivapy.tools.factory import _factory
from rivapy.pricing.factory import _factory
from rivapy.pricing.deposit_pricing import DepositPricer
from rivapy.pricing.fra_pricing import ForwardRateAgreementPricer
from rivapy.pricing.interest_rate_swap_pricing import InterestRateSwapPricer


if _pyvacon_available:
    # from pyvacon.finance.pricing import *
    from pyvacon.finance.pricing import BasePricer

    def price(pr_data):
        if hasattr(pr_data, "price"):
            return pr_data.price()
        else:
            return BasePricer.price(pr_data)

else:

    def price(pr_data):
        if hasattr(pr_data, "price"):
            return pr_data.price()
        raise Exception("Pricing of " + type(pr_data).__name__ + " not possible without pyvacon.")


def _add_to_factory(cls):
    factory_entries = _factory()
    factory_entries[cls.__name__] = cls


_add_to_factory(DepositPricer)
_add_to_factory(ForwardRateAgreementPricer)
_add_to_factory(InterestRateSwapPricer)

if __name__ == "__main__":
    pass
