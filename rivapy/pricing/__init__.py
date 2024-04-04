# from pyvacon.pricing import *
# del price

# __all__ = ['pricer', 'pricing_data', 'pricing_request']
from rivapy.pricing.bond_pricing import *
from rivapy import _pyvacon_available

if _pyvacon_available:
    from rivapy.pricing.pricing_data import CDSPricingData
    from rivapy.pricing.pricing_data import Black76PricingData, ResultType, AmericanPdePricingData


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
