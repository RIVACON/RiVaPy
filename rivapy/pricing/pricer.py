from rivapy import _pyvacon_available
if _pyvacon_available:
    from pyvacon.finance.pricing import BasePricer as _BasePricer

def price(pr_data):
    if hasattr(pr_data, 'price'):
        return pr_data.price()
    else:
        if _pyvacon_available:
            return _BasePricer.price(pr_data)
        else:
            raise Exception('Cannot price: pyvacon not available.')