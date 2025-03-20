Sample and Test Data
================================
The package provides some functionality to create test data for the different classes. 
For example, for first simple tests of some fancy calibration method one would like to have a 
bunch of instruments together with certain prices. 
Here, the methods for the creation may be of special use.

Some classes provide a _create_sample method. This method can be used to create a sample of the respective classes.


Spreadcurves
---------------------------------
.. autoclass:: rivapy.sample_data.market_data.spread_curves.SpreadCurveSampler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: rivapy.sample_data.market_data.spread_curves.SpreadCurveCollection
   :members:
   :undoc-members:
   :show-inheritance:

Credit Default Data
---------------------------------------
.. autoclass:: rivapy.sample_data.market_data.credit_default.CreditDefaultData
   :members:
   :undoc-members:
   :show-inheritance:

Dummy Power Spot Price
---------------------------------------
.. autofunction:: rivapy.sample_data.dummy_power_spot_price.spot_price_model
