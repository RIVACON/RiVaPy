Energy and Weather
================================

Wind
----------------------------------
The WindPowerModels model the total production of power from wind in percentage of the total wind capacity. That means that all simulated values that will be returned by these models are between 0 and 1.

Instantenous Production Models (Spot Models)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rivapy.models.WindPowerModel
   :members:
   :undoc-members:
   :show-inheritance:

Forecast Models (Forward Models)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

WindPowerForecastModel
""""""""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: rivapy.models.WindPowerForecastModel
   :members:
   :undoc-members:
   :show-inheritance:

MultiRegionWindForecastModel
""""""""""""""""""""""""""""""""""""""""""""""""
.. autoclass:: rivapy.models.MultiRegionWindForecastModel
   :members:
   :undoc-members:
   :show-inheritance:

   

Solar
------------------------------------

SolarPowerModel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rivapy.models.SolarPowerModel
   :members:
   :undoc-members:
   :show-inheritance:


Power
----------------------------------------



ResidualDemandModel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rivapy.models.ResidualDemandModel
   :members:
   :undoc-members:
   :show-inheritance:


SimpleRenewableModel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rivapy.models.ResidualDemandForwardModel
   :members:
   :undoc-members:
   :show-inheritance:

LoadModel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rivapy.models.LoadModel
   :members:
   :undoc-members:
   :show-inheritance:

SupplyFunction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rivapy.models.SupplyFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. footbibliography::