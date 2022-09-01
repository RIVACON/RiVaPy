Models
===========================

Equity
-------------------------------

Local Volatility Model
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rivapy.models.LocalVol
   :members:
   :undoc-members:
   :show-inheritance:

Heston Model
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rivapy.models.HestonModel
   :members:
   :undoc-members:
   :show-inheritance:

Scott Chesney
^^^^^^^^^^^^^^^^^^^^^^^^
Scott Chesney Model is a stochastic volatility model of the form

   .. math:: dS =  e^y S dW_S
   .. math:: dy = \kappa (\theta-y)dt \alpha dW_y
   .. math:: E[dW_S\dot dW_y] = \rho dt

.. autoclass:: rivapy.models.ScottChesneyModel
   :members:
   :undoc-members:
   :show-inheritance:

Stochastic Local Volatility Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rivapy.models.StochasticLocalVol
    :members:
    :undoc-members:
    :show-inheritance:
