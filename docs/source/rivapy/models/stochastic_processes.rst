Stochastic Processes
===================================

Ornstein-Uhlenbeck
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: rivapy.models.OrnsteinUhlenbeck
   :members:
   :undoc-members:
   :show-inheritance:


Lucia-Schwartz
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Lucia-Schwartz two-factor model for power prices is a mathematical model used to explain the behavior of electricity prices. The model assumes that electricity prices are determined by two 
factors: a short-term factor and a long-term factor. It was introduced in discussed in :footcite:t:`Lucia2002`.
The short-term factor is related to the current supply and demand conditions in the electricity market. This factor is influenced by factors such as weather conditions, maintenance schedules, and unexpected events such as power plant failures. 
The short-term factor is modeled using a mean-reverting process, which means that prices tend to move back towards their long-term average over time.
The long-term factor is related to the overall trends in the economy and the energy market. This factor is influenced by factors such as changes in 
fuel prices, technological advances, and government policies. The long-term factor is modeled using a random walk process, which means that prices tend to move in a random and unpredictable manner over time.
Overall, the Lucia-Schwartz two-factor model provides a useful framework for understanding the complex factors that influence electricity prices. By separating short-term and long-term factors, 
the model can help energy traders and analysts make more informed decisions about their 
trading strategies and risk management.

.. autoclass:: rivapy.models.LuciaSchwartz
   :members:
   :undoc-members:
   :show-inheritance:



.. footbibliography::
