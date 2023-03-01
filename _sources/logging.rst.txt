Logging
=========================
Usage
------------------------------
The rivapy package provides logging using the standard python logging module. Here, a separate logger for each submodule 
exists:

    * rivapy.instruments
    * rivapy.marketdata
    * rivapy.models
    * rivapy.numerics
    * rivapy.pricing
    * rivapy.sample_data
    * rivapy.tools

So if you just want to switch on logging globally, you may just use the usual logic

>>> import logging
>>> logging.basicConfig(level=logging.DEBUG, format="%(asctime)s  - %(levelname)s - %(filename)s:%(lineno)s - %(message)s ")

In some circumstances, it may be useful to set different loglevels for the different modules. Here,
one can use the usual logic, e.g.

>>> logger = logging.getLogger('rivapy.pricing')
>>> logger.setLevel(logging.ERROR)

to set the loglevel of the rivapy.pricing module.

Developer notes
------------------------
To apply the logging in your code, just import the logger from the _logger.py file in the respective module you are currently developing for.
So, if you are currently working in rivapy.pricing, just use the following line in your file

>>> from rivapy.pricing._logger import logger

to retrieve the correct logger.