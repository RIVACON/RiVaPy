Developer Guide
================================
This section contains some rules and best practices developers for rivapy should consider. It is not meant to be a strict rule set and there may be good reasons why someone decides to deviate from some rule. But please think about the points here when developing. And if you think one of the rules should be modified or deleted, just rais an issue ;-)

Styleguide
-------------------------------
We try to follow the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_  writing the code. We cite some of the points in the aformentioned document that we think are most relevant for developers of rivapy in the following but if you are interested in more depth, read the original document.

Imports
^^^^^^^^^^^^
Use import statements for packages and modules only, not for individual types, classes, or functions.
Exemptions from this rule: Symbols from typing may be directly imported

Type Annotation
^^^^^^^^^^^^^^^^^^
Use type annotation with type hints from the module typing for function or method arguments and return types. You may also use it to declare a type of a variable. Use type checkers like pytype during development (for VSCode users we recommend to enable the built-in type checking tool).

Docstrings
^^^^^^^^^^^^^^^^^^^^^
Python uses docstrings to document code. A docstring is a string that is the first statement in a package, module, class or function. 
A docstring is mandatory for every function or method that has one or more of the following properties:
    - being part of the public API
    - nontrivial size
    - non-obvious logic

The docstring of class constructors describe also the principle working of the class and it should the thoroughly documented. As a sample docstring, we provide a template by the documentation of the 

.. autoclass:: rivapy.tools.example_docstring.DocStringExample
   :members:
   :undoc-members:

TODO Comments
^^^^^^^^^^^^^^^^^^^^^^^^
Use TODO comments for code that is temporary, a short-term solution, or good-enough but not perfect.

Naming
^^^^^^^^^^^^^^^^^^^^^^^^^^
module_name, package_name, ClassName, method_name, ExceptionName, function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name, function_parameter_name, local_var_name, query_proper_noun_for_thing, send_acronym_via_https.

Function names, variable names, and filenames should be descriptive; avoid abbreviation. In particular, do not use abbreviations that are ambiguous or unfamiliar to readers outside your project, and do not abbreviate by deleting letters within a word.

Unit Testing
---------------------
Please write as much unittests as possible. The tests are located in the folder *tests* and use the python unittest framework.


Logging
-----------------------
For logging functions that expect a pattern-string (with %-placeholders) as their first argument: Always call them with a string literal (not an f-string!) as their first argument with pattern-parameters as subsequent arguments.

To log, just import the logger from the _logger.py file in the respective module you are currently developing for.
So, if you are currently working in rivapy.pricing, just use the following line in your file

>>> from rivapy.pricing._logger import logger

to retrieve the correct logger.


Logging Usage
==============================
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

