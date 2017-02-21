DataScienceFaker
================

|Build Status| |codecov| |doc| |pypi_v| |pypi_month|

DSFaker aims at providing data scientists with functions for generating
and streaming arrays of data such as:

-  Time-series
-  Date/Datetime
-  Autoincrement & random autoincrement
-  Trigonometric (Sin, Sinh, Cos, Cosh, Tan, Tanh)
-  Random distributions
-  Pattern repetition
-  Noise application

Basic operators between generators and/or python int/float types (+, -,
/, //, \*, …) are also implemented.

To generate such data, you can specify the distribution you want amongst
the ones provided by numpy.random. Or you can also implement your own
distribution to generate data.

Installation
------------

.. code-block:: shell-session

    pip install dsfaker

Documentation
-------------

The documentation is available here: https://dubrzr.github.io/dsfaker/.


.. |Build Status| image:: https://travis-ci.org/Dubrzr/dsfaker.svg?branch=master
   :target: https://travis-ci.org/Dubrzr/dsfaker
.. |codecov| image:: https://codecov.io/gh/Dubrzr/dsfaker/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/Dubrzr/dsfaker
.. |doc| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: https://dubrzr.github.io/dsfaker/
.. |pypi_v| image:: https://img.shields.io/pypi/v/dsfaker.svg
   :target: https://pypi.python.org/pypi/dsfaker
.. |pypi_month| image:: https://img.shields.io/pypi/dm/dsfaker.svg
   :target: https://pypi.python.org/pypi/dsfaker
