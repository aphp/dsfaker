.. dsfaker documentation master file, created by
   sphinx-quickstart on Tue Feb  7 13:58:38 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Data Science Faker documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`


Functionalities
===============

This package aims at providing data scientists with functions for generating and streaming arrays of data such as:

- Time-series
- Date/Datetime
- Autoincrement
- Random (unique or not) subsets of data from existing provided dataset
- Random (unique or not) pick-up of data from existing provided dataset

Parameterization
----------------

When possible, you can specify:

- The `numpy.dtype <https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html>`_
- The limits (lower bound and upper bound)

Returned types
--------------

We provide multiple ways of getting the generated data:

- `numpy.array <https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`_
- Stream one element at a time throught python's yield
- Stream batches (numpy.array) of certain size throught python's yield

+---------------------------------------+---------------------------------------------+------------+----------------------------------------------------------------+
| Returns                               | RAM                                         | Speed      | Advised usage                                                  |
+=======================================+=============================================+============+================================================================+
| a numpy.array containing all the data | size of the numpy.array limited by the RAM  | Fastest    | Nice when generating not too large arrays (<1 millions values) |
+---------------------------------------+---------------------------------------------+------------+----------------------------------------------------------------+
| a single element at a time            | No RAM limits                               | Slowest    | When you don't care about the generating speed                 |
+---------------------------------------+---------------------------------------------+------------+----------------------------------------------------------------+
| batches as a numpy.array              | Size of a single batch limited by the RAM   | Quite fast | A good compromise between RAM usage and generation speed       |
+---------------------------------------+---------------------------------------------+------------+----------------------------------------------------------------+


Installation
============

From pypi
---------

Install the package:
  pip install dsfaker

From sources
------------

Install requirements (possibly in a venv):
  pip install -r requirements.txt

Install the package:
  pip install .

To build the documentation:
  pip install -r requirements.doc.txt
  cd docs && make html

To launch tests:
  pip install -r requirements.test.txt

Usage
=====

.. toctree::
   dsfaker