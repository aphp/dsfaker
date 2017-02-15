.. dsfaker documentation master file, created by
   sphinx-quickstart on Tue Feb  7 13:58:38 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Data Science Faker documentation!
=================================

DSFaker aims at providing data scientists with functions for generating and streaming arrays of data such as:

- Time-series
- Date/Datetime
- Autoincrement
- Random (unique or not) from existing distributions
- Random (unique or not) subsets of data from existing provided dataset
- Random (unique or not) pick-up of data from existing provided dataset

To generate such data, you can specify :ref:`the distribution you want amongst the ones provided by numpy.random <dsfaker-distributions>`.
Or you can also implement your own distribution to generate data.


Parameterization
----------------

When possible, you can specify:

- The return type (`numpy.dtype <https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html>`_)
- The limits (lower bound: lb, and upper bound: ub)

.. code-block:: python

   >>> import numpy
   >>> from dsfaker.distributions import Beta
   >>> from dsfaker.infinite_generators import RandomNumberBounded
   >>> b = Beta(a=2, b=5)
   >>> rnb = RandomNumberBounded(distribution=b, lb=-5, ub=42, dtype=numpy.float16)

Returned types
--------------

We provide four different ways of getting the generated data:

- Get a single element

.. code-block:: python

   >>> rnb.get_single()
   -3.9368686022877473

- Stream one element at a time through python's yield:

.. code-block:: python

   >>>for e in rnb.stream_single():
   >>>  print(e)
   9.27201727507207
   2.413713107525754
   ...

- Get a single batch of elements (in a `numpy.array <https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html>`_):

.. code-block:: python

   >>> rnb.get_batch(batch_size=3)
   array([ 11.390625,   8.765625,   5.671875], dtype=float16)

- Stream batches through python's yield:

.. code-block:: python

   >>> for batch in rnb.stream_batch(batch_size=10000):
   >>>  print(batch)
   [  2.46484375  10.9375       1.203125   ...,   6.78125     19.921875
       7.7265625 ]
   [ 11.484375     8.953125    -1.61132812 ...,   2.33203125  -0.11328125
      -1.44335938]
   ...

When the generator is finite, you can get all the data at a time:

.. code-block:: python

   from dsfaker.finite_generators import TODO
   todo = TODO(distribution=b, dtype=numpy.int16)
   todo.get_all()



+---------------------------------------+---------------------------------------------+------------+----------------------------------------------------------------+
| Returns                               | RAM                                         | Speed      | Advised usage                                                  |
+=======================================+=============================================+============+================================================================+
| a single element at a time            | No RAM limits                               | Slowest    | When you don't care about the speed                            |
+---------------------------------------+---------------------------------------------+------------+----------------------------------------------------------------+
| batches as a numpy.array              | Size of a single batch limited by the RAM   | Quite fast | A good compromise between RAM usage and generation speed       |
+---------------------------------------+---------------------------------------------+------------+----------------------------------------------------------------+
| a numpy.array containing all the data | size of the numpy.array limited by the RAM  | Fastest    | Nice when generating not too large arrays (<10 millions values)|
+---------------------------------------+---------------------------------------------+------------+----------------------------------------------------------------+


User guide
----------

.. toctree::
   :maxdepth: 3

   overview
   dsfaker.distributions
   dsfaker.generators
   dsfaker.noise
