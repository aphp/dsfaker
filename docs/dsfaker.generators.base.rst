.. _generators-base:

Base
====

Generator()
-----------

This is the mother class of all others! Every new Generator must implement its four methods.


BoundedGenerator(Generator)
---------------------------

This Generator returns bounded values.

This Generator is an abstract class that provides three attributes:

- bounded = True (Boolean)
- lb (lower bound)
- ub (upper bound)


FiniteGenerator(Generator)
--------------------------

This Generator is finite. It will stop after a certain number of values returned.

This Generator is an abstract class that provides one more attribute:

- finite = True (Boolean)


InfiniteGenerator(Generator)
----------------------------

This Generator is infinite. It will never stop returning values if needed.

This Generator is an abstract class that provides one more attribute:

- finite = False (Boolean)


ReduceOperator(Generator)
-------------------------

This Generator implements a reduce operation on two or more provided Generators.

.. code-block:: python

   >>> ReduceOperator(Sin(), Beta(2, 2), Tan(), lambda a, b: a + b)


There are many available operators that inherits from ReduceOperator:

- AddOperator
- SubOperator
- TrueDivOperator
- FloorDivOperator
- MulOperator
- PowOperator
- ModOperator
- AndOperator
- OrOperator
- XorOperator


Distribution(Generator)
-----------------------

This Generator implements a pseudo-random function.

This Generator is an abstract class that provides four more attributes:

- bounded (Boolean)
- continuous (Boolean)
- lb (lower bound)
- ub (upper bound)


DistributionUnbounded(Distribution)
-----------------------------------

This Generator implements a pseudo-random function that is unbounded.

- bounded = False


DistributionNonNegative(Distribution)
-------------------------------------

This Generator implements a pseudo-random function that is non-negative.

- bounded = True
- lb = 0


DistributionBounded(Distribution)
---------------------------------

This Generator implements a pseudo-random function that is non-negative.

- bounded = True
- lb = provided by the pseudo-random function
- lb = provided by the pseudo-random function

