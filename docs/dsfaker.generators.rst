Generators
==========

In DSFaker, everything is a Generator.

What is a Generator?
--------------------

A Generator is a python class implementing four different functions:

1 - get_single()
................

Which returns a single value

.. code-block:: python

   >>> generator.get_single()
   value


2 - stream_single()
...................

Which returns an iterable returning one value at a time

.. code-block:: python

   >>> for v in generator.stream_single():
   >>>   print(v)
   value1
   value2
   ...


3 - get_batch(batch_size)
.........................

Which returns *batch_size* values

.. code-block:: python

   >>> generator.get_batch(3)
   array([ value1,   value2,   value3])


4 - stream_batch(batch_size)
............................

Which returns an iterable returning *batch_size* values at a time

.. code-block:: python

   >>> for v in generator.stream_batch(3):
   >>>   print(v)
   array([ value1,   value2,   value3])
   array([ value4,   value5,   value6])
   ...


Generators operations
---------------------

A Generator can consist of multiple other Generators. The base Generator class implements many basic operations that combines Generators into new ones.

Let's take four different Generators:

.. code-block:: python

   >>> g1 = Sin()
   >>> g2 = Autoincrement()
   >>> g3 = RepeatPattern([0,1,2,3,4,5,6])
   >>> g4 = Normal()

Then, you can combine those Generators using basic operations, such as:

.. code-block:: python

   >>> g5 = g1 + g2
   >>> g6 = g3 / g4
   >>> g7 = g6 * g5

The resulting objects (g5, g6, g7) are new Generators that will transparently apply the operator to the two Generators

And many more are available! See: :ref:`generators.base <generators-base>`.

.. DANGER::
   Do not use a Generator that is already used by another one because we only store references to those Generators.
   If you want to duplicate a Generator, use `generator.copy()`.

You can also work with python int and float types:

.. code-block:: python

   >>> g8 = 22.9 * g7 // 3 + 4.2


Available generators
--------------------

There are many different Generators available, but every Generator implements an abstract class from :ref:`generators.base <generators-base>`

.. toctree::

   dsfaker.generators.base
   dsfaker.generators.autoincrement
   dsfaker.generators.date
   dsfaker.generators.distributions
   dsfaker.generators.series
   dsfaker.generators.str
   dsfaker.generators.timeseries
   dsfaker.generators.trigonometric
   dsfaker.generators.utils

