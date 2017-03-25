Listeners
=========

Listeners are objects than listen at Generators.
When a generator is used (gen.get_single() or gen.get_batch(x)), it also gives the generated value(s) to the listeners registered with it.

This allows you to get values of intermediate Generators.


Available listeners
-------------------

There are many different Listeners available, but every Listener implements an abstract class from :ref:`listeners.base <listeners-base>`

.. toctree::

   dsfaker.listeners.base
   dsfaker.listeners.buffers

