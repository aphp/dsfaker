String
======

Regex
-----

This Generator is returning random strings matching a provided regex.


.. code-block:: python3

    from dsfaker.generators.str import Regex

    pattern = r'(0|\+33|0033)[1-9][0-9]{8}'

    gen = Regex(pattern)
    gen.get_single()
    -> '0834673019'
    gen.get_single()
    -> '0033470369802'
    gen.get_single()
    -> '+33329135952'
    gen.get_single()
    -> '0289786407'

