Utils
=====

AbsoluteOperator
----------------

This Generator takes a Generator and transforms negatives values to positive.

.. code-block:: python3

    import numpy
    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.trigonometric import Sin
    from dsfaker.generators.utils import AbsoluteOperator

    x = Autoincrement(start=0, step=0.1, dtype=numpy.float32)
    y = Sin(x.copy())
    by = AbsoluteOperator(Sin(x.copy()))
    x_vals = x.get_batch(300)
    plt.plot(x_vals, y.get_batch(300), 'o', label='Sin')
    plt.plot(x_vals, by.get_batch(300), '.', label='Abs(Sin)')
    plt.legend()
    plt.show()

.. plot::

    import numpy
    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.trigonometric import Sin
    from dsfaker.generators.utils import AbsoluteOperator

    x = Autoincrement(start=0, step=0.1, dtype=numpy.float32)
    y = Sin(x.copy())
    by = AbsoluteOperator(Sin(x.copy()))
    x_vals = x.get_batch(300)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x_vals, y.get_batch(300), 'o', label='Sin')
    ax.plot(x_vals, by.get_batch(300), '.', label='Abs(Sin)')
    plt.legend()
    plt.show()


ApplyFunctionOperator
---------------------

This Generator apply a function to another Generator.

.. code-block:: python3

    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.utils import ApplyFunctionOperator

    x = Autoincrement(start=-50)
    y = ApplyFunctionOperator(lambda x: 3*x**2-2*x-10, x.copy())
    plt.plot(x.get_batch(100), y.get_batch(100), '.')
    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.utils import ApplyFunctionOperator

    x = Autoincrement(start=-50)
    y = ApplyFunctionOperator(lambda x: 3*x**2-2*x-10, x.copy())
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x.get_batch(100), y.get_batch(100), '.')
    plt.show()


BoundingOperator
----------------

This Generator bounds another Generator between a lower bound (lb) and an upper bound (ub).

.. code-block:: python3

    import numpy
    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.trigonometric import Sin
    from dsfaker.generators.utils import BoundingOperator

    x = Autoincrement(start=0, step=0.1, dtype=numpy.float32)
    y = Sin(x.copy()) * 10
    by = BoundingOperator(y.copy(), lb=-5, ub = 7.5)
    x_vals = x.get_batch(500)
    plt.plot(x_vals, y.get_batch(500), '.', label="Sin")
    plt.plot(x_vals, by.get_batch(500), '.', label="Bounded Sin")
    plt.legend()
    plt.show()

.. plot::

    import numpy
    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.trigonometric import Sin
    from dsfaker.generators.utils import BoundingOperator

    x = Autoincrement(start=0, step=0.1, dtype=numpy.float32)
    y = Sin(x.copy()) * 10
    by = BoundingOperator(y.copy(), lb=-5, ub = 7.5)
    fig, ax = plt.subplots(figsize=(10,5))
    x_vals = x.get_batch(500)
    ax.plot(x_vals, y.get_batch(500), '.', label="Sin")
    ax.plot(x_vals, by.get_batch(500), '.', label="Bounded Sin")
    plt.legend()
    plt.show()


CastOperator
------------

This Generator takes another Generator and cast its values to a given dtype.

.. code-block:: python3

    import numpy
    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.distributions import Normal
    from dsfaker.generators.utils import CastOperator
    x = Autoincrement()
    y = Normal(seed=22)
    by = CastOperator(y.copy(), dtype=numpy.int16)
    x_vals = x.get_batch(50)
    plt.plot(x_vals, y.get_batch(50), '.', label='Normal law')
    plt.plot(x_vals, by.get_batch(50), '.', label='Cast to int16')
    plt.legend()
    plt.show()

.. plot::

    import numpy
    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.distributions import Normal
    from dsfaker.generators.utils import CastOperator
    x = Autoincrement()
    y = Normal(seed=22)
    by = CastOperator(y.copy(), dtype=numpy.int16)
    fig, ax = plt.subplots(figsize=(10,5))
    x_vals = x.get_batch(50)
    ax.plot(x_vals, y.get_batch(50), '.', label='Normal law')
    ax.plot(x_vals, by.get_batch(50), '.', label='Cast to int16')
    plt.legend()
    plt.show()


ConstantValueGenerator
----------------------

This Generator simply returns a constant given value indefinitely.

.. code-block:: python3

    import numpy
    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.utils import ConstantValueGenerator
    x = Autoincrement()
    y = ConstantValueGenerator(42, dtype=numpy.int16)
    plt.plot_date(x.get_batch(50), y.get_batch(50), '.')
    plt.show()

.. plot::

    import numpy
    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.utils import ConstantValueGenerator
    x = Autoincrement()
    y = ConstantValueGenerator(42, dtype=numpy.int16)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x.get_batch(50), y.get_batch(50), '.')
    plt.show()


MeanHistory
-----------

This Generator provide a simple solution to save the last `x` values from a generator and get the

.. code-block:: python3

    import numpy
    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.distributions import Normal
    from dsfaker.generators.trigonometric import Sin
    from dsfaker.generators.utils import MeanHistory


    xx = Autoincrement().get_batch(100)
    y = Normal(seed=42)
    z = MeanHistory(Normal(seed=42), 10)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(xx, y.get_batch(100), '-')
    ax.plot(xx, z.get_batch(100), '-')
    plt.show()


.. plot::

    import numpy
    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.distributions import Normal
    from dsfaker.generators.trigonometric import Sin
    from dsfaker.generators.utils import MeanHistory


    xx = Autoincrement().get_batch(100)
    y = Normal(seed=42)
    z = MeanHistory(Normal(seed=42), 10)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(xx, y.get_batch(100), '-')
    ax.plot(xx, z.get_batch(100), '-')
    plt.show()


ScalingOperator
---------------

This Generator scales a BoundedGenerator to another range of values.

.. code-block:: python3

    import numpy
    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.trigonometric import Sin
    from dsfaker.generators.utils import ScalingOperator

    x = Autoincrement(start=0, step=0.1, dtype=numpy.float32)
    y = Sin(x.copy())
    by = ScalingOperator(y.copy(), lb=0, ub = 10, dtype=numpy.float32)
    x_vals = x.get_batch(500)
    plt.plot(x_vals, y.get_batch(500), '.', label="Sin")
    plt.plot(x_vals, by.get_batch(500), '.', label="Bounded Sin")
    plt.legend()
    plt.show()

.. plot::

    import numpy
    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.trigonometric import Sin
    from dsfaker.generators.utils import ScalingOperator

    x = Autoincrement(start=0, step=0.1, dtype=numpy.float32)
    y = Sin(x.copy())
    by = ScalingOperator(y.copy(), lb=0, ub = 10, dtype=numpy.float32)
    fig, ax = plt.subplots(figsize=(10,5))
    x_vals = x.get_batch(500)
    ax.plot(x_vals, y.get_batch(500), '.', label="Sin")
    ax.plot(x_vals, by.get_batch(500), '.', label="Scaled Sin")
    plt.legend()
    plt.show()
