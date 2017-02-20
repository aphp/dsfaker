Trigonometric
=============

This module provides classes that implements trigonometric functions.

Sinus/Cosinus
-------------

These are the default Sinus and Cosinus functions with a constant step.

.. code-block:: python3

    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.trigonometric import Sin, Cos
    import numpy
    nb_vals = 500
    x = Autoincrement(step=0.05, dtype=numpy.float32)
    y1 = Sin(x.copy())
    y2 = Cos(x.copy())
    x_vals = x.get_batch(nb_vals)
    plt.plot_date(x_vals, y1.get_batch(nb_vals), '.', label='Sin')
    plt.plot_date(x_vals, y2.get_batch(nb_vals), '.', label='Cos')
    plt.legend()
    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.trigonometric import Sin, Cos
    import numpy

    nb_vals = 500
    x = Autoincrement(step=0.05, dtype=numpy.float32)
    y1 = Sin(x.copy())
    y2 = Cos(x.copy())
    x_vals = x.get_batch(nb_vals)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x_vals, y1.get_batch(nb_vals), '.', label='Sin')
    ax.plot(x_vals, y2.get_batch(nb_vals), '.', label='Cos')
    plt.legend()
    plt.show()


Tangent
-------

This is the default Tangent function with a constant step.

.. code-block:: python3

    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.trigonometric import Tan
    import numpy
    nb_vals = 1000
    x = Autoincrement(step=0.1, dtype=numpy.float32)
    y = Tan(x.copy())
    x_vals = x.get_batch(nb_vals)
    plt.plot_date(x_vals, y.get_batch(nb_vals))
    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.trigonometric import Tan
    import numpy

    nb_vals = 1000
    x = Autoincrement(step=0.1, dtype=numpy.float32)
    y = Tan(x.copy())
    x_vals = x.get_batch(nb_vals)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x_vals, y.get_batch(nb_vals))
    plt.show()


Hyperbolic Sinus/Cosinus/Tangent
--------------------------------

These are hyperbolic functions with a constant step.

.. code-block:: python3

    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.trigonometric import Sinh, Cosh, Tanh
    import numpy

    nb_vals = 100
    x = Autoincrement(step=0.01, dtype=numpy.float32)
    y1 = Sinh(x.copy())
    y2 = Cosh(x.copy())
    y3 = Tanh(x.copy())
    x_vals = x.get_batch(nb_vals)
    plt.plot(x_vals, y1.get_batch(nb_vals), '.', label='Sinh')
    plt.plot(x_vals, y2.get_batch(nb_vals), '.', label='Cosh')
    plt.plot(x_vals, y3.get_batch(nb_vals), '.', label='Tanh')
    plt.legend()
    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.trigonometric import Sinh, Cosh, Tanh
    import numpy

    nb_vals = 100
    x = Autoincrement(step=0.01, dtype=numpy.float32)
    y1 = Sinh(x.copy())
    y2 = Cosh(x.copy())
    y3 = Tanh(x.copy())
    x_vals = x.get_batch(nb_vals)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x_vals, y1.get_batch(nb_vals), '.', label='Sinh')
    ax.plot(x_vals, y2.get_batch(nb_vals), '.', label='Cosh')
    ax.plot(x_vals, y3.get_batch(nb_vals), '.', label='Tanh')
    plt.legend()
    plt.show()

