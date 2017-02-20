Autoincrement
=============

This module provides classes that implements an autoincrement logic.

Autoincrement
-------------

This is a simple autoincrement that has a constant step.

.. code-block:: python3

    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement

    x = Autoincrement()
    y = Autoincrement(start=-10, step=-0.2, dtype=np.float32)
    plt.plot(x.get_batch(20), y.get_batch(20), 'o')
    plt.plot(x.get_batch(20), y.get_batch(20), 'o')
    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    x = Autoincrement()
    y = Autoincrement(start=-10, step=-0.2, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x.get_batch(20), y.get_batch(20), 'o')
    ax.plot(x.get_batch(20), y.get_batch(20), 'o')
    plt.show()

AutoincrementWithGenerator
--------------------------

This is a autoincrement that has a dynamic step based on a generator.

.. code-block:: python3

    import matplotlib.pyplot as plt
    from dsfaker.generators.distributions import Beta
    from dsfaker.generators.autoincrement import Autoincrement, AutoincrementWithGenerator

    x = AutoincrementWithGenerator(start=42, generator=Beta(a=2, b=2))
    y = Autoincrement()
    plt.plot(x.get_batch(20), y.get_batch(20), 'o')
    plt.plot(x.get_batch(20), y.get_batch(20), 'o')
    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    from dsfaker.generators.distributions import Beta
    from dsfaker.generators.autoincrement import Autoincrement, AutoincrementWithGenerator
    x = AutoincrementWithGenerator(start=42, generator=Beta(a=2, b=2))
    y = Autoincrement()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x.get_batch(20), y.get_batch(20), 'o')
    ax.plot(x.get_batch(20), y.get_batch(20), 'o')
    plt.show()

Take care to use a non-negative generator for the step is you want values that always increase/decrease.

For exemple, autoincrement with an unbounded Normal law would give:

.. code-block:: python3

    import matplotlib.pyplot as plt
    from dsfaker.generators.distributions import Normal
    from dsfaker.generators.autoincrement import Autoincrement, AutoincrementWithGenerator

    x = AutoincrementWithGenerator(start=0, generator=Normal())
    y = Autoincrement()
    plt.plot(x.get_batch(20), y.get_batch(20), 'o')
    plt.plot(x.get_batch(20), y.get_batch(20), 'o')
    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    from dsfaker.generators.distributions import Normal
    from dsfaker.generators.autoincrement import Autoincrement, AutoincrementWithGenerator
    x = AutoincrementWithGenerator(start=0, generator=Normal())
    y = Autoincrement()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x.get_batch(20), y.get_batch(20), 'o')
    ax.plot(x.get_batch(20), y.get_batch(20), 'o')
    plt.show()


