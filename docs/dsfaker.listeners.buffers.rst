Buffers
=======


History
-------

This Generator provide a simple solution to save the last `x` values from a generator and retrieve them if needed.

.. code-block:: python3

    import numpy
    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.distributions import Normal
    from dsfaker.listeners.buffers import CircularBuffer

    xx = Autoincrement().get_batch(50)
    y = Normal()
    cb = CircularBuffer(10)
    y.add_listener(cb)
    yy = []
    zz = []
    for i in range(50):
        yy.append(y.get_single())
        zz.append(cb.get_prev(-2))

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(xx, yy, '-')
    ax.plot(xx, zz, '-')
    plt.show()

.. plot::

    import numpy
    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.distributions import Normal
    from dsfaker.listeners.buffers import CircularBuffer

    xx = Autoincrement().get_batch(50)
    y = Normal()
    cb = CircularBuffer(10)
    y.add_listener(cb)
    yy = []
    zz = []
    for i in range(50):
        yy.append(y.get_single())
        zz.append(cb.get_prev(-2))

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(xx, yy, '-')
    ax.plot(xx, zz, '-')
    plt.show()


