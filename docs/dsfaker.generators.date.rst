Date
====

This module provides classes that implements date-related logic.

RandomDatetime
--------------

This Generator produces random dates between two dates following a given distribution.

.. code-block:: python3

    >>> import matplotlib.pyplot as plt
    >>> from numpy import datetime64
    >>> from dsfaker.generators.autoincrement import Autoincrement
    >>> from dsfaker.generators.date import RandomDatetime
    >>> from dsfaker.generators.distributions import Vonmises
    >>> x = RandomDatetime(Vonmises(mu=0, kappa=1.8), start=datetime64("1046"), end=datetime64("1789"), unit="Y")
    >>> y = Autoincrement()
    >>> plt.plot_date(x.get_batch(2000).astype('O'), y.get_batch(2000), 'o')
    >>> plt.show()

.. plot::

    import matplotlib.pyplot as plt
    from numpy import datetime64
    from dsfaker.generators.autoincrement import Autoincrement
    from dsfaker.generators.date import RandomDatetime
    from dsfaker.generators.distributions import Vonmises
    x = RandomDatetime(Vonmises(mu=0, kappa=1.8), start=datetime64("1046"), end=datetime64("1789"), unit="Y")
    y = Autoincrement()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot_date(x.get_batch(2000).astype('O'), y.get_batch(2000), 'o')
    plt.show()
