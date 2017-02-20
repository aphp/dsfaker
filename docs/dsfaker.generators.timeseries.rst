TimeSeries
==========

This modules provides classes for time-series logic.

TimeSeries
----------

This Generator combines two Generators into one.

.. code-block:: python3

    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement, AutoincrementWithGenerator
    from dsfaker.generators.distributions import Beta, Normal
    from dsfaker.generators.series import RepeatPattern
    from dsfaker.generators.timeseries import TimeSeries
    from dsfaker.generators.utils import BoundingOperator


    div1 = Autoincrement()
    div2 = AutoincrementWithGenerator(1, Normal(std=0.1))
    time = AutoincrementWithGenerator(0, BoundingOperator(div1 / div2, lb=0.5, ub=1.1))

    heartbeat = [0,0,0,0,0,0,0,0,0,0,0.5,1,1.5,2,1,0,-3,0,3,8,15, 20, 29,-19,-14,-10,-7,-5,0,5,10,5,2,0,0,0,0,0,0,0,0,0,0,0,0]

    rp = RepeatPattern(heartbeat) # Seasonality
    rp += AutoincrementWithGenerator(0, Beta(0.25,2)) # Trend
    rp += Normal() # Noise

    ts = TimeSeries(time, rp)
    x,y = ts.get_batch(500)
    plt.plot(x, y)
    plt.show()

.. plot::

    import matplotlib.pyplot as plt
    from dsfaker.generators.autoincrement import Autoincrement, AutoincrementWithGenerator
    from dsfaker.generators.distributions import Beta, Normal
    from dsfaker.generators.series import RepeatPattern
    from dsfaker.generators.timeseries import TimeSeries
    from dsfaker.generators.utils import BoundingOperator


    div1 = Autoincrement()
    div2 = AutoincrementWithGenerator(1, Normal(std=0.1))
    time = AutoincrementWithGenerator(0, BoundingOperator(div1 / div2, lb=0.5, ub=1.1))

    heartbeat = [0,0,0,0,0,0,0,0,0,0,0.5,1,1.5,2,1,0,-3,0,3,8,15, 20, 29,-19,-14,-10,-7,-5,0,5,10,5,2,0,0,0,0,0,0,0,0,0,0,0,0]

    rp = RepeatPattern(heartbeat) # Seasonality
    rp += AutoincrementWithGenerator(0, Beta(0.25,2)) # Trend
    rp += Normal() # Noise

    ts = TimeSeries(time, rp)
    x,y = ts.get_batch(500)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x, y)
    plt.show()
