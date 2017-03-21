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