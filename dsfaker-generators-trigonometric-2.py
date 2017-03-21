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