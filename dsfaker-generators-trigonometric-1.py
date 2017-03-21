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