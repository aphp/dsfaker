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