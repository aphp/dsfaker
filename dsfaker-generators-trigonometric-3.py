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