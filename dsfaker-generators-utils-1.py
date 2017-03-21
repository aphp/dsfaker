import numpy
import matplotlib.pyplot as plt
from dsfaker.generators.autoincrement import Autoincrement
from dsfaker.generators.trigonometric import Sin
from dsfaker.generators.utils import AbsoluteOperator

x = Autoincrement(start=0, step=0.1, dtype=numpy.float32)
y = Sin(x.copy())
by = AbsoluteOperator(Sin(x.copy()))
x_vals = x.get_batch(300)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(x_vals, y.get_batch(300), 'o', label='Sin')
ax.plot(x_vals, by.get_batch(300), '.', label='Abs(Sin)')
plt.legend()
plt.show()