import numpy
import matplotlib.pyplot as plt
from dsfaker.generators.autoincrement import Autoincrement
from dsfaker.generators.utils import ConstantValueGenerator
x = Autoincrement()
y = ConstantValueGenerator(42, dtype=numpy.int16)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(x.get_batch(50), y.get_batch(50), '.')
plt.show()