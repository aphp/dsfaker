import matplotlib.pyplot as plt
from dsfaker.generators.autoincrement import Autoincrement
from dsfaker.generators.utils import ApplyFunctionOperator

x = Autoincrement(start=-50)
y = ApplyFunctionOperator(lambda x: 3*x**2-2*x-10, x.copy())
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(x.get_batch(100), y.get_batch(100), '.')
plt.show()