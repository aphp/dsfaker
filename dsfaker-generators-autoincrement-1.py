import matplotlib.pyplot as plt
from dsfaker.generators.autoincrement import Autoincrement
x = Autoincrement()
y = Autoincrement(start=-10, step=-0.2, dtype=np.float32)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(x.get_batch(20), y.get_batch(20), 'o')
ax.plot(x.get_batch(20), y.get_batch(20), 'o')
plt.show()