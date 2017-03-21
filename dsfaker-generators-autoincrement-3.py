import matplotlib.pyplot as plt
from dsfaker.generators.distributions import Normal
from dsfaker.generators.autoincrement import Autoincrement, AutoincrementWithGenerator
x = AutoincrementWithGenerator(start=0, generator=Normal())
y = Autoincrement()
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(x.get_batch(20), y.get_batch(20), 'o')
ax.plot(x.get_batch(20), y.get_batch(20), 'o')
plt.show()