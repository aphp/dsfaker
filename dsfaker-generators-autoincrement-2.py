import matplotlib.pyplot as plt
from dsfaker.generators.distributions import Beta
from dsfaker.generators.autoincrement import Autoincrement, AutoincrementWithGenerator
x = AutoincrementWithGenerator(start=42, generator=Beta(a=2, b=2))
y = Autoincrement()
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(x.get_batch(20), y.get_batch(20), 'o')
ax.plot(x.get_batch(20), y.get_batch(20), 'o')
plt.show()