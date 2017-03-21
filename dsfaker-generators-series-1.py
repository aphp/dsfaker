import matplotlib.pyplot as plt
from dsfaker.generators.autoincrement import Autoincrement
from dsfaker.generators.series import RepeatPattern
from dsfaker.generators.trigonometric import Sinh, Cosh, Tanh
import numpy

heartbeat = [0,0,0,0,0,0,0,0,0,0,0.5,1,1.5,2,1,0,-3,0,3,8,15, 20, 29,-19,-14,-10,-7,-5,0,5,10,5,2,0,0,0,0,0,0,0,0,0,0,0,0]

nb_vals = 300
time = Autoincrement().get_batch(nb_vals)
rp = RepeatPattern(heartbeat).get_batch(nb_vals)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(time, rp)
plt.show()