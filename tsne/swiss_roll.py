import sklearn.datasets as skd
import numpy as np
import matplotlib.pyplot as plt

roll = skd.make_swiss_roll(n_samples=10000, noise=0.0, random_state=None)

np.savetxt("swiss_roll_2.txt", roll[0], delimiter=" ")

# data = np.loadtxt("swiss_roll.txt", delimiter=" ", skiprows=1)

# ax = plt.axes(projection="3d")
# ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=data[:, 2], cmap="Greens")
# plt.show()
