import sklearn.datasets as skd
import numpy as np
import matplotlib.pyplot as plt

roll, sr_color = skd.make_swiss_roll(n_samples=10000, noise=0, random_state=None)

np.savetxt("swiss_roll.txt", roll, delimiter=" ")
np.savetxt("swiss_roll_color.txt", sr_color, delimiter=" ")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(roll[:, 0], roll[:, 1], roll[:, 2], c=sr_color, s=1)
plt.show()
