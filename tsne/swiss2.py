import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn
import sklearn.datasets

# Generate 3D swiss roll data
data, _ = sklearn.datasets.make_swiss_roll(
    n_samples=10000, random_state=0
)

# Rescale data to [-1, 1]
data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
data = data * 2 - 1

# Calculate the number of samples in each part
n_samples = data.shape[0]
part_size = n_samples // 4

# Color parts of the swiss roll with 4 different colors
colors = np.concatenate(
    [
        np.array([[1, 0, 0, 1]] * part_size),
        np.array([[0, 1, 0, 1]] * part_size),
        np.array([[0, 0, 1, 1]] * part_size),
        np.array([[0, 0, 0, 1]] * (n_samples - 3 * part_size)),
    ]
)

# Save the swiss roll data and colors to separate files
np.savetxt("swiss_roll_data.txt", data)
np.savetxt("swiss_roll_colors.txt", colors)

# Plot the swiss roll
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors)
plt.show()
