import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create the meshgrid
x, y = np.meshgrid(np.arange(-5, 5, 0.5), np.arange(-5, 5, 0.5))
print(f"x: {x}")
print(f"y: {y}")
# Define the plane equation
z = 2*x + 3*y - 5
print(f"z: {z}")
# Create the figure and 3D subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x, y, z)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()

