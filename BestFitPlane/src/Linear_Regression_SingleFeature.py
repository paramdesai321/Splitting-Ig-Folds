import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from parsingcoord import x_coordinates,z_coordinates,y_coordinates

# Retrieve the coordinates
x = x_coordinates()  # Get x-coordinates as a list/array
print("x:", x)  # Debug: Print x to ensure it's a list/array

z = z_coordinates()  # Get y-coordinates as a list/array
print("y:", z)  # Debug: Print y to ensure it's a list/arra
y = y_coordinates()  # Get z-coordinates as a list/array
print(type(y))  # Debug: Print z to ensure it's a list/array

# Convert lists to numpy arrays if they aren't already
x = np.array(x)
y = np.array(y)
z = np.array(z)
print(z)
# Plot the initial data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, label='Data')  # Plot the original data points

# Initialize weights and bias
w1 = np.random.rand(1)
w2 = np.random.rand(1)
b = random.uniform(-100, 100)

# Linear regression function
def f(x, y, w1, w2, b):
    return x * w1 + y * w2 + b

# Learning rate
k = 0.1

# Gradient Descent
for i in range(400):
    z_pred = f(x, y, w1, w2, b)
    print("Predicted z:", z_pred)  # Debug: Print predicted z values
    print("Actual z:", z)          # Debug: Print actual z values
    
    loss = ((z_pred - z) ** 2).mean()
    print(f"Iteration {i+1}, Loss: {loss}")  # Debug: Print loss for each iteration

    w1_grad = 2 * ((z_pred - z) * x).mean()
    w2_grad = 2 * ((z_pred - z) * y).mean()
    b_grad = 2 * (z_pred - z).mean()

    w1 -= k * w1_grad
    w2 -= k * w2_grad
    b -= k * b_grad

# Final prediction
z_pred = f(x, y, w1, w2, b)

# Plot the final results
ax.plot_trisurf(x, y, z_pred, color='red', alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()

# Final loss
loss = ((z_pred - z) ** 2).mean()
print('Final loss:', loss)
print('Predicted values:', z_pred)
print("Target values:", z)
  