import numpy as np
from scipy.spatial.transform import Rotation as R

# Step 1: Define normal vector of the plane
n = np.array([1, 1, 1])
n_unit = n / np.linalg.norm(n)

# Step 2: Define target vector (Z-axis)
target = np.array([0, 0, 1])

# Step 3: Find rotation matrix to align n → z-axis
rotation, _ = R.align_vectors([target], [n_unit])
R_matrix = rotation.as_matrix()

# Step 4: Apply to a point
x = np.array([1, 1, 2])
rotated_x = R_matrix @ x  # or np.dot(R_matrix, x)

print("Rotated point:", rotated_x)

