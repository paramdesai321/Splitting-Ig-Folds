import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x = np.random.rand(10)
y = np.random.rand(10)
z_pos = np.abs(x+y)
z_neg = -z_pos

struct1 = np.vstack((x,y,z_pos))
struct2= np.vstack((x,y,z_neg))
print(struct1.shape)
model = LinearRegression()
model.fit(x.reshape(-1,1),y.reshape(-1,1))
m = model.coef_
b = model.intercept_
line_eqn = np.append(m,b)
print(f"Line Eqn: {line_eqn}")

def R_z(theta):
   
    c, s = np.cos(theta), np.sin(theta)
    # both np functions take the angle is rads 
    R = np.array([
        [ c,-s ,0],
        [ s, c,  0], 
        [0, 0,  1]  
    ])  
    return R

# A simpler way:
# The vector is [1, m]. Angle with x-axis is atan2(m,1).
# We want to rotate it to y-axis (angle pi/2).
# Rotation angle is pi/2 - atan2(m,1).
m_slope = line_eqn[0]
angle_of_line_rad = np.arctan2(m_slope, 1)
rotation_angle = np.pi/2 - angle_of_line_rad

print(f"Rotation angle (deg): {np.degrees(rotation_angle)}")

# The R_z matrix is for CCW rotation on column vectors.
# For row vectors, we need R_z.T for CCW rotation.
rot_matrix = R_z(rotation_angle)

result1 = np.dot(struct1.T, rot_matrix.T)
print(result1.shape)
result2= np.dot(struct2.T, rot_matrix.T)
print(result2.shape)
line_points = np.vstack((x,line_eqn[0]*x+line_eqn[1],np.zeros(x.size)))

print(f"Line Points: {line_points.shape}")

line_transform = np.dot(line_points.T, rot_matrix.T) 

print(result1.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z_pos, color='blue', marker='o', label='Original Points')
ax.scatter(x, y, z_neg, color='blue', marker='o')
ax.scatter(result1[:,0],result1[:,1],result1[:,2],color='y',marker='x', label='Transformed Points')
ax.scatter(result2[:,0],result2[:,1],result2[:,2],color='y',marker='x')
ax.plot(x, line_eqn[0]*x+line_eqn[1],0, color='red', linewidth=2, label='Original Line')
ax.plot(line_transform[:,0], line_transform[:,1],line_transform[:,2], color='black', linewidth=2, label='Transformed Line')

# Plot y-axis
all_y = np.concatenate([y, result1[:,1], result2[:,1]])
ymin, ymax = np.min(all_y), np.max(all_y)
ax.plot([0,0], [ymin, ymax], 0, color='green', linewidth=2, label='Y-axis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
