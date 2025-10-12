import numpy as np
import transformation
import strand_alignment
import non_colinear
import sklearn_svm as svm
#k = np.array([0,0,1])

def get_equation_for_plane(points):
    non_co_points = non_colinear.find_non_collinear_points_fast(points)
    a,b,c = non_co_points
    return strand_alignment.plane_equation_from_points(a,b,c)

def get_angle(v,k):
    angle= svm.angle_between_planes(v,k)
    print(angle)
    return angle   
    
def get_unit_vector(v,k):
    vector = transformation.test_unit_vector_for_rotation_axis(v,k) 
    print(vector)
    return vector.T
def apply_Rodriguez_rotation(points,k):
    plane_vector = get_equation_for_plane(points)
    v = plane_vector[:3]
    v = np.array(v)
    print(v.shape)
    angle = get_angle(v,k)
    unit_vector = get_unit_vector(v,k)
    unit_vector = np.array(unit_vector)
    print(unit_vector.shape)
    matrix = transformation.Rodriguez_rotation_matrix(unit_vector[0][0],unit_vector[0][1],angle)

    print(matrix)
    product = np.dot(matrix,v.T)
    return product.T




if __name__ =="__main__":
   # Choose a grid of x and y values
    x_vals = np.linspace(-5, 5, 5)
    y_vals = np.linspace(-5, 5, 5)
    X, Y = np.meshgrid(x_vals, y_vals)

# Solve for Z using the plane equation
    Z = (1.223253242 * X + 2.353435 * Y) / 2.5433532

# Stack into (N, 3) shape
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1) 
    print(points.shape)
    k = np.array([0,0,1])
    k_col = k.reshape(-1,1)
    print(k_col.shape)
    apply_Rodriguez_rotation(points,k_col)
