import numpy as np
from CA_C_N_parsing import coordinates
import parsing_coords
import non_colinear
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import translation_protein
import sklearn_svm as svm

plane_coords = svm.get_plane_coords()
#print(plane_coords.T.shape)

shift = translation_protein.get_centroid_BCEF()
#print(f"Centroid of BCEF:{shift}")

def closest_point_from_origin(plane_equation):
    a,b,c,d = plane_equation
    r = (a**2)+(b**2)+(c**2)
    cx = (-1*d*a)/r
    cy = (-1*d*b)/r
    cz = (-1*d*c)/r
    return np.array([cx,cy,cz])
actual_plane_equation = np.concatenate((svm.model.coef_[0],svm.model.intercept_),axis=0)
shift_to_origin  = closest_point_from_origin(actual_plane_equation)
shifted_plane = translation_protein.shift_structure(plane_coords,shift_to_origin)
# print(shifted_plane)
#print(f"Distance of origin to the plane: {shift_to_origin}")    
shifted_protein_BCEF_cm = translation_protein.shift_structure(translation_protein.shifted_protein_BCEF,shift_to_origin)
shifted_protein_cm = translation_protein.shift_structure(translation_protein.shifted_protein,shift_to_origin)



#shifted_plane = plane_coords


#print(f'Shifted Plane: {shifted_plane}')        
#print(shifted_plane.shape)
#print(f'Shifted Protein: {shifted_protein}')        
#print("==========SHIFTED PLANE============")
# Check for origin  
HasOrigin = np.where(shifted_plane== [0,0,0])
# print(f"HasOrigin:{HasOrigin}")


#def establish_non_colinearity(points):
#    desired_points = []
#    for point in points:
## Incomplete function:: to do 
def plane_equation_from_points(p1, p2, p3):
    # NOTE:  p1 and p2 and p3 must be colinear but we will selcted three points randomnly from 3 place in the plane (implemented:check non_colinearity.py)
    p1, p2, p3 = map(np.array, (p1, p2, p3))
    v1 = p2 - p1 # making vectors
    v2 = p3 - p1
    normal = np.cross(v1, v2) # orthagonal vector to the plane
    A, B, C = normal
    # Compute D using point p1
    D = -np.dot(normal, p1)
    if np.isclose(D, 0):
        D = 0.0
    norm = np.linalg.norm([A, B, C])
    if norm != 0:
        A, B, C = A / norm, B / norm, C / norm
    return A, B, C, D

# Check for orthogonality
plane_coef = svm.model.coef_[0]
shifted_plane_equation = np.append(svm.model.coef_[0],0)


#print(shifted_plane_equation)
def is_cross_product_orthogonal(a, b, tol=1e-8):
    
    a = np.asarray(a)
    b = np.asarray(b)
    cross = np.cross(a, b)
    return abs(np.dot(a, cross)) < tol and abs(np.dot(b, cross)) < tol 

# Example
#print("is Orthangonal?")
#print(is_cross_product_orthogonal(shifted_plane_equation[:3],plane_coef))  # True
#print("============END SHIFTED PLANE=========")
#print("----")
#print(f"SVM hyperplane: {np.concatenate((svm.model.coef_[0],svm.model.intercept_),axis=0)}")
#print("Shifted_Plane")
#print(shifted_plane_equation)
