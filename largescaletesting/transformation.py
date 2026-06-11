import numpy as np

import parsing_coords
#import sklearn_svm as svm
import translation_protein
import translation_plane
import non_colinear
# _____________________________________________________
#def z_shift(structure, c, d):
#    shifted_atoms = []
#    new_struct  = np.array(structure)
#    new_struct = new_struct.T
#    for atom in structure:
#        x = atom[0]
#        y =atom[1]
#        z =atom[2]
#        shifted_z = z - (d / c)
#        shifted_atoms.append(shifted_z)
#    shifted_atoms = np.array(shifted_atoms)
#    new_struct[2]  = shifted_atoms.T
#    return new_struct.T
#print("Investigating the params of the acutal plane---")
#
#print(f"coefs: {svm.model.coef_}") 
#print(f"intercept:{svm.model.intercept_}")
##plane_coords = svm.get_plane_coords()
#print(f'Plane coords: {plane_coords}')
#shifted_plane = z_shift(plane_coords,c,d)
def angle_between_planes(n1, n2):
   # Avoid divide-by-zero
   # n1 and n2 vectors  are the parmaeters of the planes 
    n1 = np.array(n1)
    n2 = np.array(n2)

    # Dot product and norms
    dot_product = (np.dot(n1, n2))  # use abs to get angle between 0-90
    norm_product = np.linalg.norm(n1) * np.linalg.norm(n2)

    if norm_product == 0:
        raise ValueError("One of the normal vectors is zero-length")

    # Compute angle in radians, then convert to degrees
    cos_theta = dot_product / norm_product
    #angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0)) 
    angle_rad = np.arccos(cos_theta)
    return angle_rad


# _____________________________________________________
shifted_plane_coords = translation_plane.shifted_plane
shifted_plane_equation = translation_plane.shifted_plane_equation
#print(shifted_plane_equation)
k = np.array([0,0,1])
#print(f"d/intercept of the plane equation: {shifted_plane_equation[3:]}")
shifted_plane_vector= shifted_plane_equation[:3]
shifted_plane_vector = np.array(shifted_plane_vector)
#print("Shifted plane equation:")
#print(shifted_plane_vector)
theta = angle_between_planes(shifted_plane_vector.T,k.T)

#print(f'theta: {theta}') 


    
def unit_vector_for_rotation_axis():
    axis = np.cross(shifted_plane_vector.T,k.T)
    #print(axis)
    unit = axis/np.linalg.norm(axis)
    return unit
#print(f"Unit vector : {unit_vector_for_rotation_axis()}")


vector = unit_vector_for_rotation_axis()
u1 = vector[0]
u2 = vector[1]


def Rodriguez_rotation_matrix(u1,u2,angle,tol=1e-8):
    c,s  = np.cos(angle),np.sin(angle)
    #print(u2)
    #print(u1)
    rotation =  np.array([
            [c+(u1**2)*(1-c), u1*u2*(1-c),  u2*s],
    
            [u1*u2*(1-c), c+(u2**2)*(1-c), -1*u1*s],
            
            [-u2*s,       u1*s,            c]        ])
    rotation[np.abs(rotation) < tol] = 0.0
    return rotation 
def apply_Rodriguez_rotation(Rotation,structure,tol=1e-8):
    product = np.dot(Rotation,structure.T).T
    product[np.abs(product) < tol] = 0.0
    return product   
rotation = Rodriguez_rotation_matrix(u1,u2,theta)
#print(rotation)
#print("Product")
transformed_plane = apply_Rodriguez_rotation(rotation,shifted_plane_coords)
#print(transformed_plane)
transformed_protein = apply_Rodriguez_rotation(rotation,translation_plane.shifted_protein_cm)
#print(f"Tranformed Plane : {transformed_plane}")
#print(f"Tranformed Protein: {transformed_protein}")
transformed_protein_BCEF = apply_Rodriguez_rotation(rotation,translation_plane.shifted_protein_BCEF_cm)

#print(f"Tranformed Protein BCEF: {np.array(transformed_protein_BCEF)}")
points_for_BCEF_transform = non_colinear.find_non_collinear_points_fast(transformed_plane)
tranformed_plane_equation = translation_plane.plane_equation_from_points(points_for_BCEF_transform[0],points_for_BCEF_transform[1],points_for_BCEF_transform[2])
#print(f"{tranformed_plane_equation}")
#def get_shifted_plane():
#    shifted_plane = z_shift(plane_coords,c,d) 
#    return shifted_plane
#
