import os
import sys
import numpy as np
import re
from CA_C_N_parsing import coordinates
import parsing_coords 
from scipy.spatial import KDTree 
import non_colinear 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
X = coordinates()
protein =  parsing_coords.coordinates()
transposed_X = np.array(X).T
#print(f"Transposed X: {transposed_X.shape}")
# Taking centroid of x,y coordinates of Backbone BCEF
def centroid(x,y,z):
    result_x = np.sum(x)/len(x)
    result_y = np.sum(y)/len(y)
    result_z = np.sum(z)/len(z)
    return np.array([result_x,result_y,result_z])


#print(f"Centroid before: {centroid(transposed_X[0],transposed_X[1],transposed_X[2])}")
def nearest_plane_coord(plane_coords,centroid_point):
#    print(centroid_point)
    T = KDTree(plane_coords)
    distance, index = T.query(centroid_point,k=1)
    return distance,index

 
 
#print(index)


def shift_structure(structure, shift_point):
    shifted = []
    for atom in structure:
        shifted_atom = [
            atom[0] - shift_point[0],
            atom[1] - shift_point[1],
            atom[2] - shift_point[2]
        ]
        shifted.append(shifted_atom)
    return np.array(shifted)


centroid_BCEF = centroid(transposed_X[0],transposed_X[1],transposed_X[2])
print(f"Centroid of BCEF: {centroid_BCEF}")
shifted_protein_BCEF = shift_structure(X,centroid_BCEF)
#print(shifted_protein_BCEF.shape)
centroid_shifted_BCEF = centroid(shifted_protein_BCEF.T[0],shifted_protein_BCEF.T[1],shifted_protein_BCEF.T[2]) 

print(centroid_shifted_BCEF)
shifted_protein = shift_structure(protein,centroid_BCEF)

def get_centroid_BCEF(tol=1e-5):
    rounded_centroid = []
    for elem in centroid_BCEF:
        if abs(elem) > tol:
            elem = 0.0
            rounded_centroid.append(elem)
    return np.array(rounded_centroid)
    

  # flipped_protein


#orth = svm.angle_between_planes([])
#def R_y(angle):
#   c,s = np.cos(angle),np.sin(angle)
#   rotation_matrix = np.array([[c,0,-s],
#                               [0,1,0],
#                                [s,0,c]])
#   return rotation_matrix
#
#def applying_rotation_matrix(grid):
#  
#    angle_to_y = svm.angle_between_planes(svm.model.coef_[0],[0,1,0]) # protein is not aligned to x axis
#    return np.dot(R_y(angle_to_y),np.transpose(grid))
#    
##    
#X = np.array(X)
#print(X.T)
#shift = centroid(np.array(X.T[0]),np.array(X.T[1]))
#shift = np.append(shift,0)
#coords_of_aligned_protein = (np.array(parsing_coords.coordinates())-shift)
#
#print(coords_of_aligned_protein)
#angle_to_y = svm.angle_between_planes(svm.model.coef_[0],[0,1,0]) 
#print(angle_to_y)
#applying_rotation_matrix(coords_of_aligned_protein)
#angle_to_z = svm.angle_between_planes(svm.model.coef_[0],[0,0,1]) # protein is not aligned to x axis
#
# 

