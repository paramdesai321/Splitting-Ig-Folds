import CA_C_N_parsing
import labels 
import numpy as np
import sys
from BestFitLine_Projection import forming_strand_from_indices 
import y_alignment 
import feature_vector
PIN = sys.argv[1]
protein_BCEF = y_alignment.y_aligned_protein_BCEF
protein_BCEF = protein_BCEF.T
def strand_vector(strand):
    return strand[-1] - strand[0]
def angle_of_strand_vector_to_axis(strand,axis):
    vector = strand_vector(strand)
    vector = np.array(vector)
    axis = np.array(axis)
    dot_product = np.dot(vector,axis)
    vector_norm = np.linalg.norm(vector)
    axis_norm = np.linalg.norm(axis)
    phi = np.arccos(dot_product / (vector_norm * axis_norm))
    theta = np.pi/2 -phi
 #   return theta 
    return phi

x = [1,0,0]
y = [0,1,0]
z = [0,0,1]

angle_of_strand_vector_to_axis_feature_vector = []
B_index = labels.B_strand_dict[f"{PIN}_seg0"]
B = forming_strand_from_indices(protein_BCEF,B_index)
angle_of_B_strand_vector_to_x_axis = angle_of_strand_vector_to_axis(B,x)
angle_of_B_strand_vector_to_y_axis = angle_of_strand_vector_to_axis(B,y)
angle_of_B_strand_vector_to_z_axis = angle_of_strand_vector_to_axis(B,z)

#print(f"Angle of B strand to x axis: {angle_of_B_strand_vector_to_x_axis}") 
#print(f"Angle of B strand to y ayis: {angle_of_B_strand_vector_to_y_axis}") 
#print(f"Angle of B strand to z azis: {angle_of_B_strand_vector_to_z_axis}") 
feature_vector.extract_feature(PIN,'Angle of B strand vector to x axis',angle_of_B_strand_vector_to_x_axis)  
feature_vector.extract_feature(PIN,'Angle of B strand vector to y axis',angle_of_B_strand_vector_to_y_axis)  
feature_vector.extract_feature(PIN,'Angle of B strand vector to z axis',angle_of_B_strand_vector_to_z_axis)  

C_index = labels.B_strand_dict[f"{PIN}_seg0"]
C = forming_strand_from_indices(protein_BCEF,C_index)
angle_of_C_strand_vector_to_x_axis = angle_of_strand_vector_to_axis(C,x)
angle_of_C_strand_vector_to_y_axis = angle_of_strand_vector_to_axis(C,y)
angle_of_C_strand_vector_to_z_axis = angle_of_strand_vector_to_axis(C,z)
# print(f"Angle of C strand to x axis: {angle_of_C_strand_vector_to_x_axis}") 
# print(f"Angle of C strand to y ayis: {angle_of_C_strand_vector_to_y_axis}") 
# print(f"Angle of C strand to z azis: {angle_of_C_strand_vector_to_z_axis}") 
feature_vector.extract_feature(PIN,'Angle of C strand vector to x axis',angle_of_C_strand_vector_to_x_axis)  
feature_vector.extract_feature(PIN,'Angle of C strand vector to y axis',angle_of_C_strand_vector_to_y_axis)  
feature_vector.extract_feature(PIN,'Angle of C strand vector to z axis',angle_of_C_strand_vector_to_z_axis)  
E_index = labels.E_strand_dict[f"{PIN}_seg0"]
E = forming_strand_from_indices(protein_BCEF, E_index)
angle_of_E_strand_vector_to_x_axis = angle_of_strand_vector_to_axis(E, x)
angle_of_E_strand_vector_to_y_axis = angle_of_strand_vector_to_axis(E, y)
angle_of_E_strand_vector_to_z_axis = angle_of_strand_vector_to_axis(E, z)
feature_vector.extract_feature(PIN,'Angle of E strand vector to x axis',angle_of_E_strand_vector_to_x_axis)  
feature_vector.extract_feature(PIN,'Angle of E strand vector to y axis',angle_of_E_strand_vector_to_y_axis)  
feature_vector.extract_feature(PIN,'Angle of E strand vector to z axis',angle_of_E_strand_vector_to_z_axis)  
# print(f"Angle of E strand to x axis: {angle_of_E_strand_vector_to_x_axis}")
# print(f"Angle of E strand to y axis: {angle_of_E_strand_vector_to_y_axis}")
# print(f"Angle of E strand to z axis: {angle_of_E_strand_vector_to_z_axis}")

F_index = labels.F_strand_dict[f"{PIN}_seg0"]
F = forming_strand_from_indices(protein_BCEF, F_index)
angle_of_F_strand_vector_to_x_axis = angle_of_strand_vector_to_axis(F, x)
angle_of_F_strand_vector_to_y_axis = angle_of_strand_vector_to_axis(F, y)
angle_of_F_strand_vector_to_z_axis = angle_of_strand_vector_to_axis(F, z)
feature_vector.extract_feature(PIN,'Angle of F strand vector to x axis',angle_of_F_strand_vector_to_x_axis)  
feature_vector.extract_feature(PIN,'Angle of F strand vector to y axis',angle_of_F_strand_vector_to_y_axis)  
feature_vector.extract_feature(PIN,'Angle of F strand vector to z axis',angle_of_F_strand_vector_to_z_axis)  
# print(f"Angle of F strand to x axis: {angle_of_F_strand_vector_to_x_axis}")
# print(f"Angle of F strand to y axis: {angle_of_F_strand_vector_to_y_axis}")
# print(f"Angle of F strand to z axis: {angle_of_F_strand_vector_to_z_axis}")
# print(feature_vector.get_features())
with open(f"features.txt", 'a') as wf:
    wf.write(f"Angle of B strand to x axis: {angle_of_B_strand_vector_to_x_axis}\n")
    wf.write(f"Angle of B strand to y axis: {angle_of_B_strand_vector_to_y_axis}\n")
    wf.write(f"Angle of B strand to z axis: {angle_of_B_strand_vector_to_z_axis}\n")
    wf.write(f"Angle of C strand to x axis: {angle_of_C_strand_vector_to_x_axis}\n")
    wf.write(f"Angle of C strand to y axis: {angle_of_C_strand_vector_to_y_axis}\n")
    wf.write(f"Angle of C strand to z axis: {angle_of_C_strand_vector_to_z_axis}\n")
    wf.write(f"Angle of E strand to x axis: {angle_of_E_strand_vector_to_x_axis}\n")
    wf.write(f"Angle of E strand to y axis: {angle_of_E_strand_vector_to_y_axis}\n")
    wf.write(f"Angle of E strand to z axis: {angle_of_E_strand_vector_to_z_axis}\n")
    wf.write(f"Angle of F strand to x axis: {angle_of_F_strand_vector_to_x_axis}\n")
    wf.write(f"Angle of F strand to y axis: {angle_of_F_strand_vector_to_y_axis}\n")
    wf.write(f"Angle of F strand to z axis: {angle_of_F_strand_vector_to_z_axis}\n")

