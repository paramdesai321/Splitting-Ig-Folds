import y_alignment
import labels
from BestFitLine_Projection import forming_strand_from_indices
import sys
import numpy as np
import feature_vector
PIN = sys.argv[1]                                                                                                                 
protein_BCEF = y_alignment.y_aligned_protein_BCEF                                                                                   
protein_BCEF = protein_BCEF.T                                                                                                           
def strand_vector(strand):
    return strand[-1] - strand[0]
def angle_between_strand_vectors(s1,s2):
        
    s1 = np.array(s1, dtype=float)
    s2 = np.array(s2, dtype=float)
    s1 = strand_vector(s1)
    s2 = strand_vector(s2)
    dot = np.dot(s1, s2)
    norm_s1 = np.linalg.norm(s1)
    norm_s2 = np.linalg.norm(s2)
    s1 = s1/norm_s1
    s2 = s2/norm_s2
    
     
    cos_theta = np.dot(s1, s2)
    return np.arccos(cos_theta)
B_index = labels.B_strand_dict[f"{PIN}_seg0"]
B = forming_strand_from_indices(protein_BCEF, B_index)
C_index = labels.C_strand_dict[f"{PIN}_seg0"]
C= forming_strand_from_indices(protein_BCEF, C_index)
E_index = labels.E_strand_dict[f"{PIN}_seg0"]
E= forming_strand_from_indices(protein_BCEF, E_index)
F_index = labels.F_strand_dict[f"{PIN}_seg0"]
F = forming_strand_from_indices(protein_BCEF, F_index)
angle_between_B_C = angle_between_strand_vectors(B,C)

# print(f"Angle between B and C : {angle_between_B_C}")
feature_vector.extract_feature(PIN,"Angle between B and C",angle_between_B_C)
angle_between_B_F = angle_between_strand_vectors(B, F)
# print(f"Angle between B and F : {angle_between_B_F}")
feature_vector.extract_feature(PIN,"Angle between B and F",angle_between_B_F)

angle_between_C_E = angle_between_strand_vectors(C, E)
# print(f"Angle between C and E : {angle_between_C_E}")
feature_vector.extract_feature(PIN,"Angle between C and E",angle_between_C_E)

angle_between_C_F = angle_between_strand_vectors(C, F)
# print(f"Angle between C and F : {angle_between_C_F}")
feature_vector.extract_feature(PIN,"Angle between C and F",angle_between_C_F)
angle_between_B_E = angle_between_strand_vectors(B, E)
# print(f"Angle between B and E : {angle_between_B_E}")
feature_vector.extract_feature(PIN,"Angle between B and E",angle_between_B_E)
# print(feature_vector.get_features())
angle_between_E_F = angle_between_strand_vectors(E,F)
feature_vector.extract_feature(PIN,"Angle between E and F",angle_between_E_F)

with open(f"features.txt", 'a') as wf:
    wf.write(f"Angle between B and C : {angle_between_B_C}\n")
    wf.write(f"Angle between B and F : {angle_between_B_F}\n")
    wf.write(f"Angle between C and E : {angle_between_C_E}\n")
    wf.write(f"Angle between C and F : {angle_between_C_F}\n")
    wf.write(f"Angle between B and E : {angle_between_B_E}\n")
    wf.write(f"Angle between E and F : {angle_between_E_F}\n")

