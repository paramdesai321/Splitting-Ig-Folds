import y_alignment                                                                                                                  
import labels                                                                                                                       
from BestFitLine_Projection import forming_strand_from_indices                                                                     
import sys                                                                                                                         
import numpy as np                                                                                                                 
import angle_between_strands
import feature_vector
PIN = sys.argv[1]                                                                                                                  
protein_BCEF = y_alignment.y_aligned_protein_BCEF                                                                                  
protein_BCEF = protein_BCEF.T   

def strand_projections_on_z_plane(strand):  # on the plane z=0
    # make sure strand is n x 3
    assert strand.shape[1] ==3
    strand[:, 2] = 0
    return strand

B_index = labels.B_strand_dict[f"{PIN}_seg0"]                                                                                       
B = forming_strand_from_indices(protein_BCEF, B_index)  
B_projection = strand_projections_on_z_plane(B)
B_projection_vector = angle_between_strands.strand_vector(B)

C_index = labels.C_strand_dict[f"{PIN}_seg0"]
C = forming_strand_from_indices(protein_BCEF, C_index) 
C_projection = strand_projections_on_z_plane(C)
C_projection_vector = angle_between_strands.strand_vector(C)

E_index = labels.E_strand_dict[f"{PIN}_seg0"]
E = forming_strand_from_indices(protein_BCEF, E_index) 
E_projection = strand_projections_on_z_plane(E)
E_projection_vector = angle_between_strands.strand_vector(E)

F_index = labels.F_strand_dict[f"{PIN}_seg0"]
F = forming_strand_from_indices(protein_BCEF, F_index) 
F_projection = strand_projections_on_z_plane(F) 
F_projection_vector = angle_between_strands.strand_vector(F)

angle_between_strand_projections = []
angle_between_strand_projections_B_C = angle_between_strands.angle_between_strand_vectors(B,C)
# print(f"Angle between the projections of B and C on the z=0 plane: {angle_between_strand_projections_B_C} ")
feature_vector.extract_feature(PIN,"Angle between the projections of B and C on the z=0 plane",{angle_between_strand_projections_B_C})
angle_between_strand_projections.append(angle_between_strand_projections_B_C)
angle_between_strand_projections_B_F = angle_between_strands.angle_between_strand_vectors(B, F)
# print(f"Angle between the projections of B and F on the z=0 plane: {angle_between_strand_projections_B_F}")
feature_vector.extract_feature(PIN,"Angle between the projections of B and F on the z=0 plane",{angle_between_strand_projections_B_F})
angle_between_strand_projections.append(angle_between_strand_projections_B_F)

angle_between_strand_projections_C_E = angle_between_strands.angle_between_strand_vectors(C, E)
# print(f"Angle between the projections of C and E on the z=0 plane: {angle_between_strand_projections_C_E}")
feature_vector.extract_feature(PIN,"Angle between the projections of C and E on the z=0 plane",{angle_between_strand_projections_C_E})
angle_between_strand_projections.append(angle_between_strand_projections_C_E)
angle_between_strand_projections_E_F = angle_between_strands.angle_between_strand_vectors(E, F)
feature_vector.extract_feature(PIN,"Angle between the projections of E and F on the z=0 plane",{angle_between_strand_projections_E_F})
angle_between_strand_projections.append(angle_between_strand_projections_E_F)

angle_between_strands = np.array(angle_between_strands)
# print(feature_vector.get_features())
with open(f"features.txt", 'a') as wf:
    wf.write(f"Angle between the projections of B and C on the z=0 plane: {angle_between_strand_projections_B_C} \n")
    wf.write(f"Angle between the projections of B and F on the z=0 plane: {angle_between_strand_projections_B_F} \n")
    wf.write(f"Angle between the projections of C and E on the z=0 plane: {angle_between_strand_projections_C_E} \n")
    wf.write(f"Angle between the projections of E and F on the z=0 plane: {angle_between_strand_projections_E_F} \n")


    
