import y_alignment 
import numpy as np
import labels
import sys
import CA_C_N_parsing
from BestFitLine_Projection import forming_strand_from_indices
import feature_vector
PIN = sys.argv[1]

protein_BCEF = y_alignment.y_aligned_protein_BCEF

protein_BCEF = protein_BCEF.T
#print(protein_BCEF.shape)
def cm_of_strand(strand):
     strand  = np.array(strand)
#      print(f"Shape of Strand: {strand.shape}")
     x = np.mean(strand[:,0])
     y = np.mean(strand[:,1])
     z = np.mean(strand[:,2])
     return x,y,z
def dist_of_cm_from_plane(strand,plane):
    centroid = cm_of_strand(strand)
    
    x0, y0, z0 = centroid
    A, B, C, D = plane
    
    numerator = abs(A*x0 + B*y0 + C*z0 + D)
    denominator = np.sqrt(A**2 + B**2 + C**2)
    
    return numerator / denominator


B_index = labels.B_strand_dict[f"{PIN}_seg0"]
B= forming_strand_from_indices(protein_BCEF,B_index)
dist_of_B_cm_from_plane = dist_of_cm_from_plane(B,[0,0,1,0])
feature_vector.extract_feature(PIN,'Distance of Center of Mass of B from plane',dist_of_B_cm_from_plane)
# print(f"Distance of CM of B from Plane z=0: {dist_of_B_cm_from_plane}")

C_index = labels.C_strand_dict[f"{PIN}_seg0"]
C= forming_strand_from_indices(protein_BCEF,C_index)
dist_of_C_cm_from_plane = dist_of_cm_from_plane(C,[0,0,1,0])
feature_vector.extract_feature(PIN,'Distance of Center of Mass of C from plane',dist_of_C_cm_from_plane)
# print(f"Distance of CM of C from Plane z=0: {dist_of_C_cm_from_plane}")

E_index = labels.E_strand_dict[f"{PIN}_seg0"]
E= forming_strand_from_indices(protein_BCEF,E_index)
dist_of_E_cm_from_plane = dist_of_cm_from_plane(E,[0,0,1,0])
feature_vector.extract_feature(PIN,'Distance of Center of Mass of E from plane',dist_of_E_cm_from_plane)
# print(f"Distance of CM of E from Plane z=0: {dist_of_E_cm_from_plane}")

F_index = labels.F_strand_dict[f"{PIN}_seg0"]
F= forming_strand_from_indices(protein_BCEF,F_index)
dist_of_F_cm_from_plane = dist_of_cm_from_plane(F,[0,0,1,0])
feature_vector.extract_feature(PIN,'Distance of Center of Mass of F from plane',dist_of_F_cm_from_plane)
# print(feature_vector.get_features())
# print(f"Distance of CM of F from Plane z=0: {dist_of_F_cm_from_plane}")
with open(f"features.txt",'a') as wf:
    wf.write(f"{PIN} \n")
    wf.write(f"Distance of CM of B from Plane z=0: {dist_of_B_cm_from_plane} \n")
    wf.write(f"Distance of CM of C from Plane z=0: {dist_of_C_cm_from_plane} \n") 
    wf.write(f"Distance of CM of E form plane z=0 : {dist_of_E_cm_from_plane} \n")
    wf.write(f"Distance of CM of F form plane z=0 : {dist_of_F_cm_from_plane} \n")
def export_dist_strand_cm_to_plane_vector():
    dist_strand_cm_plane = [
        dist_of_B_cm_from_plane,
        dist_of_C_cm_from_plane,
        dist_of_E_cm_from_plane,
        dist_of_F_cm_from_plane,
    ]
    return np.array(dist_strand_cm_plane)


