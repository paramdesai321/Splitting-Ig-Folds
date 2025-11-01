import numpy as np
from BestFitLine_Projection import forming_strand_from_indices
import feature_vector
import sys
import y_alignment
import labels

PIN = sys.argv[1]
protein_BCEF = y_alignment.y_aligned_protein_BCEF          
protein_BCEF = protein_BCEF.T
def strand_vector(strand):
    return strand[-1] - strand[0]

def magnitude_of_strand_vector(strand):
    vec = strand_vector(strand)
    return np.linalg.norm(vec)

B_index = labels.B_strand_dict[f"{PIN}_seg0"]
B = forming_strand_from_indices(protein_BCEF, B_index)
C_index = labels.C_strand_dict[f"{PIN}_seg0"]
C= forming_strand_from_indices(protein_BCEF, C_index)
E_index = labels.E_strand_dict[f"{PIN}_seg0"]
E= forming_strand_from_indices(protein_BCEF, E_index)
F_index = labels.F_strand_dict[f"{PIN}_seg0"]
F = forming_strand_from_indices(protein_BCEF, F_index)

B_mag = magnitude_of_strand_vector(B)
feature_vector.extract_feature(PIN,"Length of B Strand Vector",B_mag)

C_mag = magnitude_of_strand_vector(C)
feature_vector.extract_feature(PIN,"Length of C Strand Vector",C_mag)

E_mag = magnitude_of_strand_vector(E)
feature_vector.extract_feature(PIN,"Length of E Strand Vector",E_mag)

F_mag = magnitude_of_strand_vector(F)
feature_vector.extract_feature(PIN,"Length of F Strand Vector",F_mag)
