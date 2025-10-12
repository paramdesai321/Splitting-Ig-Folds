from scipy.spatial import KDTree
import sklearn_svm as svm 
import numpy as np
from parsing_coords import coordinates 
import sys 
coords_from_protein = coordinates()
coords_from_plane = svm.get_plane_coords();
def lim():
    T = KDTree(coords_from_protein)
    neighbors = T.query_ball_point(coords_from_plane,5)
    return [i for i, nbrs in enumerate(neighbors) if nbrs]

matches = lim()
print(f"positive match: {matches}")
print(f"Number of matches: {len(matches)}")
desired_plane_atoms = []
for match in matches:
    desired_plane_atoms.append(coords_from_plane[match])
print(f"Desired Atoms: {desired_plane_atoms}")
print(f"Number of Desired Atoms: {len(desired_plane_atoms)}")
        
    
PIN = sys.argv[1]

def create_pdb_with_coordinates(output_file=f"plane_{PIN}.pdb"):
    with open(output_file, "w") as f:
        neigbors = lim()
        for i in range(len(desired_plane_atoms)):
            line = "HETATM{:5d}  F   PLN A{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00\n".format(
                i + 1, 999, desired_plane_atoms[i][0], desired_plane_atoms[i][1], desired_plane_atoms[i][2]
            )   
            f.write(line)


if __name__ == "__main__":
    # Example coordinates array
    coordinates = [ 
        (98.804, 73.247, 25.567),
        (99.123, 74.001, 26.010),
        (97.654, 72.888, 24.999)
    ]   

    create_pdb_with_coordinates()

    #create_pdb_with_coordinates(coordinates[0],coordinates[1],coordinates[2])
    print("Custom PDB created!")
