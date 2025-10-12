from scipy.spatial import KDTree
import sklearn_svm as svm 
import numpy as np
from parsing_coords import coordinates 
import y_alignment 
import transformation
import sys 
import plane_grid
coords_from_protein = y_alignment.y_aligned_protein
#print(coords_from_protein)
#print("@@@")
#print(coords_from_protein.shape)
coords_from_plane = plane_grid.make_square_plane_grid() 
#coords_from_plane = transformation.transformed_plane
#print("##")
#print(coords_from_plane.shape)

def lim():
    T = KDTree(np.transpose(coords_from_protein))
    neighbors = T.query_ball_point(coords_from_plane,20)
    return [i for i, nbrs in enumerate(neighbors) if nbrs]

#matches = lim()
#print(f"positive match: {matches}")
#print(f"Number of matches: {len(matches)}")
#desired_plane_atoms = []
#for match in matches:
#    desired_plane_atoms.append(coords_from_plane[match])
#print(f"Desired Atoms: {desired_plane_atoms}")
#print(f"Number of Desired Atoms: {len(desired_plane_atoms)}")
        
desired_plane_atoms = coords_from_plane
PIN = sys.argv[1]

def create_pdb_with_coordinates(output_file=f"plane_{PIN}.pdb"):
    with open(output_file, "w") as f:
        neigbors = lim()
        for i in range(len(desired_plane_atoms)):
            line = "HETATM{:5d}  H   PLN A{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00\n".format(
                i + 1, 999, desired_plane_atoms[i][0], desired_plane_atoms[i][1], desired_plane_atoms[i][2]
            )   
            f.write(line)


if __name__ == "__main__":
    # Testcoordinates array
    coordinates = [ 
        (98.804, 73.247, 25.567),
        (99.123, 74.001, 26.010),
        (97.654, 72.888, 24.999)
    ]   

    create_pdb_with_coordinates()

    #create_pdb_with_coordinates(coordinates[0],coordinates[1],coordinates[2])
 #   print("Custom PDB created!")
