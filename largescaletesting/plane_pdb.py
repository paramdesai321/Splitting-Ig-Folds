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
#print("Coords from Plane: ",coords_from_plane)
#coords_from_plane = transformation.transformed_plane
#print("##")
#print(coords_from_plane.shape)

def lim():
    T = KDTree(coords_from_protein)
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

def create_pdb_with_coordinates(output_file=f"./Plane_CD_HIT_90/plane_{PIN}_seg0.pdb"):
    with open(output_file, "w") as f:
        neigbors = lim()
        for i in range(len(desired_plane_atoms)):
            line = "HETATM{:5d}  H   PLN P{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00\n".format(
                i + 1, 999, desired_plane_atoms[i][0], desired_plane_atoms[i][1], desired_plane_atoms[i][2]
            )   
            f.write(line)
desired_plane_atoms = np.array(desired_plane_atoms) # leave this here!! you want this list to be np array for the following fn
def reference_points(output_file=f"./Plane_CD_HIT_90/plane_{PIN}.pdb"):
   #desired_plane_atoms = np.asarray(desired_plane_atoms, dtype=float)
   if desired_plane_atoms.ndim != 2 or desired_plane_atoms.shape[1] < 3:
       raise ValueError("desired_plane_atoms must be an array of shape (N,3).")
   if coords_from_protein.ndim !=2 or coords_from_protein.shape[1] < 3: 
       raise ValueError("Y aligned protein must be an array of shape (N,3).")

   xs = desired_plane_atoms[:, 0]
   ys = desired_plane_atoms[:, 1]
   zs = coords_from_protein[:, 2] # Note that zs from plane coords is always 0, becase the plane is z=0
   x_max, x_min = int(np.max(xs)), int(np.min(xs))
   y_max, y_min = int(np.max(ys)), int(np.min(ys))
   z_max, z_min = int(np.max(zs)), int(np.min(zs))
   
   
   #print(desired_plane_atoms[x_min]) 
   ordered_refs = [
       ('O', np.array([x_min,0,0])),  # x min
       ('N', np.array([x_max,0,0])),  # x max
       ('O', np.array([0,y_min,0])),  # y min
       ('N', np.array([0,y_max,0])),  # y max
       ('O', np.array([0,0,z_min])),  # z min
       ('N', np.array([0,0,z_max]))  # z max
   ]
   #print(z_max)
   
   serial = 1
   with open(output_file, 'a') as f:
       for atom_label, coords in ordered_refs:
           x, y, z = map(float, coords[:3])
           line = "HETATM{:5d}  {:>2s}  PLN P{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00\n".format(
               serial, atom_label, 999, x, y, z
           )
           f.write(line)
           serial += 1

   return ordered_refs
  


if __name__ == "__main__":
    # Testcoordinates array
    coordinates = [ 
        (98.804, 73.247, 25.567),
        (99.123, 74.001, 26.010),
        (97.654, 72.888, 24.999)
    ]   

    create_pdb_with_coordinates()
    reference_points()

    #create_pdb_with_coordinates(coordinates[0],coordinates[1],coordinates[2])
 #   print("Custom PDB created!")
