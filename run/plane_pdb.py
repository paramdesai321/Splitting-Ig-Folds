import sklearn_svm as svm
import numpy as np
import sys
coords = svm.get_plane_coords();

PIN = sys.argv[1]

def create_pdb_with_coordinates(coords, output_file=f"plane_{PIN}.pdb"):
    with open(output_file, "w") as f:
        for i in range(len(coords)):
            line = "ATOM  {:5d}  F   PLN A{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00\n".format(
                i + 1, 999, coords[i][0], coords[i][1], coords[i][2]
            )
            f.write(line)


if __name__ == "__main__":
    # Example coordinates array
    coordinates = [
        (98.804, 73.247, 25.567),
        (99.123, 74.001, 26.010),
        (97.654, 72.888, 24.999)
    ]

    create_pdb_with_coordinates(coords)

    #create_pdb_with_coordinates(coordinates[0],coordinates[1],coordinates[2])
    print("Custom PDB created!")
       
                         
    
