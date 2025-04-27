import sklearn_svm as svm


coords = svm.get_plane_coords();
x = coords[0][0]
y = coords[1][0]
z = coords[1][0]

def create_pdb_with_coordinates(x,y,z, output_file="custom.pdb"):
    # Fixed header for your format
    header = "ATOM      1  O   PLN z 999"

    with open(output_file, "w") as f:
        for i in range(len(x)):
            line = f"{header}{x[i]:9.3f}{y[i]:8.3f}{z[i]:8.3f}\n"
            f.write(line)
        #f.write("END\n")

if __name__ == "__main__":
    # Example coordinates array
    coordinates = [
        (98.804, 73.247, 25.567),
        (99.123, 74.001, 26.010),
        (97.654, 72.888, 24.999)
    ]

    create_pdb_with_coordinates(x,y,z)
    print("Custom PDB created!")
       
                         
    
