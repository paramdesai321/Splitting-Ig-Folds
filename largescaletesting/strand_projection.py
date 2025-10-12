import os 
import numpy as np
import sys
from sklearn_svm import model,apply_rotation_matrix

PIN = sys.argv[1]


def project_point_to_plane_implicit(P, plane,intercept):
    
    w1, w2, w3 = plane
    b = intercept
    P = np.array(P, dtype=float)
    x0, y0, z0 = P
    t = (w1*x0 + w2*y0 + w3*z0 + b) / (w1*w1 + w2*w2 + w3*w3)
    return np.array([x0 - w1*t, y0 - w2*t, z0 - w3*t])

def create_projections(rf, wf, Strand):
    i = 0
    for line in rf:
        if line.startswith("ATOM") and line[21:22].strip() == Strand:
            # Parse original coordinates from PDB columns
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            print(line[21:22].strip())
            # Project onto the plane defined by the SVM model
            projected = project_point_to_plane_implicit([x, y, z], model.coef_[0], model.intercept_)
            # Apply rotation matrix to align projections
            rotated = apply_rotation_matrix(projected.T)

            # Unpack rotated coordinates
            x_p, y_p, z_p = map(float, rotated)

            # Format new HETATM line for the projection
            new_line = (
                "HETATM{:5d}  O   PLN A{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00\n"
            ).format(i + 1, 999, x_p, y_p, z_p)

            wf.write(new_line)
            i += 1


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <PIN>")
        sys.exit(1)

    PIN = sys.argv[1]
    input_path = f'./Beta_Strands/ATOMlines{PIN}_BCEF_Beta.pdb'

    # Generate projections for each specified strand
    for strand in ['B', 'C', 'E', 'F']:
        output_path = f'{PIN}_{strand}_Projection.pdb'
        with open(input_path, 'r') as rf, open(output_path, 'w') as wf:
            create_projections(rf, wf, strand)
#    print("heheheh")
main()

