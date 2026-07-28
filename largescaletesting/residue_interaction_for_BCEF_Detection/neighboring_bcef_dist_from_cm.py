# dist_from_cm_neighboring_strands.py

import sys
import numpy as np

import dssp_label_to_range_mapping


def cm_of_strand(coords):
    coords = np.asarray(coords, dtype=float)
    return np.mean(coords, axis=0)


def dist_of_cm_from_plane(coords, plane=(0, 0, 1, 0)):
    centroid = cm_of_strand(coords)

    x0, y0, z0 = centroid
    A, B, C, D = plane

    numerator = abs(A*x0 + B*y0 + C*z0 + D)
    denominator = np.sqrt(A**2 + B**2 + C**2)

    if denominator == 0:
        raise ValueError("Invalid plane: normal vector cannot be zero.")

    return float(numerator / denominator)


def neighboring_strand_cm_distances(label_coords):
    """
    Computes distance of neighboring strand center of mass from z=0 plane.

    Only computes:
      B-1, C+1, E-1, F+1
    """

    neighboring_labels = [
        "B-1",
        "C+1",
        "E-1",
        "F+1",
    ]

    results = {}

    for label in neighboring_labels:
        if label not in label_coords:
            continue

        coords = label_coords[label]["coords"]
        results[label] = dist_of_cm_from_plane(coords, plane=(0, 0, 1, 0))

    return results

def calculate_features(label_coords):
    distances = neighboring_strand_cm_distances(label_coords)

    return {
        "cm_distance_Bm1": distances.get("B-1"),
        "cm_distance_Cp1": distances.get("C+1"),
        "cm_distance_Em1": distances.get("E-1"),
        "cm_distance_Fp1": distances.get("F+1"),
    }
def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python dist_from_cm_neighboring_strands.py <normal_pdb_file> <bcef_pdb_file>")
        sys.exit(1)

    pdb_path = sys.argv[1]
    bcef_path = sys.argv[2]

    label_coords = dssp_label_to_range_mapping.get_label_coords_mapping(
        pdb_path,
        bcef_path
    )

    distances = neighboring_strand_cm_distances(label_coords)

    #print(distances)


if __name__ == "__main__":
    main()
