# length_of_neighboring_strands.py

import sys
import numpy as np

import dssp_label_to_range_mapping


def backbone_centroids_from_coords(coords):
    coords = np.asarray(coords, dtype=float)

    if len(coords) % 3 != 0:
        raise ValueError("Expected coords to be ordered as N, CA, C triplets.")

    centroids = []

    for i in range(0, len(coords), 3):
        residue_atoms = coords[i:i+3]
        centroids.append(np.mean(residue_atoms, axis=0))

    return np.asarray(centroids)


def strand_geometric_length(label_coords, label):
    """
    End-to-end geometric strand length using backbone centroids.
    """
    coords = label_coords[label]["coords"]
    centroids = backbone_centroids_from_coords(coords)

    if len(centroids) < 2:
        return None

    v = centroids[-1] - centroids[0]
    return float(np.linalg.norm(v))


def neighboring_strand_lengths(label_coords):
    """
    Only computes geometric lengths for neighboring non-BCEF strands:
      B-1, C+1, E-1, F+1
    """

    neighboring_labels = [
        "B-1",
        "C+1",
        "E-1",
        "F+1",
    ]

    lengths = {}

    for label in neighboring_labels:
        if label not in label_coords:
            continue

        lengths[label] = strand_geometric_length(label_coords, label)

    return lengths
def calculate_features(label_coords):
    lengths = neighboring_strand_lengths(label_coords)

    return {
        "length_Bm1": lengths.get("B-1"),
        "length_Cp1": lengths.get("C+1"),
        "length_Em1": lengths.get("E-1"),
        "length_Fp1": lengths.get("F+1"),
    }

def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python length_of_neighboring_strands.py <normal_pdb_file> <bcef_pdb_file>")
        sys.exit(1)

    pdb_path = sys.argv[1]
    bcef_path = sys.argv[2]

    label_coords = dssp_label_to_range_mapping.get_label_coords_mapping(
        pdb_path,
        bcef_path
    )

    lengths = neighboring_strand_lengths(label_coords)

    #print(lengths)


if __name__ == "__main__":
    main()
