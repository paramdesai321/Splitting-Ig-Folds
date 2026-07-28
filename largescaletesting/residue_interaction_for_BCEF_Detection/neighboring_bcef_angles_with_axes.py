
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


def strand_vector_from_label(label_coords, label):
    coords = label_coords[label]["coords"]
    centroids = backbone_centroids_from_coords(coords)

    if len(centroids) < 2:
        return None

    v = centroids[-1] - centroids[0]
    norm = np.linalg.norm(v)

    if norm == 0:
        return None

    return v / norm


def angle_vector_to_axis(v, axis, degrees=True):
    if v is None:
        return None

    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)

    if axis_norm == 0:
        raise ValueError("Axis cannot be zero vector.")

    axis = axis / axis_norm

    cos_theta = np.dot(v, axis)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle = np.arccos(cos_theta)

    if degrees:
        angle = np.degrees(angle)

    return float(angle)

def neighboring_strand_axis_angles(label_coords, degrees=True):
    """
    Returns axis angles only for the neighboring strands.
    """

    neighboring_labels = [
        "B-1",
        "C+1",
        "E-1",
        "F+1",
    ]

    axes = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }

    results = {}

    for label in neighboring_labels:

        if label not in label_coords:
            continue

        v = strand_vector_from_label(label_coords, label)

        results[label] = {
            axis_name: angle_vector_to_axis(
                v,
                axis,
                degrees=degrees
            )
            for axis_name, axis in axes.items()
        }

    return results

def calculate_features(label_coords):
    """
    Return the 12 neighboring-strand axis-angle features in flat form.

    Missing strands or invalid strand vectors produce None.
    The batch feature-matrix script will later replace None with 0.0.
    """

    angles = neighboring_strand_axis_angles(
        label_coords,
        degrees=True
    )

    label_names = {
        "B-1": "Bm1",
        "C+1": "Cp1",
        "E-1": "Em1",
        "F+1": "Fp1",
    }

    features = {}

    for label, feature_label in label_names.items():
        strand_angles = angles.get(label, {})

        features[f"axis_angle_{feature_label}_x"] = strand_angles.get("x")
        features[f"axis_angle_{feature_label}_y"] = strand_angles.get("y")
        features[f"axis_angle_{feature_label}_z"] = strand_angles.get("z")

    return features

def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python angle_from_axes_neighboring_strands.py <normal_pdb_file> <bcef_pdb_file>")
        sys.exit(1)

    pdb_path = sys.argv[1]
    bcef_path = sys.argv[2]

    label_coords = dssp_label_to_range_mapping.get_label_coords_mapping(
        pdb_path,
        bcef_path
    )

    angles = neighboring_strand_axis_angles(
        label_coords,
        degrees=True
    )

    #print(angles)


if __name__ == "__main__":
    main()
