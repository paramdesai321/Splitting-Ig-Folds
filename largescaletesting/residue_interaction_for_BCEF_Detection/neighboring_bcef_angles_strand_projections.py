# neighboring_bcef_projection_angles.py

import sys
import numpy as np

import dssp_label_to_range_mapping


def backbone_centroids_from_coords(coords):
    """
    coords are ordered:
      N, CA, C, N, CA, C, ...

    Returns one backbone centroid per residue.
    """
    coords = np.asarray(coords, dtype=float)

    if len(coords) % 3 != 0:
        raise ValueError("Expected coords to be ordered as N, CA, C triplets.")

    residue_centroids = []

    for i in range(0, len(coords), 3):
        residue_atoms = coords[i:i+3]
        centroid = np.mean(residue_atoms, axis=0)
        residue_centroids.append(centroid)

    return np.asarray(residue_centroids)


def project_points_to_z0(points):
    """
    Project 3D points to z=0 plane:
      (x,y,z) -> (x,y,0)
    """
    projected = np.asarray(points, dtype=float).copy()
    projected[:, 2] = 0.0
    return projected


def projected_strand_vector_from_label(label_coords, label):
    """
    Build strand vector from first and last backbone centroid
    after projecting the strand to z=0.
    """
    coords = label_coords[label]["coords"]
    centroids = backbone_centroids_from_coords(coords)

    if len(centroids) < 2:
        return None

    projected = project_points_to_z0(centroids)

    v = projected[-1] - projected[0]
    norm = np.linalg.norm(v)

    if norm == 0:
        return None

    return v / norm


def angle_between_vectors(v1, v2, degrees=True, unsigned=False):
    """
    unsigned=False gives angle in [0,180].
    unsigned=True treats parallel/anti-parallel as similar, angle in [0,90].
    """
    if v1 is None or v2 is None:
        return None

    cos_theta = np.dot(v1, v2)

    if unsigned:
        cos_theta = abs(cos_theta)

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle = np.arccos(cos_theta)

    if degrees:
        angle = np.degrees(angle)

    return float(angle)


def projection_angle_between_labeled_strands(label_coords, label1, label2, degrees=True, unsigned=False):
    v1 = projected_strand_vector_from_label(label_coords, label1)
    v2 = projected_strand_vector_from_label(label_coords, label2)

    return angle_between_vectors(
        v1,
        v2,
        degrees=degrees,
        unsigned=unsigned
    )


def neighboring_bcef_projection_angles(label_coords, degrees=True, unsigned=False):
    """
    Computes projection angles between neighboring BCEF-relative strand labels.

    Pairs:
      B-1 vs B
      C vs C+1
      E-1 vs E
      F vs F+1

    Returns:
        {
            ("B-1", "B"): angle or None,
            ("C", "C+1"): angle or None,
            ("E-1", "E"): angle or None,
            ("F", "F+1"): angle or None
        }
    """

    pairs = [
        ("B-1", "B"),
        ("C", "C+1"),
        ("E-1", "E"),
        ("F", "F+1"),
    ]

    angles = {}

    for a, b in pairs:
        if a not in label_coords or b not in label_coords:
            angles[(a, b)] = None
            continue

        angles[(a, b)] = projection_angle_between_labeled_strands(
            label_coords,
            a,
            b,
            degrees=degrees,
            unsigned=unsigned
        )

    return angles

def calculate_features(label_coords):
    angles = neighboring_bcef_projection_angles(
        label_coords,
        degrees=True,
        unsigned=False
    )

    return {
        "angle_3d_Bm1_B": angles.get(("B-1", "B")),
        "angle_3d_C_Cp1": angles.get(("C", "C+1")),
        "angle_3d_Em1_E": angles.get(("E-1", "E")),
        "angle_3d_F_Fp1": angles.get(("F", "F+1")),
    }
def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python neighboring_bcef_projection_angles.py <normal_pdb_file> <bcef_pdb_file>")
        sys.exit(1)

    pdb_path = sys.argv[1]
    bcef_path = sys.argv[2]

    label_coords = dssp_label_to_range_mapping.get_label_coords_mapping(
        pdb_path,
        bcef_path
    )

    angles = neighboring_bcef_projection_angles(
        label_coords,
        degrees=True,
        unsigned=False
    )

    #print(angles)


if __name__ == "__main__":
    main()
