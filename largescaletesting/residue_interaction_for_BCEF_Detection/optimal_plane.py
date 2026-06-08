from distance_rep_point_per_strand import optimal_dbscan_labels
from pathlib import Path
import os
import centroid_for_each_strand
import detect_strands_to_dict
import sys
from sklearn.svm import SVC
pdb_path = sys.argv[1]
chain_id = Path(pdb_path).stem.split("_")[1]

pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
chain_id
strands = detect_strands_to_dict.result_dict[pdb_name]['strands']
print(f"Strands: {strands}")
strand_coords = centroid_for_each_strand.parse_pdb_backbone_coords_by_strand(pdb_path, chain_id, strands)
print(f"Strand Coords: {strand_coords}")
labels = optimal_dbscan_labels()

from itertools import combinations
import numpy as np


cluster0 = []
cluster1 = []
def difference_from_both_sheets_from_hyperplane(strands_coords,labels):
    for strand_id, label in zip(sorted(strand_coords.keys()), labels):
        if label == 0:
            cluster0.append(strand_id)
        elif label == 1:
            cluster1.append(strand_id)

    for zero_pair in combinations(cluster0, 2):
        for one_pair in combinations(cluster1, 2):

            X = []
            y = []

            # class 0 residues
            for strand in zero_pair:
                X.extend(strand_coords[strand])
                y.extend([0] * len(strand_coords[strand]))

            # class 1 residues
            for strand in one_pair:
                X.extend(strand_coords[strand])
                y.extend([1] * len(strand_coords[strand]))

            X = np.asarray(X, dtype=float)
            y = np.asarray(y)

            clf = SVC(kernel="linear", C=1e6)
            clf.fit(X, y)
            w_norm = np.linalg.norm(clf.coef_[0])
            distances = clf.decision_function(X) / w_norm
            side0_plane_distance = np.mean(np.abs(distances[y == 0]))
            side1_plane_distance = np.mean(np.abs(distances[y == 1]))
            margin = 2.0 / w_norm
    return side0_plane_distance,side1_plane_distance


def best_fit_plane_for_two_strands(strand_coords, strand_pair):
    """
    Fit best-fit plane to all residue coords from two strands.

    Returns:
        point_on_plane: centroid of points
        normal: unit normal vector to plane
    """
    points = []

    for strand_id in strand_pair:
        points.extend(strand_coords[strand_id]) # extend is like append but is flattened. 

    X = np.asarray(points, dtype=float)

    if X.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a plane.")

    centroid = X.mean(axis=0)
    X_centered = X - centroid

    # SVD: normal is direction of smallest variance
    _, _, vh = np.linalg.svd(X_centered)
    normal = vh[-1]

    normal = normal / np.linalg.norm(normal)

    return centroid, normal
def best_fit_vector_for_two_strands(strand_coords, strand_pair):
    vectors = []

    for strand_id in strand_pair:
        coords = np.asarray(strand_coords[strand_id])

        v = coords[-1] - coords[0]
        v = v / np.linalg.norm(v)

        vectors.append(v)

    v1, v2 = vectors

    # make directions consistent
    best_fit = v1 -  v2
    best_fit /= np.linalg.norm(best_fit)

    return best_fit
def svm_hyperplane_for_four_strands(strand_coords, zero_pair, one_pair, C=1e6):
    """
    Fit linear SVM separating residue coords from:
        zero_pair -> class 0
        one_pair  -> class 1

    Returns:
        point_on_plane: one point on SVM plane
        normal: unit normal vector
        clf: fitted SVM
    """
    X = []
    y = []

    for strand_id in zero_pair:
        X.extend(strand_coords[strand_id])
        y.extend([0] * len(strand_coords[strand_id])) # [0] * len(x) = [0,0,0,0,....] where the length of this array is the len of list x

    
    for strand_id in one_pair:
        X.extend(strand_coords[strand_id])
        y.extend([1] * len(strand_coords[strand_id])) # same for cluster but filled with 1s this will create y to be combinations of zeroes and ones representative of classes

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    clf = SVC(kernel="linear", C=C)
    clf.fit(X, y)

    w = clf.coef_[0]
    b = clf.intercept_[0]

    normal = w / np.linalg.norm(w)

    # Closest point on SVM plane to origin:
    # w.x + b = 0  =>  x0 = -b * w / ||w||^2
    point_on_plane = -b * w / np.dot(w, w)

    return point_on_plane, normal, clf

def angle_between_planes(normal_a, normal_b, degrees=True):
    """
    Angle between two planes = angle between their normal vectors.
    Uses absolute dot product so angle is always in [0, 90].
    """
    normal_a = np.asarray(normal_a, dtype=float)
    normal_b = np.asarray(normal_b, dtype=float)

    normal_a = normal_a / np.linalg.norm(normal_a)
    normal_b = normal_b / np.linalg.norm(normal_b)

    cos_theta = abs(np.dot(normal_a, normal_b))
    #cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle = np.arccos(cos_theta)

    if degrees:
        return np.degrees(angle)

    return angle


def evaluate_beta_sheet_plane_combinations(strand_coords, labels):
    """
    For every combination:
        2 strands from cluster 0
        2 strands from cluster 1

    Compute:
        - best-fit plane for cluster 0 pair
        - best-fit plane for cluster 1 pair
        - SVM hyperplane for all four strands
        - angle between cluster0 plane and SVM plane
        - angle between cluster1 plane and SVM plane
    """
    strand_ids = sorted(strand_coords.keys())

    cluster0 = []
    cluster1 = []

    for strand_id, label in zip(strand_ids, labels):
        if label == 0:
            cluster0.append(strand_id)
        elif label == 1:
            cluster1.append(strand_id)

    results = []

    for zero_pair in combinations(cluster0, 2):
        for one_pair in combinations(cluster1, 2):

            X = []
            y = []

            for strand in zero_pair:
                X.extend(strand_coords[strand])
                y.extend([0] * len(strand_coords[strand]))

            for strand in one_pair:
                X.extend(strand_coords[strand])
                y.extend([1] * len(strand_coords[strand]))

            X = np.asarray(X, dtype=float)
            y = np.asarray(y)

            #_, normal0 = best_fit_plane_for_two_strands(
            #    strand_coords,
            #    zero_pair
            #)
            normal0 = best_fit_vector_for_two_strands(
                strand_coords,
                zero_pair
            )

            normal1 = best_fit_vector_for_two_strands(
                strand_coords,
                one_pair
            )


            #_, normal1 = best_fit_plane_for_two_strands(
            #    strand_coords,
            #    one_pair
            #)

            _, svm_normal, clf = svm_hyperplane_for_four_strands(
                strand_coords,
                zero_pair,
                one_pair
            )
            w_norm = np.linalg.norm(clf.coef_[0])

            distances = clf.decision_function(X) / w_norm

            side0_plane_distance = np.mean(np.abs(distances[y == 0]))
            side1_plane_distance = np.mean(np.abs(distances[y == 1]))

            margin = 2.0 / w_norm

            angle0 = angle_between_planes(normal0, svm_normal)
            angle1 = angle_between_planes(normal1, svm_normal)
            mean_angle = (angle0 + angle1) / 2
            side_diff  = abs(side0_plane_distance-side1_plane_distance)
            normalized_angle = mean_angle/90
            normalized_side_diff = np.exp(-1*side_diff)
               
            result = {
                "zero_pair": zero_pair,
                "one_pair": one_pair,
                "angle_cluster0_to_svm": angle0,
                "angle_cluster1_to_svm": angle1,
                "mean_angle": mean_angle,
                "Distance from Cluster 0":  side0_plane_distance,
                "Distance from Cluster 1": side1_plane_distance,
                "Difference of distance to plane from two sheets":side_diff,
                "Ratio Mean_Angle/Diff_Distance": normalized_angle/normalized_side_diff,
                "Ratio Diff_Distance/Mean_angle": side_diff/mean_angle,
                "Score normalized Diff_Distance*Mean_angle": normalized_side_diff* normalized_angle,
            }
   

            results.append(result)

            print(
                f"0-strands={zero_pair}, "
                f"1-strands={one_pair}, "
                f"angle0={angle0:.8f}, "
                f"angle1={angle1:.8f}, "
                f"mean={result['mean_angle']:.2f}"
                f"Diff of distance to plane from two sheets: {result['Difference of distance to plane from two sheets']}",
                f"Ratio Mean_Angle/Diff_Distance: {result['Ratio Mean_Angle/Diff_Distance']}",
                f"Ratio Diff_Distance/Mean_angle: {result['Ratio Diff_Distance/Mean_angle']}",
                f"Score normalized Diff_Distance x Mean_angle: {result['Score normalized Diff_Distance*Mean_angle']}",
            )

    return results
results = evaluate_beta_sheet_plane_combinations(strand_coords, labels)
