import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

import optimized_dbscan
import three_rep_points_per_strand


def triplet_distance_profile(all_triplets, s_ref, s):
    """
    Compute the directed triplet-distance profile from s_ref to s.

    Parameters
    ----------
    all_triplets : dict
        Format:

        {
            reference_strand: {
                target_strand: {
                    "rep_resnum": int,
                    "c1": [x, y, z],
                    "c2": [x, y, z],
                    "c3": [x, y, z],
                }
            }
        }

    s_ref : int
        Reference strand index.

    s : int
        Target strand index.

    Returns
    -------
    np.ndarray
        Three distances:

        [
            min distance from c1_ref to target triplet,
            min distance from c2_ref to target triplet,
            min distance from c3_ref to target triplet,
        ]
    """
    triplets = all_triplets[s_ref]

    ref_points = [
        np.asarray(triplets[s_ref][key], dtype=float)
        for key in ("c1", "c2", "c3")
    ]

    target_points = [
        np.asarray(triplets[s][key], dtype=float)
        for key in ("c1", "c2", "c3")
    ]

    return np.asarray(
        [
            min(
                np.linalg.norm(ref_point - target_point)
                for target_point in target_points
            )
            for ref_point in ref_points
        ],
        dtype=float,
    )


def get_neighbor_information(all_triplets):
    """
    Compute nearest and top-two neighboring strands.

    Returns
    -------
    tuple
        nearest_neighbor_dict, top_neighbors
    """
    nearest_neighbor_dict = {}
    top_neighbors = {}

    strand_ids = sorted(all_triplets.keys())

    for s_ref in strand_ids:
        distances = []

        for target_strand in strand_ids:
            if s_ref == target_strand:
                continue

            distance = float(
                np.mean(
                    triplet_distance_profile(
                        all_triplets,
                        s_ref=s_ref,
                        s=target_strand,
                    )
                )
            )

            distances.append(
                (target_strand, distance)
            )

        distances.sort(key=lambda item: item[1])

        top_neighbors[s_ref] = distances[:2]

        if distances:
            nearest_neighbor_dict[s_ref] = {
                "strand": distances[0][0],
                "distance": distances[0][1],
            }
        else:
            nearest_neighbor_dict[s_ref] = {
                "strand": None,
                "distance": None,
            }

    return nearest_neighbor_dict, top_neighbors


def build_neighbor_matrix(neighbor_information, strand_ids):
    """
    Build a directed binary neighbor matrix.

    neighbor_information may contain either:

      nearest-neighbor entries:
          {
              strand: {
                  "strand": neighbor,
                  "distance": value
              }
          }

    or top-neighbor entries:
          {
              strand: [
                  (neighbor, distance),
                  ...
              ]
          }
    """
    strand_ids = sorted(strand_ids)

    if not strand_ids:
        return np.empty((0, 0), dtype=int)

    n = max(strand_ids)
    matrix = np.zeros((n, n), dtype=int)

    for s_ref, information in neighbor_information.items():
        if isinstance(information, dict):
            neighbor = information.get("strand")

            if neighbor is not None and neighbor != s_ref:
                matrix[s_ref - 1, neighbor - 1] = 1

        else:
            for neighbor, _ in information:
                if neighbor != s_ref:
                    matrix[s_ref - 1, neighbor - 1] = 1

    return matrix


def distance_matrix(all_triplets):
    """
    Build the directed strand-distance matrix.

    The value D[i, j] is the mean of the three directed triplet distances
    from strand i to strand j.

    Values are rounded to one decimal place, matching the original code.

    Parameters
    ----------
    all_triplets : dict
        Output of get_all_triplets_for_pdb().

    Returns
    -------
    np.ndarray
        Directed distance matrix.
    """
    strand_ids = sorted(all_triplets.keys())
    n = len(strand_ids)

    matrix = np.zeros((n, n), dtype=float)

    for s_ref in strand_ids:
        for target_strand in strand_ids:
            if s_ref == target_strand:
                matrix[s_ref - 1, target_strand - 1] = 0.0
                continue

            distance = np.mean(
                triplet_distance_profile(
                    all_triplets,
                    s_ref=s_ref,
                    s=target_strand,
                )
            )

            matrix[s_ref - 1, target_strand - 1] = round(
                float(distance),
                1,
            )

    return matrix


def get_distance_matrix_for_pdb(
    pdb_path,
    chain_id=None,
    min_len=3,
):
    """
    Build the strand-distance matrix for one PDB file.

    Parameters
    ----------
    pdb_path : str or Path
        Original PDB file.

    chain_id : str, optional
        Chain ID. If omitted, inferred from a filename such as:
            6UDJ_E_3_107.pdb

    min_len : int
        Minimum DSSP strand length.

    Returns
    -------
    np.ndarray
        Directed strand-distance matrix.
    """
    all_triplets = (
        three_rep_points_per_strand.get_all_triplets_for_pdb(
            pdb_path=pdb_path,
            chain_id=chain_id,
            min_len=min_len,
        )
    )

    return distance_matrix(all_triplets)


def get_all_distance_data_for_pdb(
    pdb_path,
    chain_id=None,
    min_len=3,
):
    """
    Compute all triplet, neighbor, and matrix outputs for one PDB.

    Returns
    -------
    dict
        {
            "all_triplets": ...,
            "nearest_neighbors": ...,
            "top_neighbors": ...,
            "nearest_neighbor_matrix": ...,
            "top_neighbor_matrix": ...,
            "distance_matrix": ...
        }
    """
    all_triplets = (
        three_rep_points_per_strand.get_all_triplets_for_pdb(
            pdb_path=pdb_path,
            chain_id=chain_id,
            min_len=min_len,
        )
    )

    nearest_neighbors, top_neighbors = get_neighbor_information(
        all_triplets
    )

    strand_ids = sorted(all_triplets.keys())

    return {
        "all_triplets": all_triplets,
        "nearest_neighbors": nearest_neighbors,
        "top_neighbors": top_neighbors,
        "nearest_neighbor_matrix": build_neighbor_matrix(
            nearest_neighbors,
            strand_ids,
        ),
        "top_neighbor_matrix": build_neighbor_matrix(
            top_neighbors,
            strand_ids,
        ),
        "distance_matrix": distance_matrix(all_triplets),
    }


def kmedoids(
    distance_matrix_value,
    k=2,
    max_iter=100,
    random_state=20,
):
    """
    Retained from the original implementation.
    """
    np.random.seed(random_state)

    n = distance_matrix_value.shape[0]

    medoids = np.random.choice(
        n,
        k,
        replace=False,
    )

    for _ in range(max_iter):
        labels = np.argmin(
            distance_matrix_value[:, medoids],
            axis=1,
        )

        new_medoids = np.copy(medoids)

        for cluster_index in range(k):
            cluster = np.where(
                labels == cluster_index
            )[0]

            if len(cluster) == 0:
                continue

            cluster_distances = distance_matrix_value[
                np.ix_(cluster, cluster)
            ]

            costs = np.sum(
                np.linalg.norm(
                    cluster_distances,
                    axis=1,
                ) ** 2
            )

            best = cluster[np.argmin(costs)]
            new_medoids[cluster_index] = best

        if np.all(new_medoids == medoids):
            break

        medoids = new_medoids

    return labels, medoids


def fixed_dbscan_labels(
    distance_matrix_value,
    eps=4.7,
    min_samples=2,
):
    """
    Run DBSCAN using explicitly supplied parameters.
    """
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed",
    )

    return model.fit_predict(distance_matrix_value)


def tune_dbscan_parameters(
    distance_matrix_value,
    eps_values=None,
    min_samples_values=None,
):
    """
    Tune DBSCAN parameters using optimized_dbscan.tune_dbscan().
    """
    if eps_values is None:
        eps_values = np.linspace(1, 10, 10)

    if min_samples_values is None:
        min_samples_values = range(2, 4)

    return optimized_dbscan.tune_dbscan(
        distance_matrix_value,
        eps_values,
        min_samples_values,
    )


def optimal_dbscan_labels(
    distance_matrix_value,
    eps_values=None,
    min_samples_values=None,
):
    """
    Tune DBSCAN parameters and return labels plus the best settings.
    """
    best = tune_dbscan_parameters(
        distance_matrix_value,
        eps_values=eps_values,
        min_samples_values=min_samples_values,
    )

    model = DBSCAN(
        eps=best["eps"],
        min_samples=best["min_samples"],
        metric="precomputed",
    )

    labels = model.fit_predict(
        distance_matrix_value
    )

    return labels, best


def infer_chain_id(pdb_path):
    """
    Infer the chain ID from filenames such as:
        6UDJ_E_3_107.pdb
    """
    parts = Path(pdb_path).stem.split("_")

    if len(parts) < 2:
        raise ValueError(
            "Could not infer chain ID from filename "
            f"{Path(pdb_path).name!r}. "
            "Pass the chain explicitly."
        )

    return parts[1]


def main():
    if len(sys.argv) not in {2, 3}:
        print(
            "Usage: python distance_rep_point_per_strand.py "
            "<pdb_file> [chain_id]"
        )
        sys.exit(1)

    pdb_path = sys.argv[1]
    chain_id = (
        sys.argv[2]
        if len(sys.argv) == 3
        else infer_chain_id(pdb_path)
    )

    results = get_all_distance_data_for_pdb(
        pdb_path=pdb_path,
        chain_id=chain_id,
        min_len=3,
    )

    matrix = results["distance_matrix"]

    print("Raw Distance Matrix:")
    print(matrix)

    fixed_labels = fixed_dbscan_labels(
        matrix,
        eps=4.7,
        min_samples=2,
    )

    print(f"Forced DBSCAN labels: {fixed_labels}")

    optimal_labels, best = optimal_dbscan_labels(
        matrix
    )

    print("Best parameters:")
    print(f"  eps = {best['eps']}")
    print(
        "  min_samples = "
        f"{best['min_samples']}"
    )
    print(f"  score = {best['score']:.4f}")
    print(f"Optimal DBSCAN labels: {optimal_labels}")


if __name__ == "__main__":
    main()
