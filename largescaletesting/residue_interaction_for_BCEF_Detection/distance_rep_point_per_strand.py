import numpy as np
from pathlib import Path

import os
from three_rep_points_per_strand import get_triplets_relative_to_ref_strand
import three_rep_points_per_strand
import centroid_for_each_strand
import nearest_c_alpha
import detect_strands_to_dict
import optimized_dbscan
import sys
pdb_path = sys.argv[1]
#triplets = three_rep_points_per_strand.triplets
triplets = three_rep_points_per_strand.all_triplets

#pdb_path="../output_pdbs/1YJD_C_3_117.pdb"
pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
chain_id = Path(pdb_path).stem.split("_")[1]
centroids = centroid_for_each_strand.centroid_per_strand_dict(pdb_path,chain_id)
c_alpha_per_strand = nearest_c_alpha.strand_coords_CA
strands = detect_strands_to_dict.result_dict[pdb_name]['strands']


def triplet_distance_profile(triplets, s_ref, s):
    """
    For strand s_ref and strand s, return:
      [
        min distance from c1 of s_ref to [c1,c2,c3] of s,
        min distance from c2 of s_ref to [c1,c2,c3] of s,
        min distance from c3 of s_ref to [c1,c2,c3] of s
      ]

    triplets format:
      {
        1: {'rep_resnum': 5, 'c1': [...], 'c2': [...], 'c3': [...]},
        2: {...},
        ...
      }
    """

    ref_pts = [
        np.array(triplets[s_ref]["c1"], dtype=float),
        np.array(triplets[s_ref]["c2"], dtype=float),
        np.array(triplets[s_ref]["c3"], dtype=float),
    ]

    tgt_pts = [
        np.array(triplets[s]["c1"], dtype=float),
        np.array(triplets[s]["c2"], dtype=float),
        np.array(triplets[s]["c3"], dtype=float),
    ]

    out = []
    i=0
    for p in ref_pts:
        dists = [np.linalg.norm(p - q) for q in tgt_pts]
        #print(f"Dist {i}: {dists}")
        i+=1
        out.append(min(dists))

    return np.array(out)

def triplet_distance_profile(all_triplets, s_ref, s):

    triplets = all_triplets[s_ref]

    ref_pts = [
        np.array(triplets[s_ref][k])
        for k in ["c1", "c2", "c3"]
    ]

    tgt_pts = [
        np.array(triplets[s][k])
        for k in ["c1", "c2", "c3"]
    ]

    return np.array([
        min(np.linalg.norm(p - q) for q in tgt_pts)
        for p in ref_pts
    ])

#print(triplet_distance_profile(triplets,1,2))
#print(triplets)
nearest_neighbor_dict = {}
top_neighbors = {}

for i in sorted(triplets.keys()):
    dists = []

    for j in sorted(triplets.keys()):
        if i == j:
            continue

        d = np.mean(triplet_distance_profile(triplets,s_ref=i,s=j))  # FIXED
        dists.append((j, d))
        print(dists)

    dists.sort(key=lambda x: x[1])
    
    # keep top-2
    top_neighbors[i] = dists[:2]

    # also store top-1
    nearest_neighbor_dict[i] = {
        "strand": dists[0][0],
        "distance": dists[0][1]
    }    
strand_ids = sorted(nearest_neighbor_dict.keys())
n = max(strand_ids)

neighbor_matrix = np.zeros((n, n), dtype=int)

for s_ref, info in nearest_neighbor_dict.items():
    s = info["strand"]
    if s is not None and s != s_ref:
        neighbor_matrix[s_ref - 1, s - 1] = 1
#print(neighbor_matrix)    

strand_ids = sorted(top_neighbors.keys())
n = max(strand_ids)

neighbor_matrix = np.zeros((n, n), dtype=int)

for s_ref, neighbors in top_neighbors.items():
    for (s, _) in neighbors:   # unpack (strand, distance)
        if s != s_ref:
            neighbor_matrix[s_ref - 1, s - 1] = 1

#print(neighbor_matrix)
#print(dists)

def distance_matrix():

    strand_ids = sorted(triplets.keys())
    n = len(strand_ids)

    distance_matrix = np.zeros((n, n))

    for i in strand_ids:
        for j in strand_ids:
            if i == j:
                distance_matrix[i-1, j-1] = 0.0
            else:
                d = np.mean(triplet_distance_profile(triplets,i, j))
                distance_matrix[i-1, j-1] = round(d, 1)
    print(f"Raw Distance Matrix: {distance_matrix}")
    for i in strand_ids:
        for j in strand_ids:
            d = min(distance_matrix[i-1,j-1], distance_matrix[j-1,i-1])
            #d = d/2
            distance_matrix[i-1,j-1] = d
            distance_matrix[j-1,i-1] = d
    return np.array(distance_matrix)
import numpy as np

def kmedoids(D, k=2, max_iter=100, random_state=20):
    np.random.seed(random_state)
    n = D.shape[0] # Num of Rows

    # initialize medoids randomly
    medoids = np.random.choice(n, k, replace=False) # k medoids

    for _ in range(max_iter):
        # assign each point to closest medoid (using row distances)
        labels = np.argmin(D[:, medoids], axis=1)

        new_medoids = np.copy(medoids)

        for i in range(k):
            cluster = np.where(labels == i)[0]

            if len(cluster) == 0:
                continue

            # minimize total outgoing distance within cluster
            costs = np.sum(np.linalg.norm(D[np.ix_(cluster, cluster)], axis=1)**2)
            best = cluster[np.argmin(costs)]
            new_medoids[i] = best

        if np.all(new_medoids == medoids):
            break

        medoids = new_medoids

    return labels, medoids
D = distance_matrix()
print("Symmetric Distance Matrix")
print(D)
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.cluster import AgglomerativeClustering

model = DBSCAN(
    eps=4.7,            # pick your k
    min_samples=2,
    metric='precomputed'
)

labels = model.fit_predict(D)
print(f"forced : {labels}")

eps_values = np.linspace(1,10000000,1000)
min_samples_values = range(2,4)

best = optimized_dbscan.tune_dbscan(D, eps_values, min_samples_values)

print("Best parameters:")
print(f"  eps = {best['eps']}")
print(f"  min_samples = {best['min_samples']}")
print(f"  score = {best['score']:.4f}")
model = DBSCAN(
    eps=best['eps'],            # pick your k
    min_samples=best['min_samples'],
    metric='precomputed'
)
def optimal_dbscan_labels():
    model = DBSCAN(
        eps=best['eps'],
        min_samples = best['min_samples'],
        metric = 'precomputed'
    
    )
    labels = model.fit_predict(D)
    return labels
    
labels = model.fit_predict(D)
print(labels)

