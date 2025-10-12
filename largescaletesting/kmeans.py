import numpy as np
import normalization 
from sklearn.cluster import KMeans
import feature_matrix
import pca_on_features
import color_igtype
X = normalization.degree_features.T
print(f"X shape: {X.shape}")
#X = feature_matrix.X
#X = pca_on_features.X_transform
X = X.T[~np.isnan(X).any(axis=0)]
print(f"X shape: {X.shape}")
X = X.T
def dot_product(X):
    result = []
    for pdb in X:
        unit = np.sum(pdb)
        result.append(unit)
    return np.array(result)

def rmsd(X):
    X = np.array(X)
    matrix = np.zeros((len(X),len(X)))
    for i in range(0,len(X)-1):
        elem = X[i]
        sum_sd = np.zeros(len(X[0]))
        for j in range(1,len(X)-1):
            diff = X[i] - X[j]
            sq_diff = diff*diff
            sum_sd = np.sum(sq_diff)
            rmsd = np.sqrt(sum_sd/len(X))
            matrix[i][j] = rmsd
    matrix = np.array(matrix)
    return matrix

def dot_product(X):
    X = np.array(X)
    matrix = np.zeros((len(X),len(X)))
    for i in range(0,len(X)-1):
        sum_sd = np.zeros(len(X[0]))
        for j in range(1,len(X)-1):
            product = np.dot(X[i],X[j])
            matrix[i][j] = product
    matrix = np.array(matrix)
    return matrix


squared_norms = rmsd(X)
#print(squared_norms)
print(squared_norms.shape)
print(np.min(squared_norms))
print(np.max(squared_norms))


            
            
print(squared_norms.shape)
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(squared_norms.reshape(-1,1))
print("results")
#print(labels[:20])

print(X.dtype, X.shape)
print("Any NaN in X?", np.isnan(X).any())
print("Any Inf in X?", np.isinf(X).any())

sq = np.sum(np.square(X, dtype=np.float64), axis=1)
print("Squared norms stats:", np.min(sq), np.max(sq))
print("Any non-finite in squared norms?", ~np.isfinite(sq).any())
import matplotlib.pyplot as plt
#
#plt.figure(figsize=(8,4))
#plt.hist(sq, bins=30, color="lightgray", edgecolor="black")
#for c in kmeans.cluster_centers_.ravel():
#    plt.axvline(c, color="red", linestyle="--")
#plt.xlabel("Squared norm")
#plt.ylabel("Count")
#plt.title("Distribution of squared norms with cluster centers")
#plt.show()
#plt.figure(figsize=(10, 5))
#plt.scatter(range(len(squared_norms)), squared_norms, 
#           c=labels, cmap="tab10", s=30)
#plt.xlabel("Sample index")
#plt.ylabel("Squared norm")
#plt.title("Scatter plot of squared norms colored by cluster")
#plt.colorbar(label="Cluster")
#plt.show()        
import matplotlib.pyplot as plt
##

#squared_norms = np.einsum('ij,ij->i', X, X)
#
## Step 2: run k-means in 1D with 12 clusters

# --- 3) decision boundaries = midpoints between sorted centers ---
#labels = km.fit_predict(squared_norms)
#print(labels.shape)
#print(labels[:5])
#centers = km.cluster_centers_.ravel()
#rows, cols = np.indices(squared_norms.shape)
#plt.figure(figsize=(12, 6))
#plt.scatter(rows.ravel(), cols.ravel(), c=squared_norms.ravel(), cmap='viridis_r', s=30)
#plt.xlabel("Sample index")
#plt.ylabel("RMSD")
#plt.title("Scatter plot of RMSD with 12 clusters")
#plt.colorbar(label="Cluster")
#
#Step 4: overlay horizontal lines for cluster centers
#for c in centers:
# plt.axhline(c, color="red", linestyle="--", alpha=0.6)

#plt.show()
#input("Press Enter to close the plot.")
#print("Shape of x")
#print(X.shape)
#
def rmsd_vector(X, ref_index=0):
    X = np.array(X)
    ref = X[ref_index]
    diffs = X - ref
    sq_diffs = np.sum(diffs**2, axis=1)
    rmsd = np.sqrt(sq_diffs / X.shape[1])
    return rmsd
km = KMeans(n_clusters=6, n_init=10, algorithm="lloyd", random_state=0)
squared_norms = rmsd_vector(X)
labels = km.fit_predict(squared_norms.reshape(-1, 1))
centers = km.cluster_centers_.ravel()
centers_sorted = np.sort(centers)
boundaries = (centers_sorted[:-1] + centers_sorted[1:]) / 2.0  # length K-1
ig_type_change_indices = color_igtype.indices
plt.figure(figsize=(12,6))
sc = plt.scatter(range(len(squared_norms)), squared_norms, 
              c=squared_norms, cmap="viridis", s=30)
#for b in boundaries:
#    plt.axhline(b, color="black", linestyle="--", linewidth=1, alpha=0.8)
plt.xlabel("Sample index")
plt.ylabel("RMSD")
for i in ig_type_change_indices:                                                                                        
    plt.axvline(X[i, 0], ls="--", lw=1, alpha=0.9)
plt.title("Scatter plot of RMSD (colored by value)")
plt.colorbar(sc, label="RMSD value")
plt.show()


#plt.figure(figsize=(12, 6))
#plt.scatter(np.arange(len(squared_norms)), squared_norms,
#            c=labels, cmap="tab20", s=30)
#for c in centers:

#    plt.axhline(c, ls="--", lw=1, alpha=0.5)   # show cluster centers
#plt.xlabel("Sample index")
#plt.ylabel("RMSD to reference")
#plt.title("RMSD (1D) clustered into 12 groups")
#plt.colorbar(label="Cluster")
#plt.show()
#

for k in range(6):
  idx = np.where(labels == k)[0]
  print(f"Cluster {k}: {len(idx)} samples, indices {idx[:60]}...")


