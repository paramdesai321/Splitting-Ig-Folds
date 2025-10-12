import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import normalization
import pca_on_features
import color_igtype
#X = pca_on_features.X_transform.T
X = normalization.degree_features
X = X[:1050]
X = X.T
print(f"{X.shape}")

#X = X[~np.isnan(X).any(axis=1)]
print("Input: ")
print(f"Shape of PCA input : {X.shape}")
db = DBSCAN(eps=0.8, min_samples=8).fit(X)
labels = db.labels_

def rmsd_vector(X):
    X = np.array(X) 
    sq_diffs = np.sum(X**2, axis=1)
    rmsd = np.sqrt(X / X.shape[1])
    return rmsd
rmsd = rmsd_vector(X)
print(len(rmsd))
print("Cluster labels:", labels)

idx5 = np.where(labels == 1)[0] 
print("Indices with label 1:", idx5)
print("Count:", len(idx5))
ig_type_change_indices = color_igtype.indices
plt.scatter(range(len(rmsd)),rmsd, c=labels, s=24)  
for i in ig_type_change_indices:
    if 0 <= i < len(X):              # safety
        plt.axvline(X.T[i, 0], ls="--", lw=1.0, alpha=0.8)
plt.title("DBSCAN Clustering")
plt.show()

