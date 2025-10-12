import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pca_on_features
import normalization
# Use your PCA output (as you already have)

X = normalization.degree_features.T
X = X[~np.isnan(X).any(axis=1)]
print("Input from PCA", X.shape)

K_MIN, K_MAX = 2, 10

# Compute neighbors once up to the largest k (add 1 to skip self at index 0)
nbrs = NearestNeighbors(n_neighbors=K_MAX + 1)
nbrs.fit(X)
distances, _ = nbrs.kneighbors(X)

plt.figure(figsize=(8, 5))
for k in range(K_MIN, K_MAX + 1):
    kth_sorted = np.sort(distances[:, k])   # distance to the k-th NN, sorted
    plt.plot(kth_sorted, '.', ms=2, linestyle='none', label=f'k={k}', alpha=0.9)

plt.xlabel("Points (sorted within each k)")
plt.ylabel("Distance to k-th nearest neighbor")
plt.title("k-distance plots (k = 2…10) for DBSCAN eps selection")
plt.legend(title="k", ncol=3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

