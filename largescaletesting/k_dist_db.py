import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pca_on_features
X = pca_on_features.X_pca.T   # your PCA result
print("Input from PCA",X.shape)
print(X.shape)
k = 8                             # typically = min_samples

# Fit kNN and get distances to the k-th nearest neighbor (exclude self → ask for k+1)
nbrs = NearestNeighbors(n_neighbors=k+1, metric='minkowski', p=2, algorithm='auto')
nbrs.fit(X)
distances, _ = nbrs.kneighbors(X)
print(distances)
# k-th neighbor distance for each point (index k because 0 is the point itself)
kth = distances[:, k]

# Sort and plot
kth_sorted = np.sort(kth)
print(kth_sorted)
plt.plot(kth_sorted, marker='.', linestyle='none')
plt.ylabel(f"Distance to {k}-NN")
plt.xlabel("Points (sorted)")
plt.title(f"{k}-distance plot (for DBSCAN eps)")
plt.show()

# (optional) show a few candidate eps values by quantile
qs = [85, 90, 95, 98]
vals = np.percentile(kth_sorted, qs)
print({f"{q}%": v for q, v in zip(qs, vals)})

# (optional) overlay one guess on the plot
eps_guess = np.percentile(kth_sorted, 95)
plt.plot(kth_sorted, marker='.', linestyle='none')
plt.axhline(eps_guess, linestyle='--')
plt.title(f"{k}-distance with eps≈{eps_guess:.3f}")
plt.show()

# (then try DBSCAN with that eps)
# from sklearn.cluster import DBSCAN
# db = DBSCAN(eps=eps_guess, min_samples=k).fit(X)

