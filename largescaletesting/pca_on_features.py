import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import feature_matrix 
import normalization
#import kmeans 
X = normalization.degree_features
X = X
X = X[~np.isnan(X).any(axis=1)]
#print(np.where(np.isnan(X) == True))
#print(f"Shape of input for PCA")
#print(X.shape)
#X = X.T   # now: 1052 samples, 24 features
#print(X.shape)

# Standardize features
scaler = MinMaxScaler((-1,1))
X_scaled = scaler.fit_transform(X)
X_scaled  = X_scaled.astype(float)

#print(f"scaled: {X_scaled.tolist()}")
#print(np.min(X_scaled))
#print(np.max(X_scaled))

#print(f"scaled shape: {X_scaled.shape}")
# Run PCA
pca = PCA(n_components=10)   # 2 components for visualization
X_transform = pca.fit_transform(X_scaled)
#print(X_transform.shape)
#X_pca = X_pca.astype(float)
X_pca  = pca.components_
#print(f"{X_pca:}")
#print(f"shape of comps: {X_pca.shape}")
#print("Original shape:", X.shape)
#print("Transformed shape:", X_pca.shape)
#print("Explained variance ratio:", pca.explained_variance_ratio_)
#print("Total var:",np.sum(pca.explained_variance_ratio_))

#rmsd = kmeans.rmsd_vector(X_pca)
#plt.figure(figsize=(12,6))
#sc = plt.scatter(range(len(rmsd)), rmsd, 


# Scatter plot of the 2 PCs
#plt.figure(figsize=(8,6))
#plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.6)
#plt.xlabel("PC1")
#plt.ylabel("PC2")
#plt.title("PCA projection of features")
#plt.show()
#
