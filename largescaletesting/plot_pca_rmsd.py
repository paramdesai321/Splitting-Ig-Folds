import pca_on_features
import kmeans
import matplotlib.pyplot as plt 


x = pca_on_features.X_transform.T
print("Input Shape:{x.shape}")
squared_norms = kmeans.rmsd_vector(x)
plt.figure(figsize=(12,6))
sc = plt.scatter(range(len(squared_norms)), squared_norms, 
              c=squared_norms, cmap="viridis", s=30)
