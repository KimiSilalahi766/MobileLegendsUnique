from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Buat data dummy 2D
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.60, random_state=0)

# 2. K-Means
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(X)
kmeans_score = silhouette_score(X, kmeans_labels)

# 3. DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)
dbscan_score = silhouette_score(X, dbscan_labels)

# 4. Plot hasil clustering
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
axes[0].set_title(f'K-Means (Silhouette: {kmeans_score:.2f})')

axes[1].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
axes[1].set_title(f'DBSCAN (Silhouette: {dbscan_score:.2f})')

plt.suptitle('Perbandingan K-Means vs DBSCAN', fontsize=14)
plt.tight_layout()
plt.show()
