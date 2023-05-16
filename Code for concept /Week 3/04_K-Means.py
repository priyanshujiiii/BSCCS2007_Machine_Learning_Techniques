import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2)
X[:100, :] += 5
X[100:200, :] -= 5
X[200:, :] += np.array([5, -5])

# Initialize the KMeans object with 3 clusters
kmeans = KMeans(n_clusters=3)

# Fit the model to the data
kmeans.fit(X)

# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the data and centroids
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.show()
