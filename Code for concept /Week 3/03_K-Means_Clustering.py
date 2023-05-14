from sklearn.cluster import KMeans
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)

# Initialize the KMeans object with 3 clusters
kmeans = KMeans(n_clusters=3)

# Fit the model to the data
kmeans.fit(X)

# Get the cluster labels
labels = kmeans.labels_

# Get the cluster centroids
centroids = kmeans.cluster_centers_
