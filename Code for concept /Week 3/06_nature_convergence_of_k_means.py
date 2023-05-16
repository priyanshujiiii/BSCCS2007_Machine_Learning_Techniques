import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)

# Initialize the KMeans object with 3 clusters
kmeans = KMeans(n_clusters=3)

# Fit the model to the data
kmeans.fit(X)

# Get the cluster labels and centroids after convergence
final_labels = kmeans.labels_
final_centroids = kmeans.cluster_centers_

# Perform additional iterations to observe the intermediate steps of convergence
max_iterations = 10
intermediate_centroids = []
intermediate_labels = []

for i in range(max_iterations):
    # Fit the model for each iteration
    kmeans.fit(X)

    # Store the intermediate centroids and labels
    intermediate_centroids.append(kmeans.cluster_centers_)
    intermediate_labels.append(kmeans.labels_)

# Plot the data points and the intermediate centroids
plt.figure(figsize=(10, 4))

# Plot the initial data points
plt.subplot(1, max_iterations + 1, 1)
plt.scatter(X[:, 0], X[:, 1])
plt.title('Initial Data')

# Plot the intermediate steps
for i in range(max_iterations):
    plt.subplot(1, max_iterations + 1, i + 2)
    plt.scatter(X[:, 0], X[:, 1], c=intermediate_labels[i])
    plt.scatter(intermediate_centroids[i][:, 0], intermediate_centroids[i][:, 1], marker='x', s=100, linewidths=3, color='r')
    plt.title(f'Iteration {i+1}')

# Plot the final converged clusters
plt.subplot(1, max_iterations + 1, max_iterations + 2)
plt.scatter(X[:, 0], X[:, 1], c=final_labels)
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], marker='x', s=100, linewidths=3, color='r')
plt.title('Final Converged Clusters')

plt.tight_layout()
plt.show()

#In this code, we first generate a random 2D dataset of 100 points using NumPy. Then, we initialize a KMeans object with 3 clusters and fit it to the data to obtain the final converged clusters. We then perform additional iterations to observe the intermediate steps of convergence. The code stores the intermediate centroids and labels for each iteration. Finally, we plot the initial data points, intermediate steps, and final converged clusters to visualize the nature of convergence in K-means clustering.
