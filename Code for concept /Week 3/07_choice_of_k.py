import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2)

# Define a range of k values to evaluate
k_values = range(1, 11)

# Calculate the sum of squared distances for each k
wss_values = []
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    wss_values.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(k_values, wss_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WSS)')
plt.title('Elbow Curve')
plt.show()

#In this code, we generate a random 2D dataset of 300 points. We define a range of k values (1 to 10 in this case) to evaluate. For each k value, we initialize a KMeans object, fit it to the data, and calculate the within-cluster sum of squares (WSS) using the inertia_ attribute. We store the WSS values for each k. Finally, we plot the WSS values against the corresponding k values to create the elbow curve. The "elbow" in the curve represents the optimal k value where the improvement in WSS begins to level off.
