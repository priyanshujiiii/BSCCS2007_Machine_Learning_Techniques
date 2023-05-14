import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Generate random data
X = np.random.rand(50, 2)

# Perform hierarchical clustering
Z = linkage(X, method='ward')

# Plot the dendrogram
fig = plt.figure(figsize=(10, 5))
dn = dendrogram(Z)
plt.show()
