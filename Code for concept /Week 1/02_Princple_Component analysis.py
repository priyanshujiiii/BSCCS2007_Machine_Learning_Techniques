
# Import required libraries
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Perform PCA with 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Print explained variance ratio of each component
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Plot the first two principal components
import matplotlib.pyplot as plt

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Iris dataset - First two principal components")
plt.show()
