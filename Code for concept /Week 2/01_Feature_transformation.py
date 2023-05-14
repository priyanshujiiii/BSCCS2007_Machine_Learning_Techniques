# Import required libraries
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load iris dataset
iris = load_iris()
X = iris.data

# Perform feature scaling using standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA with 2 components
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X_scaled)

# Print explained variance ratio of each component
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Print transformed features
print("Transformed features:")
for i in range(X_transformed.shape[1]):
    print(f"PC{i+1}: {X_transformed[:, i]}")

    
#This code loads the iris dataset, performs feature scaling using standardization, performs PCA with 2 components, and prints the transformed features. The StandardScaler class from scikit-learn is used to perform standardization, and the PCA class is used to perform the PCA. The fit_transform method is used to perform both standardization and PCA on the data.
#You can modify this code to work with other datasets and use a different feature transformation technique as needed.
