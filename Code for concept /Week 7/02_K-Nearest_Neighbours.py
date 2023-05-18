import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Generate synthetic data for classification
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Create a meshgrid to plot the decision boundary
h = 0.02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Make predictions on the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and the data points
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Nearest Neighbors (KNN) Classifier')
plt.show()


#In this updated code, after fitting the KNN classifier, we create a meshgrid using np.meshgrid to define the coordinates for plotting the decision boundary. The meshgrid spans the range of feature values in the dataset with a step size of h.
#We then make predictions on the meshgrid using the trained KNN classifier to obtain the class labels for each point in the meshgrid.
#Finally, we plot the decision boundary and the data points using plt.contourf and plt.scatter from Matplotlib, respectively. The decision boundary is filled with color, indicating the decision regions of the classifier, and the data points are color-coded based on their true class labels.
#When you run the code, you will see a plot showing the decision boundary and the data points. The decision boundary separates the different classes in the dataset, visualizing how the KNN classifier partitions the feature space.
#Note that in this example, we assume a 2-dimensional feature space (n_features=2) for simplicity. If your dataset has more than 2 features, you may need to modify the code accordingly to visualize the decision boundary effectively.
