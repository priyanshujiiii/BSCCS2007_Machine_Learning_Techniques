import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Consider only the first two features for simplicity
y = iris.target

# Keep only two classes for binary classification
X = X[y != 2]
y = y[y != 2]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Compute the Gram matrix
gram_matrix = np.dot(X_train, X_train.T)

# Set the hyperparameters
C = 1.0  # Regularization parameter
n_samples = X_train.shape[0]

# Solve the dual optimization problem using Quadratic Programming
from cvxopt import matrix, solvers

P = matrix(np.outer(y_train, y_train) * gram_matrix)
q = matrix(-np.ones(n_samples))
G = matrix(np.vstack((np.eye(n_samples) * -1, np.eye(n_samples))))
h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))
A = matrix(y_train.reshape(1, -1))
b = matrix(np.zeros(1))

solvers.options['show_progress'] = False
solution = solvers.qp(P, q, G, h, A, b)

# Extract the Lagrange multipliers
alphas = np.array(solution['x']).flatten()

# Find the support vectors (non-zero alphas)
support_vectors = X_train[alphas > 1e-5]
support_labels = y_train[alphas > 1e-5]
support_alphas = alphas[alphas > 1e-5]

# Compute the bias term
bias = np.mean(support_labels - np.dot(support_vectors, support_vectors.T) @ (support_labels * support_alphas))

# Make predictions on the testing data
y_pred = np.sign(np.dot(X_test, support_vectors.T) @ (support_labels * support_alphas) + bias)

# Calculate the accuracy of the model
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# Plot the decision boundary
# Create a meshgrid of points spanning the feature space
h = 0.02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Use the trained model to predict the class labels for the meshgrid points
Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], support_vectors.T) @ (support_labels * support_alphas) + bias)
Z = Z.reshape(xx.shape)

# Plot the contour filled with decision regions
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)

# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu

            
