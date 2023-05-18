import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

# Create an instance of SVM classifier with a soft margin (C parameter)
svm = SVC(kernel='linear', C=1.0)

# Fit the SVM model on the training data
svm.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = svm.predict(X_test)

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
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the contour filled with decision regions
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)

# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Soft Margin SVM Decision Boundary')
plt.show()
#In this code, we first load the iris dataset and consider only the first two features for simplicity. We keep only two classes for binary classification.
#We split the dataset into training and testing sets using the train_test_split function. We also perform feature scaling using the StandardScaler.
#Next, we create an instance of the SVC (Support Vector Classification) classifier with a linear kernel and set the C parameter to control the soft margin. The C parameter determines the trade-off between allowing misclassifications and having a wider margin.
#We fit the SVM model on the training data using the fit method.
#We make predictions on the testing data using the predict method.
#We calculate the accuracy of the model by comparing the predicted labels with the actual labels.
#Finally, we plot the decision boundary by creating a meshgrid of points spanning the feature space, using the trained model to predict the class labels for those points, and plotting the contour filled with decision regions. We also plot the training points.




