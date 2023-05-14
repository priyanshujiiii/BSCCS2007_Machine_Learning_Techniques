# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define SVM classifier with linear kernel
clf_linear = SVC(kernel='linear')
clf_linear.fit(X_train, y_train)
y_pred_linear = clf_linear.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print("Accuracy with linear kernel:", accuracy_linear)

# Define SVM classifier with RBF kernel
clf_rbf = SVC(kernel='rbf', gamma=0.1)
clf_rbf.fit(X_train, y_train)
y_pred_rbf = clf_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print("Accuracy with RBF kernel:", accuracy_rbf)

# Plot decision boundary for SVM classifier with RBF kernel
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = clf_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('SVM classification with RBF kernel')
plt.show()

# This code loads the iris dataset, splits it into training and testing sets, defines SVM classifiers with linear and RBF kernels, fits them on the training set, and computes accuracy scores on the testing set. The SVC class from scikit-learn library is used to define the SVM classifiers with different kernel functions. The kernel parameter is used to specify the kernel function and the gamma parameter is used to specify the gamma value for RBF kernel.
# Additionally, the code plots the decision boundary for SVM classifier with RBF kernel using the contourf function from matplotlib library.
#You can modify this code to work with other datasets, use a different kernel function, and use a different set of parameters for the kernel function as needed.
