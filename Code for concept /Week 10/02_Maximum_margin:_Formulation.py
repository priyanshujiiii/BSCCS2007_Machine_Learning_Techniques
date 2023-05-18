import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# Generate random data
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# Create an instance of SVM classifier with linear kernel
svm = SVC(kernel='linear')

# Fit the SVM model on the training data
svm.fit(X, y)

# Get the coefficients and intercept of the separating hyperplane
w = svm.coef_[0]
b = svm.intercept_[0]

# Get the support vectors
support_vectors = svm.support_vectors_

# Calculate the slope and intercept of the decision boundary
slope = -w[0] / w[1]
intercept = -b / w[1]

# Calculate the margin
margin = 2 / np.linalg.norm(w)

# Plot the data points and decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.plot(X[:, 0], slope * X[:, 0] + intercept, 'k-')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='k')
plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Maximum Margin')
plt.text(2, 4, f'Margin = {margin:.2f}', fontsize=12)
plt.show()


#In this code, we first generate random data using the make_blobs function from scikit-learn. The X represents the input features, and y represents the corresponding target labels.

#Then, we create an instance of the SVC (Support Vector Classification) classifier with a linear kernel.

#We fit the SVM model on the training data using the fit method.

#After that, we extract the coefficients and intercept of the separating hyperplane using the coef_ and intercept_ attributes.

#We also retrieve the support vectors using the support_vectors_ attribute.

#Next, we calculate the slope and intercept of the decision boundary using the coefficients.

#We calculate the margin by taking the reciprocal of the Euclidean norm of the coefficients.

#Finally, we plot the data points, decision boundary, and support vectors. The margin is displayed as text on the plot.
#When you run the code, you will see a plot that illustrates the maximum margin solution. The decision boundary is the line that separates the two classes, and the support vectors are the data points closest to the decision boundary. The margin is the distance between the decision boundary and the support vectors, which is maximized in the SVM formulation.
