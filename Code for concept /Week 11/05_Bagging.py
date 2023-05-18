import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic moon-shaped data
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an ensemble of decision trees using Bagging
bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)

# Fit the Bagging ensemble on the training data
bagging.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = bagging.predict(X_test)

# Calculate the accuracy of the ensemble model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot the decision boundary of the ensemble model
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = bagging.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Bagging Decision Boundary')
plt.show()


#In this code, we generate synthetic moon-shaped data using the make_moons function from sklearn.datasets.
#We split the data into training and testing sets using the train_test_split function.
#We create an ensemble of decision trees using Bagging by initializing a BaggingClassifier with base_estimator=DecisionTreeClassifier() and setting the number of estimators with n_estimators=10.
#We fit the Bagging ensemble on the training data using the fit method.
#We make predictions on the testing data using the predict method.
#We calculate the accuracy of the ensemble model using the accuracy_score function.
#Finally, we plot the decision boundary of the ensemble model by creating a meshgrid of points, using the ensemble model to predict the class labels for those points, and plotting the contour filled with decision regions. We also plot the training data points.
#Note: The code assumes a binary classification task, where the target variable y has two classes (0 and 1). If you have a different number of classes or a different dataset, you may need to modify the code accordingly.






#Regenerate response
