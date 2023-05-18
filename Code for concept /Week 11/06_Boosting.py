import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an AdaBoost classifier using decision stumps as weak learners
adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)

# Fit the AdaBoost classifier on the training data
adaboost.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = adaboost.predict(X_test)

# Calculate the accuracy of the AdaBoost classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot the decision boundary of the AdaBoost classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = adaboost.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('AdaBoost Decision Boundary')
plt.show()


#In this code, we generate synthetic data using the make_classification function from sklearn.datasets.
#We split the data into training and testing sets using the train_test_split function.
#We create an AdaBoost classifier using decision stumps (trees with maximum depth 1) as weak learners by initializing an AdaBoostClassifier with base_estimator=DecisionTreeClassifier(max_depth=1) and setting the number of estimators with n_estimators=50.
#We fit the AdaBoost classifier on the training data using the fit method.
#We make predictions on the testing data using the predict method.
#We calculate the accuracy of the AdaBoost classifier using the accuracy_score function.
#Finally, we plot the decision boundary of the AdaBoost classifier by creating a meshgrid of points, using the classifier to predict the class labels for those points, and plotting the contour filled with decision regions. We also plot the training data points.
#Note: The code assumes a binary classification task, where the target variable y has two classes (0 and 1). If you have a different number of classes or a different dataset, you may need to modify the code accordingly.
