import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM with linear kernel
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Train Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Make predictions on the testing data
svm_pred = svm_model.predict(X_test)
logistic_pred = logistic_model.predict(X_test)

# Calculate the accuracy of the models
svm_accuracy = accuracy_score(y_test, svm_pred)
logistic_accuracy = accuracy_score(y_test, logistic_pred)

print("SVM Accuracy:", svm_accuracy)
print("Logistic Regression Accuracy:", logistic_accuracy)

# Plot the decision boundaries
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

fig, sub = plt.subplots(1, 2, figsize=(10, 5))

# Plot SVM decision boundary
Z_svm = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)
sub[0].contourf(xx, yy, Z_svm, cmap=plt.cm.RdYlBu, alpha=0.8)
sub[0].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
sub[0].set_title("SVM Decision Boundary")

# Plot Logistic Regression decision boundary
Z_logistic = logistic_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_logistic = Z_logistic.reshape(xx.shape)
sub[1].contourf(xx, yy, Z_logistic, cmap=plt.cm.RdYlBu, alpha=0.8)
sub[1].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
sub[1].set_title("Logistic Regression Decision Boundary")

plt.show()
