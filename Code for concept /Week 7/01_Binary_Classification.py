import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Generate synthetic data for binary classification
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

#In this code, we first generate synthetic data for binary classification using the make_classification function from scikit-learn. This function allows us to create a dataset with specified characteristics, such as the number of samples, number of features, informative features, and number of classes.
#Next, we split the data into training and testing sets using train_test_split from scikit-learn. This is a common practice to assess the performance of the classifier on unseen data.
#We then fit a logistic regression model using LogisticRegression() from scikit-learn.
#After training the model, we predict the class labels for the test set using the predict method.
#Finally, we evaluate the performance of the model by calculating various evaluation metrics such as accuracy, precision, recall, and F1-score using the appropriate functions from scikit-learn.
#The evaluation metrics provide information about the model's classification performance on the binary classification task.
#When you run the code, you will see the values of accuracy, precision, recall, and F1-score printed to the console, indicating the performance of the logistic regression model on the binary classification task.
