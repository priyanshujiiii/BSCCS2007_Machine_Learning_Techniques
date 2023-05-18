import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of logistic regression classifier
lr = LogisticRegression()

# Fit the model on the training data
lr.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = lr.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


#In this code, we first generate synthetic data using the make_classification function from scikit-learn. The X represents the input features, and y represents the corresponding target labels.
#Then, we split the data into training and testing sets using the train_test_split function.
#Next, we create an instance of the logistic regression classifier using LogisticRegression().
#We fit the model on the training data using the fit method of the logistic regression object.
#After that, we make predictions on the testing data using the predict method.
#Finally, we calculate the accuracy of the model by comparing the predicted labels with the true labels using the accuracy_score function.
#When you run the code, you will see the accuracy of the logistic regression model on the testing data. The logistic regression algorithm learns the relationship between the input features and the target labels, and then predicts the labels for new unseen data based on that learned relationship.
