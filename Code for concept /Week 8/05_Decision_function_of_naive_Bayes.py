from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Train the classifier
gnb.fit(X_train, y_train)

# Get the predicted probabilities for each class
y_prob = gnb.predict_proba(X_test)

# Set a decision threshold
decision_threshold = 0.5

# Make predictions based on the decision threshold
y_pred = (y_prob[:, 1] >= decision_threshold).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report
print(classification_report(y_test, y_pred, target_names=iris.target_names))

#In this code, we first import the necessary libraries and modules. Then, we load the Iris dataset and split it into training and testing sets.
#Next, we create an instance of the Gaussian Naive Bayes classifier (GaussianNB()), and train it on the training data using the fit() method.
#We then use the predict_proba() method to obtain the predicted probabilities for each class on the testing data. The resulting y_prob array will have shape (n_samples, n_classes).
#We set a decision threshold, which is a value between 0 and 1. In this example, we set it to 0.5.
#Based on the decision threshold, we make predictions by comparing the predicted probabilities for the positive class (index 1 in y_prob) with the threshold. If the probability is greater than or equal to the threshold, we assign the positive class; otherwise, we assign the negative class.
#We calculate the accuracy of the predictions using the accuracy_score() function.
#Finally, we display a classification report using the classification_report() function, which provides additional metrics such as precision, recall, and F1-score for each class.
#Please note that the choice of the decision threshold may depend on the specific problem and the desired balance between precision and recall. Adjusting the threshold can lead to different trade-offs between false positives and false negatives.
