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

# Make predictions on the test set
y_pred = gnb.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report
print(classification_report(y_test, y_pred, target_names=iris.target_names))

#In this code, we first import the necessary libraries: load_iris from scikit-learn to load the Iris dataset, train_test_split to split the dataset into training and testing sets, GaussianNB to create the Gaussian Naive Bayes classifier, and accuracy_score and classification_report to evaluate the classifier's performance.
#We load the Iris dataset using load_iris() and assign the input features to X and the target variable to y.
#Next, we split the data into training and testing sets using train_test_split(). In this example, we allocate 20% of the data for testing.
#We create an instance of the Gaussian Naive Bayes classifier using GaussianNB().
#We train the classifier on the training data using the fit() method.
#We make predictions on the test set using the predict() method.
#Finally, we calculate and print the accuracy of the predictions using accuracy_score(), and display a classification report using classification_report(), which provides additional metrics such as precision, recall, and F1-score for each class.
#When you run the code, you will see the accuracy and classification report for the Gaussian Naive Bayes classifier on the Iris dataset.
