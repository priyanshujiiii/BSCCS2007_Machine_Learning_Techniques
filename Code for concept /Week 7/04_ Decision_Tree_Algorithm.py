from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Decision Tree classifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# Predict on the test set
y_pred = tree.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


#In this code, we first load the Iris dataset using load_iris from scikit-learn. The dataset contains samples of iris flowers with their corresponding features and target classes.
#Next, we split the data into training and testing sets using train_test_split from scikit-learn. This is a common practice to assess the performance of the classifier on unseen data. The dataset is split into 80% for training and 20% for testing.
#We then create a Decision Tree classifier using DecisionTreeClassifier from scikit-learn.
#After creating the classifier, we fit it to the training data using the fit method.
#Next, we predict the class labels for the test set using the predict method.
#Finally, we calculate the accuracy of the model's predictions by comparing the predicted labels to the true labels using accuracy_score from scikit-learn.
#When you run the code, you will see the accuracy of the Decision Tree classifier printed to the console, indicating the performance of the algorithm on the Iris dataset.
