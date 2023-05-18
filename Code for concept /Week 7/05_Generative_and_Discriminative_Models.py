from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Generative Model: Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_y_pred = gnb.predict(X_test)
gnb_accuracy = accuracy_score(y_test, gnb_y_pred)
print("Generative Model (Gaussian Naive Bayes) Accuracy:", gnb_accuracy)

# Discriminative Model: Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg_y_pred = logreg.predict(X_test)
logreg_accuracy = accuracy_score(y_test, logreg_y_pred)
print("Discriminative Model (Logistic Regression) Accuracy:", logreg_accuracy)

#In this code, we use the Iris dataset again as an example. We split the dataset into training and testing sets using train_test_split from scikit-learn.
#First, we train a generative model, Gaussian Naive Bayes, using GaussianNB from scikit-learn. The generative model learns the probability distribution of each class and makes predictions based on the likelihood of each class given the input features.
#Next, we fit a discriminative model, Logistic Regression, using LogisticRegression from scikit-learn. The discriminative model learns the decision boundary directly between the classes.
#We then make predictions on the test set for both models using the predict method.
#Finally, we calculate and print the accuracy of both models using accuracy_score from scikit-learn.
#When you run the code, you will see the accuracy of the generative model (Gaussian Naive Bayes) and the discriminative model (Logistic Regression) printed to the console. This demonstrates the difference in performance between the generative and discriminative models on the Iris dataset.
