from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle imbalanced class distribution
X_train_resampled, y_train_resampled = resample(X_train, y_train, stratify=y_train)

# Feature scaling
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Create a Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Train the classifier
gnb.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = gnb.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


#In this code, we address some of the common pitfalls of the Naive Bayes algorithm:
#Imbalanced Class Distribution: To handle imbalanced class distribution, we use the resample() function from scikit-learn's utils module to upsample the minority class (X_train_resampled, y_train_resampled). This helps balance the class distribution in the training data.
#Feature Scaling: We apply feature scaling using StandardScaler from scikit-learn's preprocessing module. We fit the scaler on the resampled training data and transform both the resampled training data and the test data to bring them to a similar scale.
#Gaussian Naive Bayes Classifier: We use the Gaussian Naive Bayes classifier (GaussianNB()) from scikit-learn to train and make predictions.
#Accuracy Calculation: We calculate the accuracy of the predictions using accuracy_score() from scikit-learn's metrics module.
#By incorporating these techniques into the code, we can handle some of the common pitfalls of the Naive Bayes algorithm.
