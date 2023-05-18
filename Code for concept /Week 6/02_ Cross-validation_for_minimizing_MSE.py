import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + np.random.normal(0, 1, 100)

# Set the number of folds for cross-validation
n_folds = 5

# Initialize an array to store the MSE values
mse_scores = []

# Perform cross-validation
kf = KFold(n_splits=n_folds, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Create and fit a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the MSE
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

# Calculate the average MSE across all folds
average_mse = np.mean(mse_scores)

# Print the average MSE
print("Average MSE:", average_mse)

#In this code, we first generate sample data in the same way as before. We then specify the number of folds for cross-validation (n_folds) as 5.
#The code initializes an array (mse_scores) to store the MSE values calculated for each fold. It then performs cross-validation using KFold from the scikit-learn library. For each fold, it splits the data into training and test sets, creates a linear regression model using LinearRegression, fits the model to the training data, and makes predictions on the test data. It calculates the MSE using mean_squared_error from scikit-learn and appends the MSE value to the mse_scores array.
#Finally, the code calculates the average MSE across all folds by taking the mean of the mse_scores array and prints the result.
#By using cross-validation and minimizing the average MSE, you can assess the performance of the linear regression model on unseen data and select the optimal hyperparameters or model configuration.
