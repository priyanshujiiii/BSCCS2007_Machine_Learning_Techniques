from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

# Generate synthetic data for regression
X, y = make_regression(n_samples=100, n_features=10, noise=0.5, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit a Ridge regression model
alpha = 1.0  # Regularization parameter
ridge = Ridge(alpha=alpha)
ridge.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ridge.predict(X_test)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Print the MSE
print("Mean Squared Error (MSE):", mse)


#In this code, we first generate synthetic data for regression using the make_regression() function from scikit-learn
#We then split the data into training and test sets using train_test_split().
#Next, we create a Ridge regression model by instantiating the Ridge class from scikit-learn. We specify the regularization parameter alpha to control the amount of regularization applied. A larger alpha value will result in stronger regularization.
#We fit the Ridge model to the training data using the fit() method.
#We then use the trained model to make predictions on the test set using the predict() method.
#Finally, we calculate the Mean Squared Error (MSE) between the predicted values and the true target values using the mean_squared_error() function from scikit-learn.
#By using Ridge regression, we introduce regularization to the linear regression model, which helps to reduce the impact of high-variance coefficients and improve generalization performance.
