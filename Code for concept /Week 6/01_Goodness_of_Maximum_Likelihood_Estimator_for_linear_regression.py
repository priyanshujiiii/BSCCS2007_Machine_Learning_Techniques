import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y_true = 2 * X + np.random.normal(0, 1, 100)

# Perform linear regression using OLS
X = X.reshape(-1, 1)  # Reshape X to a column vector
X = np.concatenate([np.ones_like(X), X], axis=1)  # Add a column of ones for the intercept term
coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_true)

# Extract the estimated coefficients
intercept = coefficients[0]
slope = coefficients[1]

# Calculate the predicted values
y_pred = intercept + slope * X[:, 1]

# Calculate the residuals
residuals = y_true - y_pred

# Calculate the Mean Squared Error (MSE)
mse = np.mean(residuals**2)

# Calculate the coefficient of determination (R-squared)
ssr = np.sum((y_pred - np.mean(y_true))**2)  # Regression sum of squares
sst = np.sum((y_true - np.mean(y_true))**2)  # Total sum of squares
r_squared = 1 - ssr / sst

# Print the metrics
print("Mean Squared Error (MSE):", mse)
print("Coefficient of Determination (R-squared):", r_squared)

# Plot the data points and the regression line
plt.scatter(X[:, 1], y_true, label='Data')
plt.plot(X[:, 1], y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Goodness of Maximum Likelihood Estimator')
plt.legend()
plt.show()


#In this code, we first generate sample data in the same way as before. We then perform linear regression using the ordinary least squares (OLS) method to estimate the intercept and slope coefficients.
#Next, we calculate the predicted values (y_pred) by combining the intercept, slope, and the input X. We also calculate the residuals by subtracting the predicted values from the actual target values (y_true - y_pred).
#The code then calculates the Mean Squared Error (MSE) as the average of the squared residuals and the coefficient of determination (R-squared) as the proportion of the total variation in the target variable explained by the regression model.
#Finally, the code prints the calculated metrics and plots the data points as scatter plots and the regression line as a solid red line.
#By examining the MSE and R-squared, you can assess the goodness of the Maximum Likelihood Estimator for linear regression. Lower MSE values indicate better model fit, while higher R-squared values suggest a higher proportion of the target variable's variance being explained by the model.
