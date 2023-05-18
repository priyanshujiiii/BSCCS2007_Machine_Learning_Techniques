import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + np.random.normal(0, 1, 100)

# Perform linear regression using OLS
X = X.reshape(-1, 1)  # Reshape X to a column vector
X = np.concatenate([np.ones_like(X), X], axis=1)  # Add a column of ones for the intercept term
coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Extract the estimated coefficients
intercept = coefficients[0]
slope = coefficients[1]

# Calculate the predicted values
y_pred = intercept + slope * X[:, 1]

# Calculate the residuals
residuals = y - y_pred

# Plot the data points
plt.scatter(X[:, 1], y, label='Data')

# Plot the regression line
plt.plot(X[:, 1], y_pred, color='red', label='Regression Line')

# Plot the residuals
plt.scatter(X[:, 1], residuals, color='green', label='Residuals')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Probabilistic View of Linear Regression')
plt.legend()
plt.show()

#In this code, we first generate sample data in the same way as before. We then perform linear regression using the ordinary least squares (OLS) method to estimate the intercept and slope coefficients.
#Next, we calculate the predicted values (y_pred) by combining the intercept, slope, and the input X. We also calculate the residuals by subtracting the predicted values from the actual target values (y - y_pred).
#The code plots the data points as scatter plots, the regression line as a solid red line, and the residuals as green points. The residuals represent the differences between the actual target values and the predicted values.
#By considering the residuals, we can assess how well the linear regression model fits the data. In the probabilistic view, the residuals are assumed to follow a normal distribution around the regression line. Analyzing the residuals can help identify any deviations from this assumption and provide insights into model performance and potential improvements.
#Note that the probabilistic view of linear regression allows for the calculation of confidence intervals and prediction intervals to quantify uncertainty around the regression estimates.
