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

# Plot the data points, regression line, and errors
plt.scatter(X[:, 1], y, label='Data')
plt.plot(X[:, 1], y_pred, color='red', label='Regression Line')

# Plot the errors
for i in range(len(X)):
    plt.plot([X[i, 1], X[i, 1]], [y[i], y_pred[i]], color='gray', linestyle='--')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Geometric Interpretation of Linear Regression')
plt.legend()
plt.show()


#In this code, we first generate sample data in the same way as before. We then perform linear regression using the ordinary least squares (OLS) method to estimate the intercept and slope coefficients.
#Next, we calculate the predicted values y_pred by combining the intercept, slope, and the input X.
#The code then plots the data points as scatter plots and the regression line as a solid red line. Additionally, it visualizes the errors by drawing dashed gray lines between each data point and its corresponding predicted value.
#The plot provides a geometric interpretation of linear regression, showing how the regression line fits the data points and how the errors are calculated as the vertical distances between the data points and the regression line.
