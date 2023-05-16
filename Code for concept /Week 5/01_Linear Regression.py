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

# Print the estimated coefficients
print("Intercept:", intercept)
print("Slope:", slope)

# Plot the data and the linear regression line
plt.scatter(X[:, 1], y, label='Data')
plt.plot(X[:, 1], X.dot(coefficients), color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()


#In this code, we first generate sample data by defining a linear relationship between X and y. The X values are equally spaced points between 0 and 10, and the y values are calculated as 2 * X plus some random noise.
#Next, we perform linear regression using the OLS method. We add a column of ones to X to account for the intercept term. The coefficients of the linear regression are calculated using the closed-form solution coefficients = (X^T * X)^(-1) * X^T * y.
#We then extract the estimated intercept and slope coefficients from the coefficients array.
#Finally, we plot the data points and the linear regression line using Matplotlib. The linear regression line is obtained by multiplying the X values with the estimated coefficients.
#The code also prints the estimated intercept and slope coefficients.
