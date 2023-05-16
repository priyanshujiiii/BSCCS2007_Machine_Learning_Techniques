import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + np.random.normal(0, 1, 100)

# Initialize the parameters
intercept = 0.0
slope = 0.0
learning_rate = 0.01
n_iterations = 1000

# Perform gradient descent
for _ in range(n_iterations):
    # Calculate predictions
    y_pred = intercept + slope * X

    # Calculate gradients
    gradient_intercept = (2 / len(X)) * np.sum(y_pred - y)
    gradient_slope = (2 / len(X)) * np.sum((y_pred - y) * X)

    # Update parameters
    intercept -= learning_rate * gradient_intercept
    slope -= learning_rate * gradient_slope

# Print the optimized coefficients
print("Intercept:", intercept)
print("Slope:", slope)

# Calculate the final predictions
y_pred = intercept + slope * X

# Plot the data and the optimized line
plt.scatter(X, y, label='Data')
plt.plot(X, y_pred, color='red', label='Optimized Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.show()


#In this code, we first generate sample data in the same way as before. We then initialize the intercept, slope, learning rate, and the number of iterations for the gradient descent algorithm.
#The gradient descent loop iteratively updates the intercept and slope by calculating gradients and taking steps proportional to the learning rate. The gradients are computed based on the mean squared error (MSE) between the predicted values (y_pred) and the actual target values (y).
#After running the gradient descent algorithm, the code prints the optimized intercept and slope coefficients.
#Finally, the code calculates the final predictions (y_pred) using the optimized coefficients and plots the data points and the optimized regression line using Matplotlib.
#Note that the learning rate and the number of iterations are hyperparameters that you can adjust to control the convergence and accuracy of the optimization process.
