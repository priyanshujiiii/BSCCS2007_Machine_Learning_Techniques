import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.2, 100)

# Kernel regression function
def kernel_regression(X, y, x_query, kernel_width):
    weights = np.exp(-0.5 * ((pairwise_distances(x_query.reshape(-1, 1), X.reshape(-1, 1))) / kernel_width) ** 2)
    weights /= np.sum(weights)
    return np.dot(weights, y)

# Set the kernel width parameter
kernel_width = 1.0

# Perform kernel regression for each data point
y_pred = [kernel_regression(X, y, x, kernel_width) for x in X]

# Plot the data and the kernel regression line
plt.scatter(X, y, label='Data')
plt.plot(X, y_pred, color='red', label='Kernel Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Kernel Regression')
plt.legend()
plt.show()

# In this code, we first generate sample data by creating X values as equally spaced points between 0 and 10, and y values as sin(X) with some added noise.
# The kernel_regression() function implements the kernel regression algorithm. Given the training data X and y, a query point x_query, and the kernel width parameter, it calculates the weights for each training point based on the Gaussian kernel function (exp(-0.5 * ((x_query - X) / kernel_width) ** 2)) and performs a weighted average to estimate the regression value at the query point.
# The code then iterates over each data point in X, performs kernel regression using kernel_regression() function, and stores the predicted regression values in y_pred.
# Finally, the code plots the data points and the kernel regression line using Matplotlib.
# You can adjust the kernel width parameter (kernel_width) to control the smoothness of the estimated regression function. Smaller values make the estimate more localized, while larger values result in smoother estimates.
