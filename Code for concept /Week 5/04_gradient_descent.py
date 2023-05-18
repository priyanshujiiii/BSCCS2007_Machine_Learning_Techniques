import numpy as np
import matplotlib.pyplot as plt

# Define the function to optimize
def function(x):
    return x ** 2 + 5 * np.sin(x)

# Define the gradient of the function
def gradient(x):
    return 2 * x + 5 * np.cos(x)

# Gradient descent algorithm
def gradient_descent(learning_rate, initial_x, n_iterations):
    x = initial_x
    x_history = [x]
    for _ in range(n_iterations):
        gradient_value = gradient(x)
        x -= learning_rate * gradient_value
        x_history.append(x)
    return x, np.array(x_history)

# Set the initial parameters
learning_rate = 0.1
initial_x = -5
n_iterations = 50

# Run gradient descent
optimized_x, x_history = gradient_descent(learning_rate, initial_x, n_iterations)

# Print the optimized value
print("Optimized x:", optimized_x)

# Plot the function and the optimization path
x_values = np.linspace(-10, 10, 100)
y_values = function(x_values)

plt.plot(x_values, y_values, label='Function')
plt.scatter(x_history, function(x_history), color='red', label='Optimization Path')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Optimization')
plt.legend()
plt.show()

# In this code, we define a function to optimize (function(x)), which is a combination of x^2 and 5*sin(x) in this example. We also define the gradient of the function (gradient(x)), which is the derivative of the function with respect to x.
# The gradient descent algorithm is implemented in the gradient_descent() function. It takes the learning rate, initial value of x, and the number of iterations as input. It iteratively updates x by subtracting the learning rate multiplied by the gradient of the function at x. The function keeps track of the optimization path by storing the history of x values.
# We then set the initial parameters, such as the learning rate, initial x value, and the number of iterations.
#The code runs the gradient descent algorithm using the provided parameters and stores the optimized x value.
#Finally, the code plots the function curve and the optimization path (the red dots), visualizing how the optimization progresses toward the optimal solution.
#Note that you can adjust the learning rate, initial x, and the number of iterations to control the convergence and accuracy of the optimization process.
