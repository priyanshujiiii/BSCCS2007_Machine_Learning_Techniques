import numpy as np
import matplotlib.pyplot as plt

# Define a convex function: f(x) = x^2
def convex_function(x):
    return x ** 2

# Define Jensen's inequality: f(E[X]) <= E[f(X)]
def jensen_inequality(X, f):
    expected_X = np.mean(X)
    expected_f_X = f(expected_X)
    return expected_f_X, np.mean(f(X))

# Generate random data
np.random.seed(42)
X = np.random.randn(100)

# Calculate the values of the convex function
Y = convex_function(X)

# Apply Jensen's inequality
expected_f_X, mean_f_X = jensen_inequality(X, convex_function)

# Plot the data and the convex function
plt.scatter(X, Y, label='Data Points')
x_vals = np.linspace(-3, 3, 100)
plt.plot(x_vals, convex_function(x_vals), label='Convex Function')
plt.axhline(y=expected_f_X, color='r', linestyle='--', label='E[f(X)]')
plt.axhline(y=mean_f_X, color='g', linestyle='--', label='E[X]')

plt.xlabel('X')
plt.ylabel('f(X)')
plt.title('Convex Function and Jensen\'s Inequality')
plt.legend()
plt.show()

# Print the results
print('Expected f(X):', expected_f_X)
print('Mean f(X):', mean_f_X)

In this code, we define a simple convex function convex_function(x) = x^2. Then, we define the Jensen's inequality function jensen_inequality(X, f) that calculates the expected value of f(X) and the mean value of f(X) for a given random variable X and a function f.

Next, we generate random data X using NumPy's np.random.randn() function. We calculate the corresponding values of the convex function Y by applying convex_function() to each element of X.

Then, we apply Jensen's inequality using jensen_inequality() to calculate the expected value of f(X) and the mean value of f(X) based on the generated data. These values are plotted as dashed lines in the plot.

Finally, we plot the data points, the convex function curve, and the lines representing the expected value of f(X) and the mean value of f(X). We also print the calculated expected value and mean value.

The plot and printed results demonstrate the concept of a convex function and the application of Jensen's inequality.
