import numpy as np
from scipy.optimize import minimize

# Define the objective function
def objective(x):
    return x[0]**2 + x[1]**2  # Example objective function (x1^2 + x2^2)

# Define the constraint function
def constraint(x):
    return x[0] + x[1] - 1  # Example constraint function (x1 + x2 - 1 <= 0)

# Define the initial guess
x0 = np.array([0, 0])

# Define the bounds for the variables
bounds = ((None, None), (None, None))  # No bounds in this example

# Define the constraint bounds
constraint_bounds = {'type': 'ineq', 'fun': constraint}

# Perform constrained optimization
result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraint_bounds)

# Print the optimal solution
print("Optimal solution:")
print("x =", result.x)
print("Objective value =", result.fun)


#In this code, we define the objective function objective(x) which represents the function to be minimized. In this example, it is a simple quadratic function x1^2 + x2^2.
#We also define the constraint function constraint(x) which represents the constraint that needs to be satisfied. In this example, it is a linear constraint x1 + x2 - 1 <= 0.
#Next, we define the initial guess x0 for the optimization variables.
#We define the bounds for the variables using the bounds tuple. In this example, we have no bounds, so we use None for both lower and upper bounds.
#We define the constraint bounds using the constraint_bounds dictionary. The type of constraint is set to 'ineq' for an inequality constraint, and we provide the constraint function in the 'fun' key.
#Finally, we perform the constrained optimization using the minimize function from scipy.optimize. We specify the objective function, initial guess, method (SLSQP in this case), bounds, and constraints. The result is stored in the result variable.
#We then print the optimal solution, which includes the values of the variables (x), and the objective value at the optimal solution.
#When you run the code, it will perform the constrained optimization and print the optimal solution. In this example, the solution satisfies the constraint x1 + x2 - 1 <= 0 and minimizes the objective function x1^2 + x2^2.
