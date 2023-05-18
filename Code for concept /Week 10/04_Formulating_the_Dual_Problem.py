import numpy as np
from scipy.optimize import minimize

# Define the objective function of the primal problem
def objective(x):
    return x[0]**2 + 2*x[1]**2 + 3*x[0]*x[1]  # Example objective function

# Define the constraint function of the primal problem
def constraint(x):
    return x[0] + x[1] - 1  # Example constraint function

# Define the Lagrangian function
def lagrangian(x, lambd):
    return objective(x) + lambd * constraint(x)

# Define the Lagrangian dual function
def lagrangian_dual(lambd):
    # Define the Lagrangian function for fixed lambd
    def lagrangian_fixed(x):
        return lagrangian(x, lambd)
    
    # Perform unconstrained optimization to find the minimum of the Lagrangian function for fixed lambd
    result = minimize(lagrangian_fixed, x0=[0, 0])
    
    return result.fun

# Perform unconstrained optimization to maximize the Lagrangian dual function
result = minimize(lambda lambd: -lagrangian_dual(lambd), x0=0)

# Print the optimal solution of the dual problem
print("Optimal solution of the dual problem:")
print("lambda =", result.x)
print("Objective value =", -result.fun)



#In this code, we start by defining the objective function of the primal problem using the objective function. This function represents the function to be minimized.
#We also define the constraint function of the primal problem using the constraint function. This function represents the constraint that needs to be satisfied.
#Next, we define the Lagrangian function using the lagrangian function. This function takes the variables x and the Lagrange multiplier lambd as inputs and calculates the value of the Lagrangian.
#We then define the Lagrangian dual function using the lagrangian_dual function. This function takes the Lagrange multiplier lambd as input and returns the minimum value of the Lagrangian for fixed lambd.
#Finally, we perform unconstrained optimization to maximize the Lagrangian dual function using the minimize function from scipy.optimize. We specify the function to be minimized as a lambda function that negates the Lagrangian dual function, and the initial guess for the Lagrange multiplier. The result is stored in the result variable.
#We then print the optimal solution of the dual problem, which includes the optimal Lagrange multiplier (lambda) and the objective value at the optimal solution.
#When you run the code, it will perform the unconstrained optimization to find the optimal solution of the dual problem.
