import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
import matplotlib.pyplot as plt

# Generate synthetic data for regression
np.random.seed(42)
X = np.random.rand(100, 5)
y = 2 * X[:, 0] + 3 * X[:, 1] - 4 * X[:, 2] + np.random.normal(0, 1, 100)

# Fit linear regression model
linear_reg = LinearRegression()
linear_reg.fit(X, y)
coefficients_linear = linear_reg.coef_

# Fit Lasso regression model
lasso_reg = Lasso(alpha=1.0)
lasso_reg.fit(X, y)
coefficients_lasso = lasso_reg.coef_

# Plotting the coefficients
fig, ax = plt.subplots()
ax.plot(range(len(coefficients_linear)), coefficients_linear, 'bo', label='Linear Regression')
ax.plot(range(len(coefficients_lasso)), coefficients_lasso, 'ro', label='Lasso Regression')
ax.set_xlabel('Coefficient Index')
ax.set_ylabel('Coefficient Value')
ax.set_title('Comparison of Linear Regression and Lasso Regression')
ax.legend()
plt.show()


#In this code, we generate synthetic data for regression and fit both a linear regression model (LinearRegression()) and a Lasso regression model (Lasso(alpha=1.0)).
#We then obtain the coefficients of both models (coefficients_linear and coefficients_lasso).
#Next, we create a plot using Matplotlib to compare the coefficients of linear regression and Lasso regression.
#The plot shows the coefficients of linear regression as blue circles ('bo') and the coefficients of Lasso regression as red circles ('ro'). Each coefficient is represented on the x-axis by its index.
#The plot includes a title, x-label, y-label, and a legend to differentiate between the linear regression and Lasso regression coefficients.
#When you run the code, you should see a plot that visualizes the coefficients of both linear regression and Lasso regression. Comparing the two, you'll notice that Lasso regression tends to shrink some coefficients to exactly zero. This property of Lasso regression allows for feature selection, as it can effectively set coefficients to zero and exclude irrelevant features from the model.
#The code snippet demonstrates the relation between the solution of linear regression and Lasso regression by showcasing the differences in coefficient values and the potential sparsity of the Lasso regression coefficients.
