import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

# Generate synthetic data for regression
np.random.seed(42)
X = np.random.rand(100, 5)
y = 2 * X[:, 0] + 3 * X[:, 1] - 4 * X[:, 2] + np.random.normal(0, 1, 100)

# Fit linear regression model
linear_reg = LinearRegression()
linear_reg.fit(X, y)
coefficients_linear = linear_reg.coef_

# Fit ridge regression model
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X, y)
coefficients_ridge = ridge_reg.coef_

# Plotting the coefficients
fig, ax = plt.subplots()
ax.plot(range(len(coefficients_linear)), coefficients_linear, 'bo', label='Linear Regression')
ax.plot(range(len(coefficients_ridge)), coefficients_ridge, 'ro', label='Ridge Regression')
ax.set_xlabel('Coefficient Index')
ax.set_ylabel('Coefficient Value')
ax.set_title('Comparison of Linear Regression and Ridge Regression')
ax.legend()
plt.show()


#In this updated code, after fitting the linear regression and ridge regression models, we create a plot using Matplotlib to compare the coefficients of both models.
#We use the plot() function to plot the coefficients of linear regression (coefficients_linear) as blue circles ('bo') and the coefficients of ridge regression (coefficients_ridge) as red circles ('ro'). Each coefficient is represented on the x-axis by its index.
#The plot includes a title, x-label, y-label, and a legend to differentiate between the linear regression and ridge regression coefficients.
#When you run the code, you should see a plot showing the coefficients of both linear regression and ridge regression. The difference in coefficient magnitudes between the two models becomes apparent, with the ridge regression coefficients tending to be smaller due to the regularization effect.
#This plot provides a visual understanding of how ridge regression shrinks the coefficient values towards zero compared to the linear regression model.
