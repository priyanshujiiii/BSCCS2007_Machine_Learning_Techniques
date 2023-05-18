import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# Generate synthetic data for regression
np.random.seed(42)
X = np.random.rand(100, 5)
y = 2 * X[:, 0] + 3 * X[:, 1] - 4 * X[:, 2] + np.random.normal(0, 1, 100)

# Fit Lasso regression model
lasso_reg = Lasso(alpha=1.0)
lasso_reg.fit(X, y)
coefficients = lasso_reg.coef_

# Plotting the coefficients
fig, ax = plt.subplots()
ax.plot(range(len(coefficients)), coefficients, 'ro')
ax.set_xlabel('Coefficient Index')
ax.set_ylabel('Coefficient Value')
ax.set_title('Lasso Regression Coefficients')
plt.show()

#In this code, we first generate synthetic data for regression using the np.random.rand() function from NumPy.
#Next, we fit a Lasso regression model using the Lasso(alpha=1.0) from scikit-learn, where alpha is the regularization parameter controlling the strength of the penalty term.
#We obtain the coefficients of the Lasso regression model using the coef_ attribute.
#Finally, we print two characteristics of Lasso regression:
#The number of non-zero coefficients can be obtained using the np.count_nonzero() function applied to the coefficients. This characteristic demonstrates the sparsity-inducing property of Lasso regression. A higher number of non-zero coefficients suggests that more features are contributing to the model, while a lower number indicates feature selection, as some coefficients are effectively set to zero.
#The sum of absolute values of coefficients can be calculated using np.sum(np.abs(coefficients)). This characteristic measures the overall magnitude of the coefficients and can be used to assess the complexity of the model. Smaller values indicate a simpler model with fewer influential features.
#By running the code, you will see the number of non-zero coefficients and the sum of absolute values of the coefficients for the Lasso regression model.
#These characteristics highlight the sparsity and feature selection capabilities of Lasso regression, making it useful for tasks where interpretability and selecting relevant features are important.
