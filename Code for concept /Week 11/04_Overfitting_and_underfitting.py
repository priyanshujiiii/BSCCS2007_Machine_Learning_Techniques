import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(0)
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y_true = 0.5 * X**3 - 2 * X**2 + 0.5 * X + 10 + np.random.normal(0, 10, size=(100, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

# Fit a polynomial regression model with degree 1 (underfitting)
poly_deg1 = PolynomialFeatures(degree=1)
X_poly_deg1 = poly_deg1.fit_transform(X_train)
model_deg1 = LinearRegression()
model_deg1.fit(X_poly_deg1, y_train)

# Fit a polynomial regression model with degree 20 (overfitting)
poly_deg20 = PolynomialFeatures(degree=20)
X_poly_deg20 = poly_deg20.fit_transform(X_train)
model_deg20 = LinearRegression()
model_deg20.fit(X_poly_deg20, y_train)

# Generate predictions on training and testing data
X_range = np.linspace(-5, 5, 200).reshape(-1, 1)
X_range_poly_deg1 = poly_deg1.transform(X_range)
y_pred_deg1 = model_deg1.predict(X_range_poly_deg1)
X_range_poly_deg20 = poly_deg20.transform(X_range)
y_pred_deg20 = model_deg20.predict(X_range_poly_deg20)

# Calculate mean squared errors on training and testing data
mse_train_deg1 = mean_squared_error(y_train, model_deg1.predict(X_poly_deg1))
mse_test_deg1 = mean_squared_error(y_test, model_deg1.predict(poly_deg1.transform(X_test)))
mse_train_deg20 = mean_squared_error(y_train, model_deg20.predict(X_poly_deg20))
mse_test_deg20 = mean_squared_error(y_test, model_deg20.predict(poly_deg20.transform(X_test)))

# Plot the data and models
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='red', label='Testing data')
plt.plot(X_range, y_pred_deg1, color='green', label='Degree 1 (Underfitting)')
plt.plot(X_range, y_pred_deg20, color='purple', label='Degree 20 (Overfitting)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression: Underfitting vs Overfitting')
plt.legend()

# Print mean squared errors
print("Mean Squared Error (Degree 1) - Training: {:.2f}, Testing: {:.2f}".format(mse_train_deg1, mse_test_deg1))
print("Mean Squared Error (Degree 20) - Training: {:.2f}, Testing: {:.2f}".format(mse_train_deg20, mse_test_deg20))

plt.show()

#In this code, we generate synthetic data that follows a cubic function with some added noise.
#We split the data into training and testing sets using the train_test_split function.
#We fit two polynomial regression models: one with degree 1 (underfitting) and another with degree 20 (overfitting).
#We generate predictions on a range of X values using the trained models.
#We calculate the mean squared errors on the training and testing data for both models.
#Finally, we plot the training and testing data points, as well as the fitted polynomial regression curves for degree 1 (underfitting) and degree 20 (overfitting).
#The mean squared errors are printed to evaluate the performance of each model.
