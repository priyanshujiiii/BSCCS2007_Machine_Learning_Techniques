# Import required libraries
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Create a dataset with a single feature
X = np.array([2, 3, 4]).reshape(-1, 1)

# Perform feature transformation using polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Print original and transformed features
print("Original features:")
print(X)
print("Transformed features:")
print(X_poly)


#This code creates a dataset with a single feature, performs feature transformation using polynomial features with degree 2, and prints the original and transformed features. The PolynomialFeatures class from scikit-learn is used to perform the polynomial feature transformation.
#You can modify this code to work with other datasets, use a different degree for the polynomial features, and use a different feature transformation technique as needed.
