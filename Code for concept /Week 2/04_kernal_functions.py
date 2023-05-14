# Import required libraries
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

# Create two datasets
X1 = np.array([[1, 2], [3, 4], [5, 6]])
X2 = np.array([[2, 4], [1, 3], [6, 5]])

# Compute linear kernel
K_linear = pairwise_kernels(X1, X2, metric='linear')
print("Linear kernel:")
print(K_linear)

# Compute polynomial kernel with degree 2
K_poly = pairwise_kernels(X1, X2, metric='poly', degree=2)
print("Polynomial kernel with degree 2:")
print(K_poly)

# Compute RBF kernel with gamma=0.1
K_rbf = pairwise_kernels(X1, X2, metric='rbf', gamma=0.1)
print("RBF kernel with gamma=0.1:")
print(K_rbf)

#This code creates two datasets, computes the linear kernel, polynomial kernel with degree 2, and RBF kernel with gamma=0.1 between them using the pairwise_kernels function from scikit-learn library. The metric parameter is used to specify the kernel function and the degree and gamma parameters are used to specify the degree and gamma value for polynomial and RBF kernels, respectively.
#You can modify this code to work with other datasets, use a different kernel function, and use a different set of parameters for the kernel function as needed.
