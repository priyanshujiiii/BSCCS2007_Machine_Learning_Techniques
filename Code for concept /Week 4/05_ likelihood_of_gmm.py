import numpy as np
from sklearn.mixture import GaussianMixture

# Generate sample data
np.random.seed(42)
X = np.concatenate([np.random.normal(-2, 1, 300), np.random.normal(2, 1, 300)]).reshape(-1, 1)

# Initialize and fit the Gaussian Mixture Model with 2 components
gmm = GaussianMixture(n_components=2)
gmm.fit(X)

# Compute the log-likelihood of a data point
data_point = np.array([[0.5]])  # Example data point
log_likelihood = gmm.score_samples(data_point)
likelihood = np.exp(log_likelihood)

print("Log-Likelihood:", log_likelihood)
print("Likelihood:", likelihood)

#In this code, we generate a synthetic dataset with two underlying Gaussian distributions using NumPy's np.random.normal() function. We initialize a GaussianMixture object with 2 components and fit it to the data using the fit() method.
#To compute the likelihood of a data point, we create a 1D NumPy array data_point representing the data point of interest. We then use the score_samples() method of the trained GMM to calculate the log-likelihood of the data point. By exponentiating the log-likelihood, we obtain the likelihood.
#The code prints both the log-likelihood and likelihood of the given data point.
