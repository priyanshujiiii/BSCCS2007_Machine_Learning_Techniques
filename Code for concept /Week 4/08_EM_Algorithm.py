import numpy as np
from scipy.stats import multivariate_normal

# Generate sample data
np.random.seed(42)
n_samples = 500
# Parameters for the two Gaussian components
mu1 = [0, 0]
cov1 = [[2, 0], [0, 2]]
mu2 = [5, 5]
cov2 = [[1, 0], [0, 1]]
# Mixing coefficients for the two components
weights = [0.4, 0.6]
# Generate samples from the mixture model
X = np.concatenate([
    np.random.multivariate_normal(mu1, cov1, int(n_samples * weights[0])),
    np.random.multivariate_normal(mu2, cov2, int(n_samples * weights[1]))
])

# Initialize the parameters
n_components = 2
n_features = X.shape[1]
means = np.random.randn(n_components, n_features)
covariances = [np.eye(n_features)] * n_components
weights = np.ones(n_components) / n_components

# EM algorithm
n_iterations = 100
log_likelihoods = []
for _ in range(n_iterations):
    # E-step: Calculate responsibilities
    likelihoods = np.array([multivariate_normal.pdf(X, mean=means[k], cov=covariances[k]) for k in range(n_components)]).T
    responsibilities = likelihoods * weights
    responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

    # M-step: Update parameters
    total_responsibilities = np.sum(responsibilities, axis=0)
    weights = total_responsibilities / n_samples
    means = np.dot(responsibilities.T, X) / total_responsibilities[:, np.newaxis]
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / total_responsibilities[k]

    # Calculate and store the log-likelihood
    log_likelihood = np.sum(np.log(np.dot(likelihoods, weights)))
    log_likelihoods.append(log_likelihood)

# Plot the log-likelihood curve
import matplotlib.pyplot as plt
plt.plot(range(1, n_iterations + 1), log_likelihoods)
plt.xlabel('Iterations')
plt.ylabel('Log-Likelihood')
plt.title('EM Algorithm - Log-Likelihood')
plt.show()

# Print the estimated parameters
print("Estimated weights:", weights)
print("Estimated means:")
for k, mean in enumerate(means):
    print(f"Component {k+1}: {mean}")
print("Estimated covariances:")
for k, covariance in enumerate(covariances):
    print(f"Component {k+1}:")
    print(covariance)

    
#In this code, we generate a synthetic dataset by sampling from a mixture of two Gaussian distributions with known parameters. We then initialize the parameters (means, covariances, and weights) for the GMM. The EM algorithm is performed for a specified number of iterations.
#During each iteration, the E-step calculates the responsibilities of each data point belonging to each component based on the current parameter estimates. The M-step updates the parameters by maximizing the expected log-likelihood given the responsibilities. The log-likelihood is also computed and stored at each iteration.
#After running the EM algorithm, the code plots the log-likelihood curve to visualize the convergence. It then prints the estimated parameters of the GMM (weights, means, and covariances).
#Note that this code assumes you have the NumPy, Sci
