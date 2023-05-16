import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.concatenate([np.random.normal(-2, 1, 300), np.random.normal(2, 1, 300)]).reshape(-1, 1)

# Initialize the Gaussian Mixture Model with 2 components
gmm = GaussianMixture(n_components=2)

# Fit the model to the data
gmm.fit(X)

# Generate new data samples from the fitted model
X_new = gmm.sample(600)[0]

# Get the estimated parameters of the model
means = gmm.means_
covariances = gmm.covariances_

# Plot the original and generated data samples
plt.scatter(X, np.zeros_like(X), label='Original Data', alpha=0.5)
plt.scatter(X_new, np.ones_like(X_new), label='Generated Data', alpha=0.5)
plt.title('Gaussian Mixture Model')
plt.legend()
plt.show()

#In this code, we generate a synthetic dataset with two underlying Gaussian distributions using NumPy's np.random.normal() function. We initialize a GaussianMixture object with 2 components and fit it to the data using the fit() method. We then generate new data samples from the fitted model using the sample() method.
#Finally, we plot the original data samples and the generated data samples on a 1D plot. The original data samples are represented by points with a label of 'Original Data', and the generated data samples are represented by points with a label of 'Generated Data'. The plot demonstrates how the Gaussian Mixture Model can capture the underlying distribution and generate new samples from it.
