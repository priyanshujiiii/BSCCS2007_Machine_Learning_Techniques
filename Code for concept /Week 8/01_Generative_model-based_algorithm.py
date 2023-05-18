import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generate synthetic data with 3 clusters
X, y = make_blobs(n_samples=500, centers=3, random_state=42)

# Fit a Gaussian Mixture Model to the data
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Generate new samples from the learned model
X_new, y_new = gmm.sample(200)

# Plot the original and generated data points
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Original Data')
plt.scatter(X_new[:, 0], X_new[:, 1], c='red', marker='x', label='Generated Data')
plt.title('Generative Model: Gaussian Mixture Model')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


#In this code, we first generate synthetic data using the make_blobs function from scikit-learn. The data consists of 500 samples with 3 clusters.
#Next, we create a Gaussian Mixture Model (GMM) using GaussianMixture from scikit-learn. We set the number of components (clusters) to 3.
#After creating the GMM, we fit it to the data using the fit method.
#We then generate new samples from the learned model using the sample method of the GMM. In this example, we generate 200 new samples.
#Finally, we plot the original data points and the generated data points using plt.scatter from Matplotlib. The original data points are color-coded based on their cluster labels, and the generated data points are shown in red with 'x' markers.
#When you run the code, you will see a plot showing the original data points and the generated data points. This demonstrates the generative nature of the Gaussian Mixture Model, as it can generate new samples that resemble the original data distribution.

