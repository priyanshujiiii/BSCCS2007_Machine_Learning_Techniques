import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2)

# Define a range of k values to evaluate
k_values = range(1, 11)

# Calculate the maximum likelihoods for each k
max_likelihoods = []
for k in k_values:
    gmm = GaussianMixture(n_components=k)
    gmm.fit(X)
    max_likelihoods.append(gmm.score(X))

# Plot the maximum likelihood curve
plt.plot(k_values, max_likelihoods, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Maximum Log-Likelihood')
plt.title('Maximum Likelihood Estimation')
plt.show()

# Find the best k value with the maximum likelihood
best_k = k_values[np.argmax(max_likelihoods)]

# Print the best k value
print("Best k:", best_k)


#In this code, we generate a random 2D dataset of 300 points. We define a range of k values (from 1 to 10 in this case) to evaluate. For each k value, we initialize a GaussianMixture object, fit it to the data, and calculate the maximum log-likelihood using the score() method of GaussianMixture. The maximum likelihood represents the probability of the observed data given the model parameters.
#Next, we plot the maximum likelihood curve, which shows the relationship between the number of clusters (k) and the maximum log-likelihoods. The higher the maximum likelihood, the better the fit of the model to the data.
#Finally, we find the best k value by selecting the one with the maximum likelihood and print it out.




