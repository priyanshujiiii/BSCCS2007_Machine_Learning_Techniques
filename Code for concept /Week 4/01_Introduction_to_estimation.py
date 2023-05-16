import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2)

# Define a range of k values to evaluate
k_values = range(1, 11)

# Calculate the log-likelihoods for each k
log_likelihoods = []
for k in k_values:
    gmm = GaussianMixture(n_components=k)
    gmm.fit(X)
    log_likelihoods.append(gmm.score(X))

# Calculate the probabilities for each k
probabilities = np.exp(log_likelihoods) / np.sum(np.exp(log_likelihoods))

# Plot the probability curve
plt.plot(k_values, probabilities, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Probability')
plt.title('Probability-based Estimation')
plt.show()

# Find the best k value with the highest probability
best_k = k_values[np.argmax(probabilities)]

# Print the best k value
print("Best k:", best_k)


#In this code, we generate a random 2D dataset of 300 points. We define a range of k values (from 1 to 10 in this case) to evaluate. For each k value, we initialize a GaussianMixture object, fit it to the data, and calculate the log-likelihood using the score() method of GaussianMixture. We then calculate the probabilities by exponentiating the log-likelihoods and normalizing them.
#Next, we plot the probability curve, which shows the relationship between the number of clusters (k) and the probabilities. The higher the probability, the more likely that the corresponding k value is the optimal number of clusters.
#Finally, we find the best k value by selecting the one with the highest probability and print it out.
