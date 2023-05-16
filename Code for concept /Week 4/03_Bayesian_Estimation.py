import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2)

# Define a range of k values to evaluate
k_values = range(1, 11)

# Calculate the BIC scores for each k
bic_scores = []
for k in k_values:
    gmm = GaussianMixture(n_components=k)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))

# Plot the BIC scores
plt.plot(k_values, bic_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Bayesian Information Criterion (BIC) Score')
plt.title('Bayesian Information Criterion Estimation')
plt.show()

# Find the best k value with the lowest BIC score
best_k = k_values[np.argmin(bic_scores)]

# Print the best k value
print("Best k:", best_k)


#In this code, we generate a random 2D dataset of 300 points. We define a range of k values (from 1 to 10 in this case) to evaluate. For each k value, we initialize a GaussianMixture object, fit it to the data, and calculate the Bayesian information criterion (BIC) score using the bic() method of GaussianMixture. The BIC score measures the trade-off between the model likelihood and the complexity of the model.
#Next, we plot the BIC scores, which show the relationship between the number of clusters (k) and the corresponding BIC scores. The lower the BIC score, the better the fit of the model.
#Finally, we find the best k value by selecting the one with the lowest BIC score and print it out.




