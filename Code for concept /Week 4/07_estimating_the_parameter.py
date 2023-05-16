import numpy as np
from scipy.stats import norm

# Generate sample data from a Gaussian distribution with known parameters
np.random.seed(42)
mu_true = 2.0
sigma_true = 1.5
sample_data = np.random.normal(mu_true, sigma_true, size=100)

# Estimate the parameters using MLE
mu_hat, sigma_hat = norm.fit(sample_data)

# Print the estimated parameters
print("Estimated mean (mu):", mu_hat)
print("Estimated standard deviation (sigma):", sigma_hat)


#In this code, we first generate a sample of 100 data points from a Gaussian distribution with known parameters mu_true (mean) and sigma_true (standard deviation).
#Next, we use the norm.fit() function from the SciPy library to estimate the parameters of the Gaussian distribution based on the sample data. The function fits a Gaussian distribution to the data and returns the estimated mean (mu_hat) and standard deviation (sigma_hat).
#Finally, we print the estimated parameters. These estimated values represent the maximum likelihood estimates (MLE) based on the sample data.
#Note that the code assumes you have the SciPy library installed. You can install it using pip install scipy.




