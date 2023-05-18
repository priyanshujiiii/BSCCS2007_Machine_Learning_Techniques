import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + np.random.normal(0, 1, 100)

# Define the Bayesian linear regression model
with pm.Model() as model:
    # Priors for the model parameters
    intercept = pm.Normal('intercept', mu=0, sd=10)
    slope = pm.Normal('slope', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Linear regression model
    y_pred = intercept + slope * X

    # Likelihood
    likelihood = pm.Normal('y', mu=y_pred, sd=sigma, observed=y)

    # Perform Bayesian inference
    trace = pm.sample(2000, tune=1000)

# Plot the posterior distribution of the model parameters
pm.plot_posterior(trace, var_names=['intercept', 'slope', 'sigma'])
plt.show()


#In this code, we first generate sample data in the same way as before. We then define the Bayesian linear regression model using the PyMC3 library.
#Within the model context, we specify the priors for the model parameters: intercept, slope, and sigma. Here, we use normal and half-normal priors for the intercept, slope, and standard deviation, respectively.
#We then define the linear regression model y_pred as a function of the model parameters.
#Next, we define the likelihood of the observed data y given the model predictions y_pred and the standard deviation sigma.
#Finally, we perform Bayesian inference using the pm.sample() function, specifying the number of samples and tuning steps. This generates a posterior distribution for each model parameter based on the observed data.
#The code then uses pm.plot_posterior() to visualize the posterior distributions of the model parameters.
#By using Bayesian modeling for linear regression, we obtain posterior distributions for the model parameters, which provide a more comprehensive understanding of the uncertainty associated with the parameter estimates.
