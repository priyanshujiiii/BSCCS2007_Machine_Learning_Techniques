import numpy as np
from hmmlearn import hmm

# Define the transition matrix
transmat = np.array([[0.7, 0.3],
                     [0.4, 0.6]])

# Define the emission probabilities
emissionprob = np.array([[0.9, 0.1],
                         [0.2, 0.8]])

# Define the initial probabilities
startprob = np.array([0.5, 0.5])

# Create the Hidden Markov Model (HMM)
model = hmm.MultinomialHMM(n_components=2, random_state=42)
model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob

# Generate a sequence of observations from the model
X, Z = model.sample(100)

# Fit the HMM to the observed sequence
model.fit(X)

# Generate a new sequence of observations from the learned model
X_new, Z_new = model.sample(100)

# Print the original and generated sequences
print("Original Sequence:")
print(X)
print("\nGenerated Sequence:")
print(X_new)


