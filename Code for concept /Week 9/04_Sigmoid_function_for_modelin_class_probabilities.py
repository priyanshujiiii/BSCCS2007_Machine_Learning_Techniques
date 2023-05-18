import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Example usage
z = np.array([-1, 0, 1, 2])
probabilities = sigmoid(z)

print("Probabilities:", probabilities)
