import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
    
    def train(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)  # Initialize weights and bias
        self.errors = []

        for _ in range(self.max_epochs):
            error_count = 0
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1)  # Add bias term
                activation = np.dot(xi, self.weights)
                prediction = np.where(activation >= 0, 1, -1)
                update = self.learning_rate * (target - prediction)
                self.weights += update * xi
                error_count += int(update != 0.0)
            self.errors.append(error_count)
            if error_count == 0:
                break
    
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # Add bias term
        activation = np.dot(X, self.weights)
        return np.where(activation >= 0, 1, -1)

# Generate linearly separable data
np.random.seed(42)
X1 = np.random.randn(50, 2) + np.array([2, 2])
X2 = np.random.randn(50, 2) + np.array([-2, -2])
X = np.vstack((X1, X2))
y = np.hstack((np.ones(50), -np.ones(50)))

# Train the Perceptron
perceptron = Perceptron()
perceptron.train(X, y)

# Calculate the margin for each data point
margins = np.abs(np.dot(X, perceptron.weights[1:]) + perceptron.weights[0]) / np.linalg.norm(perceptron.weights[1:])

# Plot the data points and decision boundary
plt.scatter(X1[:, 0], X1[:, 1], color='red', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='blue', label='Class -1')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()

# Plot the decision boundary
x_boundary = np.linspace(-5, 5, 100)
y_boundary = (-perceptron.weights[0] - perceptron.weights[1] * x_boundary) / perceptron.weights[2]
plt.plot(x_boundary, y_boundary, color='green', label='Decision Boundary')

# Plot the margins
plt.scatter(X[:, 0], X[:, 1], c=margins, cmap='viridis', marker='o', label='Margins')
cbar = plt.colorbar()
cbar.set_label('Margin')
plt.legend()

plt.show()


#In this code, we first generate linearly separable data with two classes using the NumPy library. The first class (X1) is generated around the point (2, 2), and the second class (X2) is generated around the point (-2, -2).
#We then create an instance of the Perceptron class and train it using the generated data.
#After training, we calculate the margin for each data point by calculating the distance between each point and the decision boundary.
#Finally, we plot the data points, the decision boundary, and the margins. The margins are represented by the color of the data points, where a larger
