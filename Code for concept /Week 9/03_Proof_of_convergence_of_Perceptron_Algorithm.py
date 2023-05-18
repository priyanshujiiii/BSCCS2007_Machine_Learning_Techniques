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

# Plot the convergence of errors
plt.plot(range(1, len(perceptron.errors) + 1), perceptron.errors)
plt.xlabel('Epochs')
plt.ylabel('Number of Errors')
plt.title('Convergence of Perceptron')
plt.show()


#In this code, we generate linearly separable data with two classes using the NumPy library. The first class (X1) is generated around the point (2, 2), and the second class (X2) is generated around the point (-2, -2).
#We then create an instance of the Perceptron class and train it using the generated data.
#After training, we plot the convergence of errors over epochs using plt.plot(). The x-axis represents the number of epochs, and the y-axis represents the number of misclassified points (errors) in each epoch. The plot shows how the number of errors decreases over epochs until convergence.
#When you run the code, you will see a plot that demonstrates the convergence of the Perceptron algorithm. The number of errors decreases with each epoch until all points are correctly classified, indicating that the algorithm has converged and found a separating hyperplane.




