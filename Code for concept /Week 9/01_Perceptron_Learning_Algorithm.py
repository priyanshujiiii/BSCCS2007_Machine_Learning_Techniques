import numpy as np

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
                update = self.learning_rate * (target - self.predict(xi))
                self.weights += update * xi
                error_count += int(update != 0.0)
            self.errors.append(error_count)
            if error_count == 0:
                break
    
    def predict(self, X):
        activation = np.dot(X, self.weights[1:]) + self.weights[0]
        return np.where(activation >= 0, 1, -1)

# Example usage
X = np.array([[2, 1], [3, 4], [4, 3], [5, 6]])
y = np.array([-1, -1, 1, 1])

perceptron = Perceptron()
perceptron.train(X, y)

test_data = np.array([[1, 2], [6, 5]])
predictions = perceptron.predict(test_data)

print("Predictions:", predictions)
# Create a meshgrid of points to visualize the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Make predictions on the grid points
predictions = perceptron.predict(grid_points)
predictions = predictions.reshape(xx.shape)

# Plot the data points and decision boundary
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.contourf(xx, yy, predictions, alpha=0.5, cmap=plt.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Perceptron Decision Boundary')
plt.show()


#In this modified code, we import the matplotlib.pyplot module to plot the graph.
#After training the perceptron, we create a meshgrid of points (xx, yy) using the minimum and maximum values of the feature vectors. The np.meshgrid function generates a grid of points spanning the input space.
#We then flatten the grid points using np.ravel() and make predictions on these points using the trained perceptron. The predictions are reshaped to match the shape of the meshgrid (xx.shape).
#Finally, we plot the data points using plt.scatter(), the decision boundary using plt.contourf(), and add labels and a title to the plot. The alpha parameter in plt.contourf() sets the transparency level of the decision boundary.
#When you run the code, you will see a scatter plot of the data points along with the decision boundary learned by the perceptron algorithm. The decision boundary separates the two classes (-1 and 1) in the feature space.
