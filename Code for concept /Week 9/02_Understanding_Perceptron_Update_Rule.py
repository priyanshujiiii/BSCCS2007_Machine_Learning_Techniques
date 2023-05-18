import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
    
    def train(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)  # Initialize weights and bias

        for _ in range(self.max_epochs):
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1)  # Add bias term
                activation = np.dot(xi, self.weights)
                prediction = np.where(activation >= 0, 1, -1)
                update = self.learning_rate * (target - prediction)
                self.weights += update * xi
    
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # Add bias term
        activation = np.dot(X, self.weights)
        return np.where(activation >= 0, 1, -1)

# Example usage
X = np.array([[2, 1], [3, 4], [4, 3], [5, 6]])
y = np.array([-1, -1, 1, 1])

perceptron = Perceptron()
perceptron.train(X, y)

print("Final weights:", perceptron.weights)


#In this code, we define a Perceptron class with an __init__ method to set the learning rate and maximum epochs, a train method to train the perceptron, and a predict method to make predictions.
#In the train method, we initialize the weights and bias to zero. Then, for each epoch, we iterate over the training data and update the weights based on the prediction error. We calculate the activation by taking the dot product of the feature vector (xi) and the weights. We then predict the class based on whether the activation is greater than or equal to zero. If the prediction is incorrect, we calculate the update value by multiplying the learning rate with the difference between the target and the prediction. Finally, we update the weights by adding the update value multiplied by the feature vector (xi).
#The predict method takes the input data (X) and inserts a bias term of 1 at the beginning of each sample. It calculates the activation by taking the dot product of the input data and the weights. The prediction is determined based on whether the activation is greater than or equal to zero.
#In the example usage, we create an instance of the Perceptron class and train it using a small dataset X and corresponding labels y. After training, we print the final weights learned by the perceptron.
#When you run the code, you will see the final weights learned by the perceptron after the training process. The weights are adjusted during training to find a decision boundary that separates the two classes in the feature space.
