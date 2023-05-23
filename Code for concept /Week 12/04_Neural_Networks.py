import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize the weights and biases
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.biases1 = np.random.randn(hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.biases2 = np.random.randn(output_size)

    def forward(self, x):
        # Calculate the output of the hidden layer
        a1 = np.dot(x, self.weights1) + self.biases1
        h = np.sigmoid(a1)

        # Calculate the output of the output layer
        a2 = np.dot(h, self.weights2) + self.biases2
        y = np.softmax(a2)

        return y

    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            y_hat = self.forward(x)

            # Calculate the loss
            loss = -np.mean(y * np.log(y_hat))

            # Backpropagate the loss
            d_y_hat = y_hat - y
            d_a2 = d_y_hat * np.exp(a2) / np.sum(np.exp(a2))
            d_w2 = np.dot(h.T, d_a2)
            d_b2 = np.sum(d_a2, axis=0)
            d_h = np.dot(d_a2, self.weights2.T)
            d_a1 = d_h * sigmoid_prime(a1)
            d_w1 = np.dot(x.T, d_a1)
            d_b1 = np.sum(d_a1, axis=0)

            # Update the weights and biases
            self.weights1 += -learning_rate * d_w1
            self.biases1 += -learning_rate * d_b1
            self.weights2 += -learning_rate * d_w2
            self.biases2 += -learning_rate * d_b2

        return loss

