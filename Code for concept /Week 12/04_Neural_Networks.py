import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Generate some dummy data
X = np.random.random((1000, 10))
y = np.random.randint(2, size=(1000, 1))

# Split the data into training and testing sets
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Define the model architecture
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(10,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Plot the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


#In this code, we first generate some dummy data using np.random.random and np.random.randint.

#We split the data into training and testing sets.
#We define the model architecture using keras.Sequential and add layers.Dense layers with various activation functions.
#We compile the model using model.compile with the chosen optimizer, loss function, and metrics.
#We train the model using model.fit by providing the training data, number of epochs, and batch size.
#We evaluate the model on the testing data using model.evaluate and print the test loss and accuracy.
#This code demonstrates a basic neural network model using Keras for binary classification. You can modify the model architecture, activation functions, optimizer, and loss function to suit your specific needs.
