from tensorflow import keras
from tensorflow.keras import layers

# Define the model architecture
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Count the number of parameters
num_params = model.count_params()

print("Number of parameters:", num_params)

#In this code, we define a neural network model with three layers using the keras.Sequential class.
#We then call the count_params() method on the model to compute the total number of parameters in the network.
#The result is printed, displaying the total number of parameters in the neural network.
#You can modify the architecture of the model by adding or removing layers, changing the number of units in each layer, or using different types of layers. The count_params() method will automatically compute the updated number of parameters for the modified model.
