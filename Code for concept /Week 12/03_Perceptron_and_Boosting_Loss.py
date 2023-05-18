import numpy as np
import matplotlib.pyplot as plt

def perceptron_loss(y_true, y_pred):
    return np.mean(np.maximum(0, -y_true * y_pred))

def adaboost_loss(y_true, y_pred, weights):
    return np.sum(weights * np.exp(-y_true * y_pred)) / np.sum(weights)

def plot_loss_functions():
    # Generate sample data
    y_true = np.array([-1, 1, -1, -1, 1])
    y_pred = np.array([0.3, 0.7, -0.2, -0.6, 0.9])
    weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2])

    # Calculate the loss values for Perceptron loss and AdaBoost loss
    perceptron = perceptron_loss(y_true, y_pred)
    adaboost = adaboost_loss(y_true, y_pred, weights)

    # Plot the loss values
    loss_functions = ['Perceptron Loss', 'AdaBoost Loss']
    loss_values = [perceptron, adaboost]

    plt.bar(loss_functions, loss_values)
    plt.xlabel('Loss Function')
    plt.ylabel('Loss Value')
    plt.title('Perceptron Loss vs AdaBoost Loss')
    plt.show()

 #Plot the Perceptron loss and AdaBoost loss
plot_loss_functions()


#In this code, we define two loss functions:

#perceptron_loss: Computes the Perceptron loss, which penalizes misclassifications by the margin between the predicted and true labels.
#adaboost_loss: Computes the AdaBoost loss, which is used in the AdaBoost algorithm and assigns higher weights to misclassified samples.
#We also define a function plot_loss_functions that generates sample data, calculates the loss values for each loss function using the provided true labels y_true, predicted labels y_pred, and weights (for AdaBoost) weights, and plots the loss values using a bar chart.

#Finally, we call the plot_loss_functions function to visualize the Perceptron loss and AdaBoost loss.
