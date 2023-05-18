import numpy as np
import matplotlib.pyplot as plt

def logistic_loss(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def hinge_loss(y_true, y_pred):
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

def cross_entropy_loss(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred))

def plot_loss_functions():
    # Generate sample data
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0.1, 0.7, 0.3, 0.6, 0.9])

    # Calculate the loss values for different loss functions
    logistic = logistic_loss(y_true, y_pred)
    hinge = hinge_loss(y_true, y_pred)
    cross_entropy = cross_entropy_loss(y_true, y_pred)

    # Plot the loss values
    loss_functions = ['Logistic Loss', 'Hinge Loss', 'Cross-Entropy Loss']
    loss_values = [logistic, hinge, cross_entropy]

    plt.bar(loss_functions, loss_values)
    plt.xlabel('Loss Function')
    plt.ylabel('Loss Value')
    plt.title('Classification Loss Functions')
    plt.show()

# Plot the classification loss functions
plot_loss_functions()


#In this code, we define three classification loss functions:

#logistic_loss: Computes the logistic loss, which is commonly used for binary classification problems.
#hinge_loss: Computes the hinge loss, which is commonly used for binary classification problems with Support Vector Machines (SVM).
#cross_entropy_loss: Computes the cross-entropy loss, which is commonly used for multi-class classification problems.
#We also define a function plot_loss_functions that generates sample data, calculates the loss values for each loss function using the provided true labels y_true and predicted probabilities y_pred, and plots the loss values using a bar chart.

#Finally, we call the plot_loss_functions function to visualize the classification loss functions.
