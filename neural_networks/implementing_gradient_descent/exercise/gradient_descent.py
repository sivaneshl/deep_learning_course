import numpy as np
from data_prep import features, targets, features_test, targets_test

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# Initialize weights
weights = np.random.normal(scale=1/(n_features**0.5), size=n_features)


# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5


for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        # DONE: Calculate the output y_hat
        h = np.dot(x, weights)
        y_hat = sigmoid(h)

        # DONE: Calculate the error
        error = y - y_hat

        # DONE: Calculate the error term
        error_term = error * sigmoid_prime(h)

        # DONE: Calculate the change in weights for this sample
        #       and add it to the total weight change
        del_w += error_term * x

    # DONE: Update weights using the learning rate and the average change in weights
    weights += learnrate * del_w / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        predicts = sigmoid(np.dot(features, weights))
        loss = np.mean((predicts - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
final_predicts = sigmoid(np.dot(features_test, weights))
predictions = final_predicts > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))


