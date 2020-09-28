import numpy as np
from data_prep import features, targets, features_test, targets_test

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

np.random.seed(42)

# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 1000
learnrate = 0.005

last_loss = None
n_records, n_features = features.shape

weights_input_hidden = np.random.normal(scale=1/(n_features**0.5), size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1/(n_features**0.5), size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # DONE: Calculate the output
        hidden_input = np.dot(x, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)
        output = sigmoid(np.dot(hidden_output, weights_hidden_output))

        ## Backward pass ##
        # DONE: Calculate the network's prediction error
        error = y - output

        # DONE: Calculate error term for the output unit
        output_error_term = error * output * (1 - output)

        ## propagate errors to hidden layer
        # DONE: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, weights_hidden_output)

        # DONE: Calculate the error term for the hidden layer
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

        # DONE: Update the change in weights
        del_w_hidden_output += output_error_term * hidden_output
        del_w_input_hidden += hidden_error_term * x[:, None]

    # DONE: Update weights  (don't forget to division by n_records or number of samples)
    weights_hidden_output += learnrate * del_w_hidden_output / n_records
    weights_input_hidden += learnrate * del_w_input_hidden / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        output = sigmoid(np.dot(hidden_output, weights_hidden_output))
        loss = np.mean((output - targets)**2)

        if last_loss  and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)

# Calculate accuracy on test data
hidden_output = sigmoid(np.dot(features_test, weights_input_hidden))
output = sigmoid(np.dot(hidden_output, weights_hidden_output))
predictions = output > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))


