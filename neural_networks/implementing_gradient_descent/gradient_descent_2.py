import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """
    # Derivative of the sigmoid function
    """
    return sigmoid(x) * (1 - sigmoid(x))


learnrate = 0.5
x = np.array([1, 2, 3, 4])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5, 0.3, 0.1])

# Calculate one gradient descent step for each weight
# Note: Some steps have been consolidated, so there are fewer variable names than in the above sample code

# DONE: Calculate the node's linear combination of inputs and weights
h = np.matmul(x, w)

# DONE: Calculate output of neural network
y_hat = sigmoid(h)

# DONE: Calculate error of neural network
error = y - y_hat

# DONE: Calculate the error term
#       Remember, this requires the output gradient, which we haven't
#       specifically added a variable for.
# δ = (y−y^)f′(h) = (y−y^)f′(∑wixi)
error_term = error * sigmoid_prime(h)


# DONE: Calculate change in weights
delta_w = learnrate * error_term * x

print('Neural Network output:')
print(y_hat)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(delta_w)



