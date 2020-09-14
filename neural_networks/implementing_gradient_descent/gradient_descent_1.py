import numpy as np


# Defining the sigmoid function for activations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    # return 1 / (1 + np.power(np.e, -x))


# Derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Input data
x = np.array([0.1, 0.3])
# Target
y = 0.2
# Input to output weights
w = np.array([-0.8, 0.5])
# The learning rate, eta in the weight step equation
learnrate = 0.5

# the linear combination performed by the node (h in f(h) and f'(h))
# h = x[0] * w[0] + x[1] * w[1]
h = np.matmul(x, w)
print('h: ', h)

# The neural network output (y_hat)
y_hat = sigmoid(h)
print('y_hat: ', y_hat)

# output error (y - y-hat)
error = y - y_hat
print('error: ', error)

# output gradient (f'(h))
output_grad = sigmoid(h)
print('output_grad: ', output_grad)

# error term (lowercase delta)
# δ = (y−y^)f′(h) = (y−y^)f′(∑wixi)
error_term = error * output_grad
print('error_term: ', error_term)

# Gradient descent step
delta_w = learnrate * error_term * x
print('delta_w: ', delta_w)
