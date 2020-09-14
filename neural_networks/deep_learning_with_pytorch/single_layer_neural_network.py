import numpy as np
import torch


def activation(x):
    """
    Sigmoid activation function
    :param x: torch.Tensor
    :return: sigmoid(x)
    """
    return 1 / (1 + torch.exp(-x))


# Generate some data
torch.manual_seed(7)  # Set the random seed so that things are predicable

# Features are some random normal variables
features = torch.randn((1, 5))
# True weights for our data
weights = torch.randn_like(features)
# and a true bias term
bias = torch.randn((1, 1))

# Calculate the output of this network using the weights and bias tensors
# Using .sum()
y = activation(torch.sum(features * weights) + bias)
print(y)
y = activation((features * weights).sum() + bias)
print(y)

# Using matrix multiplication
# using torch.mm() or torch.matmul()
y = activation(torch.mm(features, weights.view(5, 1)) + bias)
# weights.view(a, b) will return a new tensor with the same data as weights with size (a, b).
print(y)
