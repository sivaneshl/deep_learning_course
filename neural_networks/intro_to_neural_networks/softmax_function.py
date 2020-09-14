import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    result = []
    exp_L = np.exp(L)
    exp_tot = exp_L.sum()
    [result.append(l / exp_tot) for l in exp_L]
    return result