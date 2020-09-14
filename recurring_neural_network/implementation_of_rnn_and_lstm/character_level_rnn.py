import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# open text file and read data as 'text'
with open('anna.txt', 'r') as f:
    text = f.read()

# check the first 100 characters
print(text[:100])

# Turn characters into numerical data as our network can only learn from numbers
# encode the text and map each character to an integer and vice versa
# create 2 dictionaries
# 1. int2char - which maps integers to characters
# 2. char2int - which maps characters to unique integers
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
# encode the text
encoded = np.array([char2int[ch] for ch in text])


# print(chars)
# print(int2char)
# print(char2int)
# print(encoded[:100])

def one_hot_encode(arr, n_labels):
    # initialize the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    # fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1
    # finally reshape it to get the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot


# check if the function works as expected
test_seq = np.array([[3, 5, 1]])
one_hot = one_hot_encode(test_seq, 8)
print(one_hot)

# Making Training mini-batches
def get_batches(arr, batch_size, seq_length):
    """
    Create a generator that returns batches of size batch_size X seq_length from arr
    :param arr: Array you want to make batches from
    :param batch_size: Batch size, the number of sequences per batch
    :param seq_length: Number of encoded chars in as sequence
    :return:
    """

    # Get the number of batches we can make
    n_batches = len(arr) // (batch_size * seq_length)

    # Keep only enough characters to make full batches
    n_chars = batch_size * seq_length * n_batches
    arr = arr[:n_chars]

    # Reshape into batch_size rows
    arr = arr.reshape((batch_size,-1))

    # Iterate over the batches using the window of seq_length
    for n in range(0, arr.shape[1], seq_length):
        # the features
        x = arr[:, n:n+seq_length]
        # the targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


# Test the implementation
batches = get_batches(encoded, 8, 50)
x, y = next(batches)

# printing out the first 10 items in a sequence
print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])
