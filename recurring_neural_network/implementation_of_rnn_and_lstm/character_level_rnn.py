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


# Check if the GPU is available
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Training on GPU')
else:
    print('No GPU available, training on CPU; consider making n_epochs very small.')

class CharRNN(nn.Module):

    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super(CharRNN, self).__init__()

        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        # Define the layers of the model
        # Define an LSTM layer that takes as params:
        #   an input size (the number of characters),
        #   a hidden layer size n_hidden,
        #   a number of layers n_layers,
        #   a dropout probability drop_prob,
        #   and a batch_first boolean (True, since we are batching)
        self.lstm = nn.LSTM(input_size=len(self.chars),
                            hidden_size=self.n_hidden,
                            num_layers=self.n_layers,
                            dropout=self.drop_prob,
                            batch_first=True)
        # Define a dropout layer
        self.dropout = nn.Dropout(p=self.drop_prob)
        # Define a fully-connected layer with params: input size n_hidden and output size (the number of characters)
        self.fc1 = nn.Linear(in_features=n_hidden, out_features=len(self.chars))


    def forward(self, x, hidden):
        """
        Forward pass through the network
        :param x:
        :param hidden:
        :return:
        """
        # Get the output and the new hidden state from the lstm
        r_out, hidden = self.lstm(x, hidden)
        # pass through dropout layer
        out = self.dropout(r_out)
        # stack up  lstm output using view
        out = out.contiguous().view(-1, self.n_hidden)
        # pass through fully connected layer
        out = self.fc1(out)
        return out, hidden

    def init_hidden(self, batch_size):
        """
        Initialize hidden state
        :param batch_size:
        :return:
        """

        # create 2 new tensors with size n_layers X batch_size X n_hidden initialized to zero for hidden state
        # and cell state of LSTM
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    """
    Training a network
    :param net: CharRNN network
    :param data: text data to train the network
    :param epochs: number of epochs to train
    :param batch_size: number of mini-sequences per mini-batch, aka batch_size
    :param seq_length: number of character steps per mini-batch
    :param lr: learning rate
    :param clip: gradient clipping
    :param val_frac: fraction of data to hold out for validation
    :param print_every: number of steps for printing training and validation loss
    :return:
    """

    # put the model in training mode
    net.train()

    # define the optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create the training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if train_on_gpu:
        net.cuda()

    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise we'd backprop through the entire history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform the backprop
            loss = criterion(output, targets.view(batch_size*seq_length).type(torch.LongTensor))

            loss.backward()
            # 'clip_grad_norm' helps prevent the exloding gradient problem in RNN / LSTM
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # one-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)
                    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                    # creating new variables for hidden_state, otherwise we'd backprob through the entire history
                    val_h = tuple([each.data for each in h])

                    if train_on_gpu:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).type(torch.LongTensor))
                    val_losses.append(val_loss.item())

                # reset to train mode
                net.train()

                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Validation Loss: {:.4f}...".format(np.mean(val_losses)))

# Instantiating the model
# define and print the net
n_hidden=512
n_layers=2

net = CharRNN(chars, n_hidden, n_layers)
print(net)

# set hyperparameters
batch_size = 128
seq_length = 100
n_epochs = 2 # start small if you are just testing initial behavior

# train the model
# train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)

# Save the model checkpoint
model_name = 'char_rnn_20_epoch.net'
checkpoint = {'n_hidden': n_hidden,
              'n_layers': n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}
with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)


# Making predictions
def predict(net, char, h=None, top_k=None):
    """
    Given a character, predict the next character
    Returns the predicted character and the hidden state
    :param net:
    :param char:
    :param h:
    :param top_k:
    :return:
    """
    # tensor inputs
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)

    if train_on_gpu:
        inputs = inputs.cuda()

    # detach hidden state from history
    h = tuple([each.data for each in h])

    # get output from the model
    output, h = net(inputs, h)

    # get the character probabilities
    p = F.softmax(output, dim=1).data
    if train_on_gpu:
        p = p.cpu()     # move to cpu

    # get top k character
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    # select the likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())

    # return the encoded value of the predicted character and the hidden state
    return net.int2char[char], h

def sample(net, size, prime='The', top_k=None):
    if train_on_gpu:
        net.cuda()
    else:
        net.cpu()

    net.eval()  # eval mode

    # first run through the prime charcters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)  # batch size is 1 as we are passing 1 char as input
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)

    # Now pass in the previous char and get the next char
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

print(sample(net, 100, prime='Anna', top_k=5))

# load the saved model
with open('char_rnn_20_epoch.net', 'rb') as f:
    checkpoint = torch.load(f, map_location=torch.device('cpu'))

loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])

print(sample(loaded, 1000, prime='Anna', top_k=5))