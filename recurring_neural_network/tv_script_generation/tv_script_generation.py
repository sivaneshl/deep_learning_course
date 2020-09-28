import helper
import problem_unittests as tests
import numpy as np
from string import punctuation
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

## Load data
data_dir = 'data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)

## Explore data
view_line_range = (0, 10)

print('Dataset stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))

## Implement pre-process functions
# Lookup table
def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # text = text.lower()
    # all_text = ''.join([c for c in text if c not in punctuation])
    # text_split = all_text.split('\n')
    # all_text = ' '.join(text_split)
    # words = all_text.split()

    word_counts = Counter(text)
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(sorted_words)}
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_words)}

    return (vocab_to_int, int_to_vocab)

# test create_lookup_tables
tests.test_create_lookup_tables(create_lookup_tables)

# token lookup
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    punctuations = {}
    punctuations['.'] = '||PERIOD||'
    punctuations[','] = '||COMMA||'
    punctuations['"'] = '||QUOTATION_MARK||'
    punctuations[';'] = '||SEMICOLON||'
    punctuations['!'] = '||EXCLAMATION_MARK||'
    punctuations['?'] = '||QUESTION_MARK||'
    punctuations['('] = '||LEFT_PAREN||'
    punctuations[')'] = '||RIGHT_PAREN||'
    punctuations['-'] = '||DASH||'
    punctuations['?'] = '||QUESTION_MARK||'
    punctuations['\n'] = '||NEW_LINE||'

    return punctuations

# test token_lookup
tests.test_tokenize(token_lookup)

## Pre-process all the data and save it
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

## Checkpoint
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

## Check for GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your network')


## Input
# Batching
def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in each batch
    :return: DataLoader with batched data
    """
    # Get the number of batches we can make
    n_batches = len(words) // batch_size

    # Keep only enough characters to make full batches
    words = words[:n_batches * batch_size]

    features, targets = [], []

    # Iterate over the batches using the window of sequence_length
    for n in range(0, len(words) - sequence_length):
        # the features
        x = words[n:n + sequence_length]

        # the targets; shifted by 1
        try:
            y = words[n + sequence_length]
        except IndexError:
            y = words[0]
        features.append(x)
        targets.append(y)

    data = TensorDataset(torch.from_numpy(np.asarray(features)), torch.from_numpy(np.asarray(targets)))
    data_loader = DataLoader(data, batch_size=batch_size)

    # return a dataloader
    return data_loader


# test batch_data()
test_text = range(50)
t_loader = batch_data(test_text, sequence_length=5, batch_size=10)
data_iter = iter(t_loader)
sample_x, sample_y = next(data_iter)
print(sample_x.shape)
print(sample_x)
print()
print(sample_y.shape)
print(sample_y)


## Build the neural network
class RNN(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN module
        :param vocab_size: The number of input dimensions of the neural network (size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of the embeddings
        :param hidden_dim: The size of the hidden layer outputs
        :param n_layers: The number of LSTM layers
        :param dropout: Dropout to add between LSTM layers
        """
        super(RNN, self).__init__()

        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # define model layers
        self.embed = nn.Embedding(num_embeddings=vocab_size,
                                  embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            dropout=dropout,
                            batch_first=True)
        # self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(in_features=hidden_dim,
                            out_features=output_size)


    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """

        batch_size = nn_input.size(0)

        # Get the embed weights
        embeddings = self.embed(nn_input)
        # Get the output and the new hidden state from the lstm
        lstm_out, hidden = self.lstm(embeddings, hidden)
        # Stack up  lstm output using view
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # Pass through dropout layer
        # lstm_out = self.dropout(lstm_out)
        # Pass through fully connected layer
        out = self.fc(lstm_out)
        # Reshape to be batch_size first
        out = out.view(batch_size, -1, self.output_size)
        # Get the last batch output
        out = out[:, -1]

        return out, hidden

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state of an LSTM
        :param batch_size: The batch_size of the hidden state
        :return: Hidden state of dims (n_layers, batch_size, hidden_dim)
        """
        # Create two new tensors with sizes n_layers X batch_size X hidden_dim, initialized to zeros
        # for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

# testing the RNN
tests.test_rnn(RNN, train_on_gpu)

## Define forward and backpropagation
def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and back propagation on the neural network
    :param rnn: The neural network
    :param optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :param hidden: The hidden state
    :return: The loss and the latest hidden state Tensor
    """

    # move the data to GPU if available
    if train_on_gpu:
        inp, target = inp.cuda(), target.cuda()

    # creating new variables for the hidden state
    hidden = tuple([each.data for each in hidden])

    # zero accumulated gradients
    rnn.zero_grad()

    # get the output from the model
    output, hidden = rnn(inp, hidden)

    # calculate the loss and perform the backpropagation
    loss = criterion(output, target)
    loss.backward()
    # 'clip_grad_norm' helps prevent the exploding gradient problem in RNNs and LSTMs
    nn.utils.clip_grad_norm(rnn.parameters(), 5)
    optimizer.step()

    return loss.item(), hidden

# testing forward_back_prop()
tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)


## Training
# Data params
# Sequence Length
sequence_length = 32  # of words in a sequence
# Batch Size
batch_size = 128

# data loader - do not change
train_loader = batch_data(int_text, sequence_length, batch_size)

def train(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []

    rnn.train()

    print('Training for %d epochs...' % n_epochs)

    for epoch_i in range(1, n_epochs+1):
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            print(batch_i, end='\r')

            # make sure you iterate over completely full batches only
            n_batches = len(train_loader.dataset) // batch_size
            if (batch_i > n_batches):
                break

            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # return the trained rnn
    return rnn

# training parameters
# Number of Epochs
num_epochs = 5
# Learning Rate
learning_rate = 0.001

# Model parameters
# Vocab size
vocab_size = len(vocab_to_int)
# Output size
output_size = vocab_size
# Embedding Dimension
embedding_dim = 200
# Hidden Dimension
hidden_dim = 250
# Number of RNN Layers
n_layers = 2

# Show stats for every n number of batches
show_every_n_batches = 500

# create the model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

# defining the loss and optimization functions for training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

# training the model
trained_rnn = train(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

# saving the trained model
helper.save_model('./save/trained_rnn', trained_rnn)
print('Model trained and saved')



