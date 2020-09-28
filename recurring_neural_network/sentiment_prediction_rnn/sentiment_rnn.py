import numpy as np
from string import punctuation
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

## Load and visualise data
with open('data/labels.txt', 'r') as f:
    labels = f.read()
with open('data/reviews.txt', 'r') as f:
    reviews = f.read()

print(reviews[:2000])
print()
print(labels[:20])

## Pre-process data
print(punctuation)

# get rid of punctuation
reviews = reviews.lower()
all_text = ''.join([c for c in reviews if c not in punctuation])
# print(all_text[:2000])

# split by new lines and spaces
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)
# create a list of words
words = all_text.split()
print(words[:30])

## Encoding the words
# build a dictionary that maps words to integers
word_counts = Counter(words)
sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(sorted_words, 1)}    # Later we're going to pad our input vectors with zeros, so make sure the integers start at 1, not 0

# use the dict to tokenize each review in reviews_split
# store the tokenized reviews in reviews_ints
reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])

# test this
# stats about vocabulary
print('Unique words: ', len((vocab_to_int)))
print()
# print tokens in first review
print('Tokenized review: \n', reviews_ints[:1])

## Encoding the labels
# convert labels from positive and negative to 0 and 1 and place them in a new list encoded_labels
encoded_labels = np.array([1 if label=='positive' else 0 for label in labels.split('\n')])
print(encoded_labels[:30])
print(len(reviews), len(labels), len(encoded_labels))

## Remve zero-length reviews
# outlier review stats
# count the reviews for each length
review_lens = Counter([len(x) for x in reviews_ints])
print('Zero-length reviews: {}'.format(review_lens[0]))
print('Maximum review length: {}'.format(max(review_lens)))

print('Number of reviews before removing outliers: {}'.format(len(reviews_ints)))
non_zero_len_idx = [ii for ii, review_int in enumerate(reviews_ints) if len(review_int) > 0]
reviews_ints = [reviews_ints[ii] for ii in non_zero_len_idx]
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_len_idx])
print('Number of reviews after removing outliers: {}'.format(len(reviews_ints)))

## Padding and truncating sequences
# For reviews shorter than some seq_length, we'll pad with 0s.
# For reviews longer than seq_length, we can truncate them to the first seq_length words.
# A good seq_length, in this case, is 200.
def pad_features(reviews_ints, seq_length):
    """
    Return features of reviews_ints, where each review is padded with 0's
    or truncated to the input seq_length
    :param reviews_ints:
    :param seq_length:
    :return:
    """
    features = []
    for review_int in reviews_ints:
        if len(review_int) < seq_length:
            features.append([0] * (seq_length - len(review_int)) + review_int)
        else:
            features.append(review_int[:seq_length])

    return np.array(features)

# Test your implementation!

seq_length = 200

features = pad_features(reviews_ints, seq_length=seq_length)
# print(len(reviews_ints), len(features))

## test statements - do not change - ##
assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 30 batches
print(features[:30,:10])

## Training, validation and test data sets
# split_frac is the fraction of data to keep in the training set
# Whatever data is left will be split in half to create the validation and testing data
split_frac = 0.8
split_idx = int(len(features)*split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

print(train_x.shape, test_x.shape, val_x.shape)

## Dataloaders and Batching
# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
val_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# data loaders
batch_size = 50
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Obtain one batch of training data
dataiter = iter(train_data)
sample_x, sample_y = next(dataiter)

print('Sample input size: ', sample_x.size())   # batch_size , seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size())   # batch_size
print('Sample label: \n', sample_y)


## Defining the model
# check if gpu is available
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Training on GPU')
else:
    print('No GPU available, training on CPU')

# An embedding layer that converts our word tokens (integers) into embeddings of a specific size.
# An LSTM layer defined by a hidden_state size and number of layers
# A fully-connected output layer that maps the LSTM layer outputs to a desired output_size
# A sigmoid activation layer which turns all outputs into a value 0-1; return only the last sigmoid output as the
# output of this network.

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis
    """
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers
        :param vocab_size:
        :param output_size:
        :param embedding_dim:
        :param hidden_dim:
        :param n_layers:
        :param drop_prob:
        """

        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # Define all layers

        # We need to add an embedding layer because there are 74000+ words in our vocabulary. It is massively
        # inefficient to one-hot encode that many classes. So, instead of one-hot encoding, we can have an embedding
        # layer and use that layer as a lookup table. You could train an embedding layer using Word2Vec, then load it
        # here. But, it's fine to just make a new layer, using it for only dimensionality reduction, and let the
        # network learn the weights.
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        # We'll create an LSTM to use in our recurrent network, which takes in an input_size, a hidden_dim,
        # a number of layers, a dropout probability (for dropout between multiple layers), and a batch_first parameter.
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.n_layers,
                            dropout=drop_prob,
                            batch_first=True)
        # Define a dropout layer
        self.dropout = nn.Dropout(p=0.3)
        # Define a fully-connected layer with params: input size n_hidden and output size (the number of characters)
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=self.output_size)

    def forward(self, x, hidden):
        """
        Forward pass thtough the network
        :param x:
        :param hidden:
        :return:
        """
        batch_size = x.size(0)

        # Get the embed weights
        x = self.embed(x)

        # Get the output and the new hidden state from the lstm
        lstm_out, hidden = self.lstm(x, hidden)

        # stack up  lstm output using view
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # pass through dropout layer
        out = self.dropout(lstm_out)

        # pass through fully connected layer
        out = self.fc1(out)

        # sigmoid activation
        sig_out = nn.Sigmoid(out)

        # reshape to be batch_size first
        ''' Now, I want to make sure that I’m returning only the last of these sigmoid outputs for a batch of input 
        data, so, I’m going to shape these outputs into a shape that is batch_size first. Then I'm getting the last 
        bacth by called `sig_out[:, -1], and that’s going to give me the batch of last labels that I want!
        '''
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]    # get only the last batch of labels

        return sig_out, hidden

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state
        :param batch_size:
        :return:
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

## Instantiate the network
# vocab_size: Size of our vocabulary or the range of values for our input, word tokens.
# output_size: Size of our desired output; the number of class scores we want to output (pos/neg).
# embedding_dim: Number of columns in the embedding lookup table; size of our embeddings.
# hidden_dim: Number of units in the hidden layers of our LSTM cells. Usually larger is better performance wise.
# Common values are 128, 256, 512, etc.
# n_layers: Number of LSTM layers in the network. Typically between 1-3
vocab_size = len(vocab_to_int)+1    # +1 for the 0 padding + our word tokens
output_size = 1     # this will be a sigmoid value between 0 and 1, indicating whether a review is positive or negative
embedding_dim = 400     # any value between like 200 and 500 or so would work
hidden_dim = 256
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
print(net)

## Training and optimization
# loss and optimization function
lr = 0.001
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

# training params
epochs = 4
counter = 0
print_every = 100
clip = 5 # gradient clipping

# move model to GPU if available
if train_on_gpu:
    net = net.cuda()

# training
net.train()
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        if train_on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        # creating new variables for the hidden state, otherwise
        # we'd have to back prop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform the back prop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # 'clip_grad_norm' helps prevent the exploding gradient problem in RNNs and LSTMs
        nn.utils.clip_grad_norm(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()

            # validation loop
            for inputs, labels in val_loader:

                # creating new variables for the hidden state
                val_h = tuple([each.data for each in val_h])

                if (train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))

## Testing
# Get the test loss and acuuracy
test_losses = []
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over the test data
for inputs, labels in test_loader:
    # create new set of hidden variables
    h = tuple([each.data for each in h])

    if train_on_gpu:
        inputs, labels = inputs.cuda(), labels.cuda()

    # get the output
    output, h = net(inputs, h)

    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss)

    # convert output probabilities to predicted classes
    pred = torch.round(output.squeeze()) # rounds to the nearest integer

    # compare predications to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

# stats
print('Test loss: {:.3f}'.format(np.mean(test_losses)))     # avg test loss
# accuracy over all test data
test_acc = num_correct / len(test_loader.dataset)
print('Test accuracy: {:.3f}'.format(test_acc))

## Inference on a test review
# Write a predict function that takes in a trained net, a plain text_review, and a sequence length,
# and prints out a custom statement for a positive or negative review!

# negative test review
test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'
# positive test review
test_review_pos = 'This movie had the best acting and the dialogue was so good. I loved it.'

def tokenize_review(test_review):
    # to lower case
    test_review = test_review.lower()
    # remove punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])
    # words
    test_words = test_text.split()
    # tokens
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])
    return test_ints

def predict(net, test_review, seq_length=200):
    """
    Points out whether a given review is predicted to be positive or negative in sentiment,
    using the trained model
    :param net: a trained net
    :param test_review: a review made of a normal text and punctuation
    :param seq_length: a padded length of a review
    :return:
    """
    net.eval()

    # tokenize review
    test_ints = tokenize_review(test_review)

    # pad tokenized sequence
    features = pad_features(test_ints, seq_length)

    # convert to tensor to pass to your model
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    # init hidden state
    h = net.init_hidden(batch_size)

    if train_on_gpu:
        feature_tensor = feature_tensor.cuda()

    # get the output from the model
    output, h = net(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())

    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    # print custom response
    if (pred.item() == 1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")