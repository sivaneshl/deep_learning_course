import random
import utils
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# dataset - https://s3.amazonaws.com/video.udacity-data.com/topher/2018/October/5bbe6499_text8/text8.zip
with open('data/text8', 'r') as f:
    text = f.read()

# print first 100 characters
print(text[:100])

# get list of words
words = utils.preprocess(text)
print(words[:30])

# print some stats about this word data
print('Total words in text: {}'.format(len(words)))
print('Unique words: {}'.format(len(set(words))))

# dictionaries
vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words]
print(int_words[:30])

# Sub-sampling - eliminate irrelevant words like 'of', 'the' 'for'
threshold = 1e-5
word_counts = Counter(int_words)
print(list(word_counts.items())[0])

# discard some frequent words according to the subsampling equation
# create new list of words for training
total_count = len(int_words)
freq = {word: count/total_count for word, count in word_counts.items()}
p_drop = {word: 1-np.sqrt(threshold/freq[word]) for word in word_counts}
train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

print(train_words[:30])

# Setting the context window
def get_target(words, idx, window_size=5):
    """
    Get the list of in a window around the index
    :param words:
    :param idx:
    :param window_size:
    :return:
    """
    r = random.randint(1, window_size+1)
    total_words = len(words)
    target = [words[idx + i] for i in range(-r, r + 1) if i != 0 and total_words > (idx + i) >= 0]
    return target

# testing the get_target() function
int_text = [i for i in range(10)]
print('Input: ', int_text)
idx=5 # word index of interest
target = get_target(int_text, idx=idx, window_size=5)
print('Target: ', target)  # you should get some indices around the idx

# Generating batches of input and output
def get_batches(words, batch_size, window_size=5):
    """
    Create a generator for word batches as a tuple (input, target)
    :param words:
    :param batch_size:
    :param window_size:
    :return:
    """

    n_batches = len(words) // batch_size

    # only full batches
    words = words[:n_batches * batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx: idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x, y

# testing the function get_batches
int_text = [i for i in range(10)]
x, y = next(get_batches(int_text, batch_size=4, window_size=5))
print('x\n', x)
print('y\n', y)


def cosine_similarity(embedding, valid_size=16, valid_window=100, device='cpu'):
    """
    Returns the cosine similarity of the validation words with the embedding matrix.
    Here, embedding should be a PyTorch embedding module
    :param embedding:
    :param valid_size:
    :param valid_window:
    :param device:
    :return:
    """
    # Here, we are calculating the cosine similarity between some random words and
    # our embedding vectors. With the similarities, we can look at what words are
    # close to our random words.

    # sim = a.b / |a||b|

    embed_vectors = embedding.weight

    # magnitude of embedding vectors, |b|
    magnitudes = embedding.pow(2).sum(dim=1).sqrt().unsqueeze(0)

    # pick N words from our ranges (0, window) and (1000,1000+window)
    # lower id implies more frequent words
    # pick N/2 words from most frequent words
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    # pick N/2 words from least frequent words
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000, 1000+valid_window), valid_size//2))
    # convert to a LongTensor
    valid_vectors = torch.LongTensor(valid_examples).to(device)

    # compute the similarity
    similarities = torch.mm(valid_vectors, embed_vectors.t()) / magnitudes

    return valid_examples, similarities


# Define the SkipGram model
class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()

        # Embedding layer
        self.embed = nn.Embedding(num_embeddings=n_vocab, embedding_dim=n_embed)
        # output layer - linear layer
        self.output = nn.Linear(n_embed, n_vocab)
        # Log softmax
        self.log_softmax = nn.LogSoftmax(dim=1)

    # define the forward behaviour
    def forward(self, x):
        x = self.embed(x)
        scores = self.output(x)
        log_ps = self.log_softmax(scores)
        return log_ps


# Train the model
# check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

embedding_dim = 300
model = SkipGram(len(vocab_to_int), embedding_dim).to(device)
# Since we are applying softmax to our model, we should use NLLLoss instead of Cross entropy loss
# Softmax + NLLLoss = CrossEntropyLoss
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

print_every = 500
steps = 0
epochs = 5

# train for some number of epochs
for e in range(epochs):
    # get input and target batches
    for inputs, targets in get_batches(train_words, 512):
        steps += 1
        inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
        inputs, targets = inputs.to(device), targets.to(device)

        log_ps = model(inputs)
        loss = criterion(log_ps, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps % print_every == 0:
            # getting examples and similarites
            valid_examples, valid_similarities = cosine_similarity(model.embed, device=device)
            _, closest_idxs = valid_similarities.topk(6)

            vaslid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
            for ii, valid_idx in enumerate(valid_examples):
                closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                print(int_to_vocab[valid_idx.item()] + "|" + ', '.join(closest_words))
            print('...')

# Save the model checkpoint
model_name = 'skipgram_5_epoch.net'
checkpoint = {'n_vocab': len(vocab_to_int),
              'n_embed': embedding_dim,
              'state_dict': model.state_dict()
             }
with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)

# load the saved model
with open(model_name, 'rb') as f:
    checkpoint = torch.load(f)

loaded_model = SkipGram(checkpoint['n_vocab'], checkpoint['n_embed'])
loaded_model.load_state_dict(checkpoint['state_dict'])

# Visualising word vectors
# get embeddings from the embedding layer of our model
embeddings = model.embed.weight.to('cpu').data.numpy()
viz_words = 600
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

fig, ax = plt.subplots(figsize=(16, 16))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
plt.show()