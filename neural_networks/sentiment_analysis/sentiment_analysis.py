from collections import Counter
import numpy as np

def pretty_print_review_and_label(i):
    print(labels[i] + '\t:\t' + reviews[i][:80] + '...')

f = open('reviews.txt', 'r')
reviews = list(map(lambda x: x[:-1], f.readlines()))
f.close()

f = open('labels.txt', 'r')
labels = list(map(lambda x: x[:-1].upper(), f.readlines()))
f.close()
# print(set(labels))

# print(len(reviews), len(labels))
# print(reviews[4], labels[4])
# pretty_print_review_and_label(47)

# DONE: Examine all the reviews.
# For each word in a positive review, increase the count for that word in both your positive counter and the total
# words counter; likewise, for each word in a negative review, increase the count for that word in both your negative
# counter and the total words counter.

positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()

for i, review in enumerate(reviews):
    total_counts.update(review.split(' '))
    if labels[i] == 'POSITIVE':
        positive_counts.update(review.split(' '))
    else:
        negative_counts.update(review.split(' '))

# print(positive_counts.most_common())
# print(negative_counts.most_common())
# print(total_counts.most_common())
# print(positive_counts['amazing'], negative_counts['amazing'])
# print(positive_counts['terrible'], negative_counts['terrible'])

# As you can see, common words like "the" appear very often in both positive and negative reviews. Instead of finding
# the most common words in positive or negative reviews, what you really want are the words found in positive reviews
# more often than in negative reviews, and vice versa. To accomplish this, you'll need to calculate the ratios of word
# usage between positive and negative reviews. Check all the words you've seen and calculate the ratio of positive to
# negative uses and store that ratio in pos_neg_ratios.
pos_neg_ratios = Counter()
# DONE: Calculate the ratios of positive and negative uses of the most common words
#       Consider words to be "common" if they've been used at least 100 times
# Go through all the ratios you calculated and convert them to logarithms. (i.e. use np.log(ratio))
for word in ({word: count for word, count in positive_counts.items() if count > 100}):
    pos_neg_ratios[word] = np.log(positive_counts[word] / float(negative_counts[word] + 1))

# print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
# print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
# print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))

# words most frequently seen in a review with a "POSITIVE" label
print(pos_neg_ratios.most_common())

# words most frequently seen in a review with a "NEGATIVE" label
print(list(reversed(pos_neg_ratios.most_common()))[0:30])

# DONE: Create a set named vocab that contains every word in the vocabulary.
vocab = set(total_counts.keys())
# print(vocab)
vocab_size = len(vocab)
print(vocab_size)

# DONE: Create a numpy array called layer_0 and initialize it to all zeros. You will find the zeros function
# particularly helpful here. Be sure you create layer_0 as a 2-dimensional matrix with 1 row and vocab_size columns.
layer_0 = np.zeros((1, vocab_size))
# print(layer_0.shape)

# Create a dictionary of words in the vocabulary mapped to index positions
# (to be used in layer_0)
word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i
# print(word2index)

def update_input_layer(review):
    """ Modify the global layer_0 to represent the vector form of review.
    The element at a given index of layer_0 should represent
    how many times the given word occurs in the review.
    Args:
        review(string) - the string of the review
    Returns:
        None
    """
    global layer_0
    # clear out previous state by resetting the layer to be all 0s
    layer_0 *= 0

    # DONE: count how many times each word is used in the given review and store the results in layer_0
    for word in review.split(' '):
        layer_0[0][word2index[word]] += 1


update_input_layer(reviews[0])
print(layer_0)

# DONE: Complete the implementation of get_target_for_labels. It should return 0 or 1, depending on whether the given
# label is NEGATIVE or POSITIVE, respectively.
def get_target_for_label(label):
    """Convert a label to `0` or `1`.
    Args:
        label(string) - Either "POSITIVE" or "NEGATIVE".
    Returns:
        `0` or `1`.
    """
    if label == 'POSITIVE':
        return 1
    else:
        return 0



