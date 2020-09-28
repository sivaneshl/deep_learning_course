import time
import sys
import numpy as np
from collections import Counter

# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes=10, learning_rate=0.1, min_count=50, polarity_cutoff=0.05):
        """ Create a SentimentNetwork with the given settings
        Args:
            reviews (list) : List of reviews used for training
            labels (list) : List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes (int) : Number of nodes to create in the hidden layer
            learning_rate (float) : Learning rate to use when training
        """
        # Assign a seed to our random number generator to ensure we get reproducible results
        np.random.seed(1)

        # Process the reviews and their associated labels so that everything is ready for training
        self.pre_process_data(reviews, labels, min_count, polarity_cutoff)

        # Build the network to have the number of hidden nodes and learning rate that were passed to this
        # initializer. Make the same number of input nodes as there are vocabulary words and create a
        # single output node.
        self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels, min_count, polarity_cutoff):
        # Positive to negative ratio
        self.pos_neg_ratios = Counter()
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        for i, review in enumerate(reviews):
            total_counts.update(review.split(' '))
            if labels[i] == 'POSITIVE':
                positive_counts.update(review.split(' '))
            else:
                negative_counts.update(review.split(' '))

        for word in ({word: count for word, count in positive_counts.items() if count > 100}):
            self.pos_neg_ratios[word] = np.log(positive_counts[word] / float(negative_counts[word] + 1))

        # DONE: populate review_vocab with all of the words in the given reviews
        #       Remember to split reviews into individual words
        #       using "split(' ')" instead of "split()".
        review_vocab = set()
        for review in reviews:
            for word in review.split(' '):
                if total_counts[word] > min_count and abs(self.pos_neg_ratios[word]) > polarity_cutoff:
                    review_vocab.add(word)

        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)

        # DONE: populate label_vocab with all of the words in the given labels.
        #       There is no need to split the labels because each one is a single word.
        label_vocab = set(labels)

        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)

        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        for i, word in enumerate(self.label_vocab):
            self.label2index[word] = i

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # Initialize weights
        # DONE: initialize self.weights_0_1 as a matrix of zeros.
        #       These are the weights between the input layer and the hidden layer.
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))

        # DONE: initialize self.weights_1_2 as a matrix of random values.
        #       These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, (self.hidden_nodes, self.output_nodes))

        self.layer_1 = np.zeros((1, self.hidden_nodes))

    def get_target_for_label(self, label):
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

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self, output):
        return self.sigmoid(output)

    def train(self, training_reviews_raw, training_labels):

        # make sure out we have a matching number of reviews and labels
        assert (len(training_reviews_raw) == len(training_labels))

        training_reviews = []
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(' '):
                if word in self.word2index.keys():
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))

        # Keep track of correct predictions to display accuracy during training
        correct_so_far = 0

        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):

            # DONE: Get the next review and its correct label
            training_review = training_reviews[i]
            training_label = training_labels[i]
            y = self.get_target_for_label(training_label)

            # DONE: Implement the forward pass through the network.
            #       That means use the given review to update the input layer,
            #       then calculate values for the hidden layer,
            #       and finally calculate the output layer.
            #
            #       Do not use an activation function for the hidden layer,
            #       but use the sigmoid activation function for the output layer.
            self.layer_1 *= 0
            for idx in training_review:
                self.layer_1 += self.weights_0_1[idx]
            hidden_output = self.layer_1
            final_input = np.dot(hidden_output, self.weights_1_2)
            final_output = self.sigmoid(final_input)

            # DONE: Implement the back propagation pass here.
            #       That means calculate the error for the forward pass's prediction
            #       and update the weights in the network according to their
            #       contributions toward the error, as calculated via the
            #       gradient descent and back propagation algorithms you
            #       learned in class.
            output_error = y - final_output
            output_error_term = output_error * final_output * (1 - final_output)
            hidden_error = np.dot(output_error_term, self.weights_1_2.T)
            hidden_error_term = hidden_error

            self.weights_1_2 += self.learning_rate * np.dot(hidden_output.T, output_error_term)
            for idx in training_reviews[i]:
                self.weights_0_1[idx] += self.learning_rate * hidden_error_term[0]

            # DONE: Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output error
            #       is less than 0.5. If so, add one to the correct_so_far count.
            if final_output >= 0.5 and training_label == 'POSITIVE':
                correct_so_far += 1
            elif final_output < 0.5 and training_label == 'NEGATIVE':
                correct_so_far += 1


            # For debug purposes, print out our prediction accuracy and speed
            # throughout the training process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i + 1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i + 1))[:4] + "%")
            if (i % 2500 == 0):
                print("")

    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing reviews and uses the testing_labels to
        calculate the accuracy of those predictions
        :param testing_reviews:
        :param testing_labels:
        :return:
        """
        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict its label.
        for i in range(len(testing_reviews)):
            predict = self.run(testing_reviews[i])
            if predict == testing_labels[i]:
                correct += 1

            # For debug purposes, print out our prediction accuracy and speed throughout the prediction process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i + 1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i + 1))[:4] + "%")

    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review
        :param review:
        :return:
        """

        # DONE: Run a forward pass through the network, like you did in the
        #       "train" function. That means use the given review to
        #       update the input layer, then calculate values for the hidden layer,
        #       and finally calculate the output layer.
        #
        #       Note: The review passed into this function for prediction
        #             might come from anywhere, so you should convert it
        #             to lower case prior to using it.
        indices = set()
        for word in review.split(' '):
            if word in self.word2index.keys():
                indices.add(self.word2index[word])

        self.layer_1 *= 0
        for idx in indices:
            self.layer_1 += self.weights_0_1[idx]
        hidden_outputs = self.layer_1
        final_inputs = np.dot(hidden_outputs, self.weights_1_2)
        final_outputs = self.sigmoid(final_inputs)

        # DONE: The output layer should now contain a prediction.
        #       Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`,
        #       and `NEGATIVE` otherwise.
        if final_outputs >= 0.5:
            return 'POSITIVE'
        else:
            return 'NEGATIVE'



g = open('reviews.txt','r')
reviews = list(map(lambda x: x[:-1], g.readlines()))
g.close()

g = open('labels.txt','r')
labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
g.close()

# mlp = SentimentNetwork(reviews[:-1000], labels[:-1000], learning_rate=0.1)
# mlp.train(reviews[:-1000], labels[:-1000])
# mlp.test(reviews[-1000:], labels[-1000:])

mlp_full = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=0,polarity_cutoff=0,learning_rate=0.01)
mlp_full.train(reviews[:-1000],labels[:-1000])


def get_most_similar_words(focus="horrible"):
    most_similar = Counter()

    for word in mlp_full.word2index.keys():
        most_similar[word] = np.dot(mlp_full.weights_0_1[mlp_full.word2index[word]],
                                    mlp_full.weights_0_1[mlp_full.word2index[focus]])

    return most_similar.most_common()

print(get_most_similar_words("excellent"))
print(get_most_similar_words("terrible"))

import matplotlib.colors as colors

words_to_visualize = list()
for word, ratio in mlp_full.pos_neg_ratios.most_common(500):
    if (word in mlp_full.word2index.keys()):
        words_to_visualize.append(word)

for word, ratio in list(reversed(mlp_full.pos_neg_ratios.most_common()))[0:500]:
    if (word in mlp_full.word2index.keys()):
        words_to_visualize.append(word)

pos = 0
neg = 0

colors_list = list()
vectors_list = list()
for word in words_to_visualize:
    if word in mlp_full.pos_neg_ratios.keys():
        vectors_list.append(mlp_full.weights_0_1[mlp_full.word2index[word]])
        if(mlp_full.pos_neg_ratios[word] > 0):
            pos+=1
            colors_list.append("#00ff00")
        else:
            neg+=1
            colors_list.append("#000000")


from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
words_top_ted_tsne = tsne.fit_transform(vectors_list)

p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="vector T-SNE for most polarized words")

source = ColumnDataSource(data=dict(x1=words_top_ted_tsne[:,0],
                                    x2=words_top_ted_tsne[:,1],
                                    names=words_to_visualize,
                                    color=colors_list))

p.scatter(x="x1", y="x2", size=8, source=source, fill_color="color")

word_labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,
                  text_font_size="8pt", text_color="#555555",
                  source=source, text_align='center')
# p.add_layout(word_labels)

show(p)

# green indicates positive words, black indicates negative words