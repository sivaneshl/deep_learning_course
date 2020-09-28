import numpy as np

g = open('reviews.txt','r')
reviews = list(map(lambda x: x[:-1], g.readlines()))
g.close()

g = open('labels.txt','r')
labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
g.close()

review_vocab = set()
for review in reviews:
    review_vocab.update(review.split(' '))
# print(review_vocab)
# print(len(review_vocab))

word2index = {}
for i, word in enumerate(review_vocab):
    word2index[word] = i
# print(word2index)
# print(len(word2index))

for review in reviews[0:3]:
    print(review)
    print([index for word, index in word2index.items() if word in review.split(' ')])



# layer_0 = np.zeros((1, len(review_vocab)))
# layer_0 *= 0
# for word in reviews[100].split(' '):
#     layer_0[0][word2index[word]] += 1
# print(layer_0)