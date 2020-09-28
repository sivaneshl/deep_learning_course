import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# inputs to scoring function
dec_hidden_state = [5, 1, 20]

# visualize this vector
plt.figure(figsize=(1.5, 4.5))
sns.heatmap(np.transpose(np.matrix(dec_hidden_state)), annot=True, cmap=sns.light_palette('purple', as_cmap=True), linewidths=1)
plt.show()

# our first scoring function will score a single annotation (encoder hidden state) which looks like this
annotation = [3, 12, 45]    # Encoder hidden state

# visualise this vector
plt.figure(figsize=(1.5, 4.5))
sns.heatmap(np.transpose(np.matrix(annotation)), annot=True, cmap=sns.light_palette('orange', as_cmap=True), linewidths=1)
plt.show()

# scoring a single annotation
def single_dot_attention_score(dec_hidden_state, enc_hidden_state):
    return np.dot(dec_hidden_state, enc_hidden_state)
print(single_dot_attention_score(dec_hidden_state, annotation))

# annotation matrix
annotations = np.transpose([[3,12,45], [59,2,5], [1,43,5], [4,3,45.3]])

# visualize this data - each column is a hidden state of encoder time step
ax = sns.heatmap(annotations, annot=True, cmap=sns.light_palette("orange", as_cmap=True), linewidths=1)
plt.show()

# scoring all annotations at once
def dot_attention_score(dec_hidden_state, annotations):
    return np.matmul(np.transpose(dec_hidden_state), annotations)

attention_weights_raw = dot_attention_score(dec_hidden_state, annotations)
print(attention_weights_raw)

# softmax
def softmax(x):
    x = np.array(x, dtype=np.float128)
    # x = np.array(x)
    print(x)
    e_x = np.exp(x)
    print(e_x)
    return e_x / e_x.sum(axis=0)

attention_weights = softmax(attention_weights_raw)
print(attention_weights)

# apply attention scores
def apply_attention_scores(attention_weights, annotations):
    return attention_weights * annotations

applied_attention = apply_attention_scores(attention_weights, annotations)
print(applied_attention)

# visualise
ax = sns.heatmap(applied_attention, annot=True, cmap=sns.light_palette("orange", as_cmap=True), linewidths=1)
plt.show()

# calculating the attention context vector
def calculate_attention_context_vector(applied_attention):
    return np.sum(applied_attention, axis=1)

attention_vector = calculate_attention_context_vector(applied_attention)
print(attention_vector)

# visualize the attention context vector
plt.figure(figsize=(1.5, 4.5))
sns.heatmap(np.transpose(np.matrix(attention_vector)), annot=True, cmap=sns.light_palette("Blue", as_cmap=True), linewidths=1)
plt.show()

