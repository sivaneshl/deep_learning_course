import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

plt.figure(figsize=(8, 5))

# how many steps/data points in one batch
seq_length = 20

# generate evenly spaced data points - using a sine wave
time_steps = np.linspace(0, np.pi, seq_length + 1)
data = np.sin(time_steps)
data.resize((seq_length + 1, 1))  # size becomes (seq_length+1, 1), adds an input_size dimension

x = data[:-1]  # all but the last piece of data
y = data[1:]  # all but the first

# display the data
plt.plot(time_steps[1:], x, 'r.', label='input, x')
plt.plot(time_steps[1:], y, 'b.', label='target, y')

plt.legend(loc='best')
plt.show()


# Defining the RNN network
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim

        # define an RNN with specified parameters
        # batch_first means the first dim of the input and output will be the batch size
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # last, fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)

        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)

        # get final output
        output = self.fc(r_out)

        return output, hidden


## check the input and output dimensions
test_rnn = RNN(input_size=1, output_size=1, hidden_dim=10, n_layers=2)

# generate evenly spaced data points - using a sine wave
time_steps = np.linspace(0, np.pi, seq_length)
data = np.sin(time_steps)
data.resize((seq_length, 1))

test_input = torch.Tensor(data).unsqueeze(0)  # give it a batch size of 1 as first dim
print('Input size: ', test_input.size())

test_out, test_hidden = test_rnn(test_input, None)
print('Output size: ', test_out.size())
print('Hidden size: ', test_hidden.size())

## training the model
# hyperparameters
input_size = 1
output_size = 1
hidden_dim = 32
n_layers = 1

# instantiate the RNN
rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)

# MSE loss and Adam optimizer with learning rate = 0.01
criterion = nn.MSELoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.01)

# train the RNN
def train(rnn, n_steps, print_every):

    # initialize the hidden state
    hidden = None

    for batch_i, step in enumerate(range(n_steps)):
        # defining the training data
        time_steps = np.linspace(step*np.pi, (step+1)*np.pi, (seq_length+1))
        # generate a different sine wave each time using the points
        data = np.sin(time_steps)
        data.resize((seq_length+1, 1))  # input size = 1

        x = data[:-1]
        y = data[1:]

        # convert data to Tensors
        x_tensor = torch.Tensor(x).unsqueeze(0)
        y_tensor = torch.Tensor(y)

        # outputs from rnn
        prediction, hidden = rnn(x_tensor, hidden)

        # representing memory
        # make a new variable for hidden and detach the hidden state from its history
        # this way we dont back propagate through the entire history
        hidden = hidden.data

        # calculate the loss
        loss = criterion(prediction, y_tensor)
        # zero gradients
        optimizer.zero_grad()
        # perform back propagation and update the weights
        loss.backward()
        optimizer.step()

        # display loss and predictions
        if batch_i % print_every == 0:
            print('Loss: ', loss.item())
            plt.plot(time_steps[1:], x, 'r.')   # input
            plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.')   # prediction
            plt.show()

    return rnn

# train the rnn and monitor the results
n_steps = 75
print_every = 15
trained_rnn = train(rnn, n_steps, print_every)