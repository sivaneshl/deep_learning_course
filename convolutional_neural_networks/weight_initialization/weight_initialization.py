import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import helper_functions

# number of sub-processes used for data loading
num_workers = 0
# how many samples per batch
batch_size = 100
# percentage of training set to use for validation
valid_size = 0.2

data_dir = 'C:\\deep_learning\\deep_learning_course\\neural_networks\\deep_learning_with_pytorch\\F_MNIST_data'

# covert data into torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test data set
train_data = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation samples
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# specify the classes of the images
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# Visualize some training data
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy()

# plot the images in the batch along with their labels
fig = plt.figure(figsize=(25, 4))
for idx in range(20):
    ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(classes[labels[idx]])
plt.show()


# Define the NN architecture
class Net(nn.Module):
    def __init__(self, hidden_1=256, hidden_2=128, constant_weight=None):
        super(Net, self).__init__()
        # Linear layer 784 --> hidden_1
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # Linear layer hidden_1 --> hidden_2
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # Linear layer hidden_2 --> 10
        self.fc3 = nn.Linear(hidden_2, 10)
        # Dropout layer 20%
        self.dropout = nn.Dropout(p=0.2)

        # initialize the weights to a specified constant value
        if (constant_weight is not None):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, constant_weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # flatten the input image
        x = x.view(-1, 28 * 28)
        # add hidden layer with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add second hidden layer with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add the output layer
        x = self.fc3(x)
        return x


## Constant weights
# Initialize two NN's with 0 and 1 constant weights
model_0 = Net(constant_weight=0)
model_1 = Net(constant_weight=1)

# put them in a list
model_list = [(model_0, 'All zeros'),
              (model_1, 'All ones')]

# plot the loss and accuracy
helper_functions.compare_init_weights(model_list, 'All Zeros vs All Ones', train_loader, valid_loader)

## Random Uniform
def weights_init_uniform(m):
    """
    Takes in a module and applies the specified weight initialization
    :param m:
    :return:
    """
    classname = m.__class__.__name__
    # for every linear layer in model
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and bias 0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

# Create a new model with these weights
model_uniform = Net()
model_uniform.apply(weights_init_uniform)

# evaluate behaviour
helper_functions.compare_init_weights([(model_uniform, 'Uniform weights')],
                                      'Uniform Baseline',
                                      train_loader, valid_loader)

## Centered weights
def weights_init_uniform_centered(m):
    """
    Takes in a module and applies the specified weight initialization
    :param m:
    :return:
    """
    classname = m.__class__.__name__
    # for every linear layer in model
    if classname.find('Linear') != -1:
        # apply a centered, uniform distribution to the weights and bias 0
        m.weight.data.uniform_(-0.5, 0.5)
        m.bias.data.fill_(0)

# create a new model with these weights
model_centered = Net()
model_centered.apply(weights_init_uniform_centered)

## Centered weights
def weights_init_uniform_general_rule(m):
    """
    Takes in a module and applies the specified weight initialization
    :param m:
    :return:
    """
    classname = m.__class__.__name__
    # for every linear layer in model
    if classname.find('Linear') != -1:
        # get the number of inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

# create a new model with these weights
model_general_rule = Net()
model_general_rule.apply(weights_init_uniform_general_rule)

# compare these 2 models
model_list = [(model_centered, 'Centered weights (-0.5, 0.5'), (model_general_rule, 'General rule (-y, y)')]
helper_functions.compare_init_weights(model_list, '[-0.5, 0.5) vs [-y, y)', train_loader, valid_loader)

## Normal distribution
def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    # m.weight.data shoud be taken from a normal distribution
    # m.bias.data should be 0
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.normal_(0, y)
        m.bias.data.fill_(0)

# create a new model with the rule-based, NORMAL weights
model_normal_rule = Net()
model_normal_rule.apply(weights_init_normal)

# compare the two models
model_list = [(model_general_rule, 'Uniform Rule [-y, y)'),
              (model_normal_rule, 'Normal Distribution')]

# evaluate behavior
helper_functions.compare_init_weights(model_list, 'Uniform vs Normal', train_loader, valid_loader)

## Instantiate a model with _no_ explicit weight initialization
model_no_init = Net()

## evaluate the behavior using helpers.compare_init_weights# compare the two models
model_list = [(model_general_rule, 'Uniform Rule [-y, y)'),
              (model_normal_rule, 'Normal Distribution'),
              (model_no_init, 'No weight initialization')]

# evaluate behavior
helper_functions.compare_init_weights(model_list, 'Uniform vs Normal vs No Weight', train_loader, valid_loader)
