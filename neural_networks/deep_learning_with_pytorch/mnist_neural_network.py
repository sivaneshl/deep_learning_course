import numpy as np
import torch
import matplotlib.pyplot as plt
# import helper

from torchvision import datasets, transforms


def activation(x):
    return 1 / (1 + torch.exp(-x))


def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1, 1)


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Load the training data
trainset = datasets.MNIST('MNIST_data\\', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
plt.show()

n_inputs = 784
n_hidden = 256
n_output = 10

W1 = torch.randn((n_inputs, n_hidden))
W2 = torch.randn((n_hidden, n_output))

B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

inputs = images.view(images.shape[0], 784) # inputs = images.view(images.shape[0], -1) -1 can also be used
print(images.shape)

h = activation(torch.mm(inputs, W1) + B1)
y = torch.mm(h, W2) + B2
# print(y)
print(y.shape)

# Here, out should be the output of the network in the previous excercise with shape (64,10)
probabilities = softmax(y)

# Does it have the right shape? Should be (64, 10)
print(probabilities.shape)
# Does it sum to 1?
print(probabilities.sum(dim=1))



