import helper
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download and load the training data
trainset = datasets.FashionMNIST('F_MNIST_data\\', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('F_MNIST_data\\', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Test if the data is loaded correctly
image, label = next(iter(trainloader))
helper.imshow(image[0, :])
plt.show()


# Building the network
# Here you should define your network. As with MNIST, each image is 28x28 which is a total of 784 pixels,
# and there are 10 classes. You should include at least one hidden layer. We suggest you use ReLU activations
# for the layers and to return the logits or log-softmax from the forward pass. It's up to you how many layers you
# add and the size of those layers.
model = nn.Sequential(nn.Linear(784, 256),
                      nn.ReLU(),
                      nn.Linear(256, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# Create the network, define the criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

# Train the network here
# Make a forward pass through the network to get the logits
# Use the logits to calculate the loss
# Perform a backward pass through the network with loss.backward() to calculate the gradients
# Take a step with the optimizer to update the weights

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:

        # Flatten the input data
        images = images.view(images.shape[0], -1)

        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass, backward pass and then update the weights
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(trainloader)}")

# Test out your network!

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)

# Calculate the class probabilities (softmax) for img
with torch.no_grad():
    output = model.forward(img)

# Output of the network are logits, need to take softmax for probabilities
ps = F.softmax(output, dim=1)

# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')
plt.show()