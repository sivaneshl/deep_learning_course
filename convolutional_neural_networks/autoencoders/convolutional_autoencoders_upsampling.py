import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# load the training and test datasets
data_dir = 'C:\\deep_learning\\deep_learning_course\\neural_networks\\deep_learning_with_pytorch\\MNIST_data\\'
train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

## create training and test data loaders
# number of sub processes to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# define the data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

## Visualize the image
# obtain one batch of training data
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy()

# get one image from batch
img = np.squeeze(images[0])
vector_size = np.product(img.shape)     # 28 * 28
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
plt.show()

# Define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # encoder
        # conv layer depth from 1 --> 16, 3X3 kernels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # conv layer depth from 16 --> 4, 3X3 kernels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1)
        # pooling layer to reduce the dim by 2; kernel and stride = 2
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # decoder
        # a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=2, stride=2)

    def forward(self, x):
        # encode
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        # decode
        # upsample, followed by conv layer, with relu activation
        # apply relu to all hidden layers except the output layer
        x = F.upsample(x, scale_factor=2, model='nearest')
        x = F.relu(self.conv4(x))
        # upsample again and apply sigmoid to output layer
        x = F.upsample(x, scale_factor=2, model='nearest')
        x = torch.sigmoid(self.t_conv2(x))
        return x

# initialize the NN
model = ConvAutoencoder()
print(model)

# specify loss function
criterion = nn.MSELoss()

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

## Training
# number of epochs to train the model
n_epochs = 3

for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

    ###################
    # train the model #
    ###################
    for data in train_loader:
        # _ stands in for labels, here
        # no need to flatten images
        images, _ = data
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        # calculate the loss
        loss = criterion(outputs, images)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * images.size(0)

    # print avg training statistics
    train_loss = train_loss / len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

# Checking out the results
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()

# get sample outputs
output = model(images)
# prep images for display
images = images.numpy()

# output is resized into a batch of iages
output = output.view(batch_size, 1, 28, 28)
# use detach when it's an output that requires_grad
output = output.detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

# input images on top row, reconstructions on bottom
for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)