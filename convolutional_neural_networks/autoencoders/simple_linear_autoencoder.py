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

## Define the autoencoder
# The input images will be flattened into 784 length vectors. The targets are the same as the inputs.
# The encoder and decoder will be made of two linear layers, each. The depth dimensions should change as follows:
# 784 inputs > encoding_dim > 784 outputs.
# All layers will have ReLu activations applied except for the final output layer, which has a sigmoid activation.
class AutoEncoder(nn.Module):
    def __init__(self, encoding_dim):
        super(AutoEncoder, self).__init__()
        ## encoder
        self.encoder = nn.Linear(vector_size, encoding_dim)
        ## decoder
        self.decoder = nn.Linear(encoding_dim, vector_size)

    def forward(self, x):
        # define feed forward behaviour and scale the output with sigmoid activation function
        x = F.relu(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))
        return x


# initialize the NN
encoding_dim = 32
model = AutoEncoder(encoding_dim)
print(model)

# Loss function and optimizer
criterion = nn.MSELoss()        # useful for comparing pixel quantities rather than class probabilities
optimizer = optim.Adam(model.parameters(), lr=0.001)

# number of epochs to train the model
n_epochs = 20

for e in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    # training loop
    for data in train_loader:
        # _ stands for labels here
        images, _ = data
        # flatten images
        images = images.view(images.size(0), -1)
        # clear the gradients for optimized variables
        optimizer.zero_grad()
        # forward pass
        output = model(images)
        # calculate loss
        loss = criterion(output, images)        # the loss is to compare the original image and reconstructed image
        # backward pass
        loss.backward()
        # optimizer step
        optimizer.step()
        # update training loss
        train_loss += loss.item() * images.size(0)

    # print avg training statistics
    train_loss = train_loss / len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(e, train_loss))

torch.save(model.state_dict(), 'simple_linear_autoencoder.pt')
model.load_state_dict(torch.load('simple_linear_autoencoder.pt'))

## check out the results
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()

images_flatten = images.view(images.size(0), -1)
# get sample outputs
output = model(images_flatten)
# prep images for display
images = images.numpy()

# output is resized into a batch of images
output = output.view(batch_size, 1, 28, 28)
# use detach when it's an output that requires_grad
output = output.detach().numpy()

# plot the first 10 images and the reconstructed images
fig, axs = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))
for images, row in zip([images, output], axs):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()
