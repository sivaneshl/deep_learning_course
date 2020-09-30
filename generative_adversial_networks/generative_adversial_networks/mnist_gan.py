import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import pickle as pkl

# number of sub processes to use for data loading
num_workers = 0

# how many samples per batch to load
batch_size = 64

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# get the training data
train_data = datasets.MNIST('C:\\deep_learning\\deep_learning_course\\neural_networks\\deep_learning_with_pytorch\\MNIST_data',
                            train=True, download=True, transform=transform)

# prepare data loader
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)

# visualise the data
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy()

# get one image from the batch
img = np.squeeze(images[0])

# plot it
fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
plt.show()

## Define the model
# GAN --> a discriminator and a generator
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()

        # define class variables
        self.input_size = input_size
        self.output_size = output_size

        # define all layers
        self.fc1 = nn.Linear(input_size, hidden_dim*4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # flatten image
        x = x.view(-1, self.input_size)

        # pass x through all layers
        # apply leaky relu activation to all hidden layers
        x = F.leaky_relu(self.fc1(x), 0.2)  # negative slope = 0.2
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        x = self.output(x)

        return x

class Generator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()

        # define all layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        self.output = nn.Linear(hidden_dim*4, output_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # pass x through all layers
        # apply leaky relu activation to all hidden layers
        x = F.leaky_relu(self.fc1(x), 0.2)  # negative slope = 0.2
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        x = torch.tanh(self.output(x))

        return x


## Model hyperparameters
# Discriminator hyperparameters
# size of the input image to the discriminator
input_size = 28*28
# size of the discriminator output (real or fake)
d_output_size = 1
# size of the last hidden layer in the discriminator
d_hidden_dim = 32
# Generator hyperparameters
# size of the latent vector to give to the generator
z_size = 100
# size of the generator output
g_output_size = 28*28
# size of the first hidden layer in the generator
g_hidden_size = 32

## Build complete network
# instantiate discriminator and generator
D = Discriminator(input_size, d_hidden_dim, d_output_size)
G = Generator(z_size, g_hidden_size, g_output_size)
print(D)
print(G)

## Discriminator and Generator Losses
# Discriminator loss : d_loss = d_real_loss + d_fake_loss
# discriminator to classify the the real images with a label = 1 and fake images with a label = 0
def real_loss(D_out, smooth=False):
    # smooth the labels if needed
    batch_size = D_out.size(0)
    if smooth:
        labels = torch.ones(batch_size)*0.9     # real labels = 0.9
    else:
        labels = torch.ones(batch_size)         # real labels = 1

    # loss function
    criterion = nn.BCEWithLogitsLoss()

    # calculate the loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)    # fake labels = 0

    # loss function
    criterion = nn.BCEWithLogitsLoss()

    # calculate the loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

## Optimizers
lr = 0.002
d_optimizer = optim.Adam(D.parameters(), lr=lr)
g_optimizer = optim.Adam(G.parameters(), lr=lr)

## Training
# Training will involve alternating between training the discriminator and the generator
# Discriminator training
# 1. Compute the discriminator loss on real, training images
# 2. Generate fake images
# 3. Compute the discriminator loss on fake, generated images
# 4. Add up real and fake loss
# 5. Perform backpropagation + an optimization step to update the discriminator's weights
# Generator training
# 1. Generate fake images
# 2. Compute the discriminator loss on fake images, using flipped labels!
# 3. Perform backpropagation + an optimization step to update the generator's weights

# training hyperparameters
num_epochs = 40
print_every = 400

# keep track of loss and generated fake samples
samples = []
losses = []

# Get some fixed data for sampling. These are images that are held constant during training,
# and allow us to inspect the model's performance
sample_size = 16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()

# train the network
D.train()
G.train()
for e in range(num_epochs):

    for batch_i, (real_images, _) in enumerate(train_loader):

        batch_size = real_images.size(0)

        # important rescaling step
        real_images = real_images*2 - 1     # re-scale input images from (0,1) to (-1, 1)

        ## Train the Discriminator
        d_optimizer.zero_grad()

        # 1. Train with real images
        D_out = D(real_images)
        # compute the discriminator loss on real images using smoothed labels
        r_loss = real_loss(D_out, smooth=True)

        # 2. Train with fake images
        # Generate fake image
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_images = G(z)
        # compute the discriminator loss on fake images
        D_out = D(fake_images)
        f_loss = fake_loss(D_out)

        # add up real and fake losses and perform back propagation
        d_loss = r_loss + f_loss
        d_loss.backward()
        d_optimizer.step()

        ## Train the Generator
        # 1. Train with fake images and flipped labels
        # Generate fake image
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        fake_images = G(z)
        # compute the discriminator loss on fake images using flipped labels
        g_optimizer.zero_grad()
        D_out = D(fake_images)
        g_loss = real_loss(D_out)   # use real_loss to flip labels
        # perfrom back prop
        g_loss.backward()
        g_optimizer.step()

        # print some loss stats
        if batch_i % print_every == 0:
            # print dicriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                e+1, num_epochs, d_loss.item(), g_loss.item()
            ))

    # After each epoch
    # append discriminator loss and generator loss
    losses.append((d_loss.item(), g_loss.item()))

    # generate and save sample, fake images
    G.eval()
    samples_z = G(fixed_z)
    samples.append(samples_z)
    G.train()

# save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)

## Plot the training loss
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title('Training losses')
plt.legend()
plt.show()

## Generator samples from training
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach()
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        im = ax.imshow(img.reshape((28, 28)), cmap='Greys_r')

# load samples from generator taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)

# -1 indicates final epochs samples
view_samples(-1, samples)
plt.show()

# view the samples generated at every 10 epochs
rows = 10
cols = 6
fig, axes = plt.subplots(figsize=(7, 12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        img = img.detach()
        ax.imshow(img.reshape((28, 28)),cmap='Greys_r')
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
plt.show()

# Sampling from the generator - We just need to pass in a new latent vector z and we'll get new samples!
# randomly generated, new latent vectors
sample_size=16
rand_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
rand_z = torch.from_numpy(rand_z).float()

G.eval() # eval mode
# generated samples
rand_images = G(rand_z)

# 0 indicates the first set of samples in the passed in list
# and we only have one batch of samples, here
view_samples(0, [rand_images])



