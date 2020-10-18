import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Tensor transform
transform = transforms.ToTensor()

# SVHN training sets
svhn_train = datasets.SVHN(root='data/', split='train', download=True, transform=transform)

batch_size = 128
num_workers = 0

# build dataloaders for SVHN dataset
train_loader = DataLoader(svhn_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Visualise the data
dataiter = iter(train_loader)
images, labels = next(dataiter)
# plot the images in the batch along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
plot_size = 20
for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size / 2, idx + 1, xticks=[], yticks=[])
    ax.imshow(np.transpose(images[idx], (1, 2, 0)))
    # print out the correct label for each image
    # .item() gets the value contained in the tensor
    ax.set_title(str(labels[idx].item()))
plt.show()

# Preprocessing : scaling from -1 to 1
# current range
img = images[0]
print('Min: ', img.min())
print('Max: ', img.max())


# helper scale function
def scale(x, feature_range=(-1, 1)):
    """
    scale takes in an image x and returns that image scaled with a feature range of pixel values from -1 to 1.
    This function assumes that the input x is already scaled from 0 - 1
    :param x:
    :param feature_range:
    :return:
    """
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min, max = feature_range
    x = x * (max - min) + min
    return x


# testing the scale function
scaled_img = scale(img)
print('Scaled min: ', scaled_img.min())
print('Scaled max: ', scaled_img.max())


# Define the model
# Discriminator

# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """
    Creates a convolutional layer with optional batch normalization
    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :param batch_norm:
    :return:
    """
    # a kernel size 4, stride 2 and padding 1 will downsize by half

    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    # append conv_layer
    layers.append(conv_layer)

    # add the optional batch_norm
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    # using Sequential container
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, conv_dim=32):
        super(Discriminator, self).__init__()

        self.conv_dim = conv_dim

        # 32 X 32 X 3
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False)
        # 16 X 16 X 32
        self.conv2 = conv(conv_dim, conv_dim * 2, 4, batch_norm=True)
        # 8 X 8 X 64
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4, batch_norm=True)
        # 4 X 4 X 128
        self.fc1 = nn.Linear(conv_dim * 4 * 4 * 4, 1)

    def forward(self, x):
        # all conv layers + leaky relu activation with negative slope 0.2
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        # flatten
        x = x.view(-1, self.conv_dim * 4 * 4 * 4)
        # last fully connected layer
        x = self.fc1(x)
        return x


# Generator
# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """
    Creates a transpose convolutional layer with optional batch normalization
    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :param batch_norm:
    :return:
    """
    # create a sequence of transpose + optional batch norm layers

    layers = []

    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels,
                                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    layers.append(transpose_conv_layer)

    # add the optional batch_norm
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    # using Sequential container
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, z_size, conv_dim=32):
        super(Generator, self).__init__()

        self.conv_dim = conv_dim

        # fully connected layer --> z_size to conv_dim X 4 X 4 X 4
        self.fc1 = nn.Linear(z_size, conv_dim * 4 * 4 * 4)

        # transpose conv layers
        # 4 x 4 x 128
        self.t_conv1 = deconv(conv_dim * 4, conv_dim * 2, 4)
        # 8 x 8 x 64
        self.t_conv2 = deconv(conv_dim * 2, conv_dim, 4)
        # 16 x 16 x 32
        self.t_conv3 = deconv(conv_dim, 3, 4, batch_norm=False)
        # 32 x 32 x 3

    def forward(self, x):
        # fully connected layer + reshape
        x = self.fc1(x)
        x = x.view(-1, self.conv_dim * 4, 4, 4)

        # hidden transpose layers with relu
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))

        # last layer + tanh
        x = self.t_conv3(x)
        x = F.tanh(x)

        return x


# Build the network
# define hyper parameters
conv_dim = 32
z_size = 100

# define discriminator and generator
D = Discriminator(conv_dim)
G = Generator(z_size=z_size, conv_dim=conv_dim)

print(D)
print(G)

# training on GPU
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    G.cuda()
    D.cuda()
    print('GPU available for training. Modules moved to GPU')
else:
    print('Training on CPU')


def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)

    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size) * 0.9
    else:
        # real labels = 1
        labels = torch.ones(batch_size)

    # move labels to GPU if available
    if train_on_gpu:
        labels = labels.cuda()

    # binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    batch_size = D_out.size(0)

    # fake labels = 0
    labels = torch.zeros(batch_size)

    if train_on_gpu:
        labels = labels.cuda()

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss


# hyper parameters
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

# Optimizers
d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])

# Discriminator training¶
# Compute the discriminator loss on real, training images
# Generate fake images
# Compute the discriminator loss on fake, generated images
# Add up real and fake loss
# Perform backpropagation + an optimization step to update the discriminator's weights
# Generator training
# Generate fake images
# Compute the discriminator loss on fake images, using flipped labels!
# Perform backpropagation + an optimization step to update the generator's weights

# training hyperparameters
num_epochs = 30

# keep track of loss and generated fake images
samples = []
losses = []

print_every = 300

# Get some fixed data for sampling. These are images that are held constant throughout training
# and allow us to inspect the model's performance
sample_size = 16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()

# train the network
for e in range(num_epochs):
    for batch_i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)

        # important rescaling step
        real_images = scale(real_images)

        ### train the discriminator
        d_optimizer.zero_grad()

        # 1. Train with real images
        if train_on_gpu:
            real_images = real_images.cuda()

        # compute the discriminator loss on real images
        D_real = D(real_images)
        d_real_loss = real_loss(D_real)

        # 2. Train with fake images
        # generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        if train_on_gpu:
            z = z.cuda()
        fake_images = G(z)

        # compute the discriminator loss on fake images
        D_fake = D(fake_images)
        d_fake_loss = fake_loss(D_fake)

        # add up losses and perform back prop
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        ### Train the generator
        g_optimizer.zero_grad()

        # 1. Train with fake images and flipped labels
        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        if train_on_gpu:
            z = z.cuda()
        fake_images = G(z)

        # Compute the discriminator loss on fake images using flipped labels
        D_fake = D(fake_images)
        g_loss = real_loss(D_fake)

        # perform back prop
        g_loss.backward()
        g_optimizer.step()

        # print some loss stats
        if batch_i % print_every == 0:
            # append discriminator loss and genaretor loss
            losses.append((d_loss.item(), g_loss.item()))
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                e + 1, num_epochs, d_loss.item(), g_loss.item()))

    # after each epoch
    # genarate and save sample image
    G.eval()  # for generating samples
    if train_on_gpu:
        fixed_z = fixed_z.cuda()
    sample_z = G(fixed_z)
    samples.append(sample_z)
    G.train()  # back to training

# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)

# training loss
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()
plt.show()


# Generator samples from training
# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16, 4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1) * 255 / (2)).astype(np.uint8)  # rescale to pixel range (0-255)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32, 32, 3)))
    plt.show()


view_samples(-1, samples)

