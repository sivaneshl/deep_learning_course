# for loading in and transforming data
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# for visualizing data
import matplotlib.pyplot as plt
import numpy as np
import warnings

# for creating the neural network
import torch.nn as nn
import torch.nn.functional as F

# for training
import torch.optim as optim

# for saving model
from helpers import save_samples, checkpoint


# Dataloaders
def get_data_loader(image_type, image_dir='summer2winter_yosemite',
                    image_size=128, batch_size=16, num_workers=0):
    """
    Returns training and test DataLoaders for a given image type, either "summer" or "winter".
    These images will be re-sized to 128X128X3, by default, converted into tensors and normalized.
    :param image_type:
    :param image_dir:
    :param image_size:
    :param batch_size:
    :param num_workers:
    :return:
    """
    # resize and normalize images
    transform = transforms.Compose([transforms.Resize(image_size),  # resize to 128X128
                                    transforms.ToTensor()])

    # get training and test directories
    image_path = './' + image_dir
    train_path = os.path.join(image_path, image_type)
    test_path = os.path.join(image_path, 'test_{}'.format(image_type))

    # define datasets using ImageFolder
    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    # create and return DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader


# Create train and test DataLoaders for the images from the two domains X and Y
# image_type = directory names of our data
dataloader_X, test_dataloader_X = get_data_loader(image_type="summer")
dataloader_Y, test_dataloader_Y = get_data_loader(image_type="winter")


# Display some training images
# helper imshow function
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some images from X
dataiter = iter(dataloader_X)
images, _ = next(dataiter)

# show images
fig = plt.figure(figsize=(12, 8))
imshow(torchvision.utils.make_grid(images))

# get some images from Y
dataiter = iter(dataloader_Y)
images, _ = next(dataiter)

# show images
fig = plt.figure(figsize=(12, 8))
imshow(torchvision.utils.make_grid(images))

# Pre-processing : scaling from -1 to 1
# current image
img = images[0]
print('Min: ', img.min())
print('Max: ', img.max())


# helper scale function
def scale(x, feature_range=(-1, 1)):
    """
    scale takes in an image x and returns that image, scaled with the feature_range of pixel
    values from -1 to 1. The function assumes that the input X is already scaled from 0-255.
    :param x:
    :param feature_range:
    :return:
    """
    # scale from 0-1 to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x


# scaled image
scaled_img = scale(img)
print('Scaled Min: ', scaled_img.min())
print('Scaled Max: ', scaled_img.max())


# Defining the model
# Discriminator

# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """
    Creates a convolutional layer with optional batch normalization.
    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :param batch_norm:
    :return:
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()

        # define all convolutional layers
        # should accept an rgb image as input and a single value as output
        self.conv_dim = conv_dim

        # 128 X 128 X 3 --> 64 X 64 X 64
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False)
        # 64 X 64 X 64 --> 32 X 32 X 128
        self.conv2 = conv(conv_dim, conv_dim * 2, 4, batch_norm=True)
        # 32 X 32 X 128 --> 16 X 16 X 256
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4, batch_norm=True)
        # 16 X 16 X 256 --> 8 X 8 X 512
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4, batch_norm=True)
        # 8 X 8 X 512 --> 1 X 1 X 1 (classification layer)
        self.conv5 = conv(conv_dim * 8, 1, 4, stride=1, batch_norm=False)

    def forward(self, x):
        # define the feed forward behavior
        # relu applied for all conv layers but last
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # classification layer
        x = self.conv5(x)
        return x


# Generator
# Residual blocks class
class ResidualBlock(nn.Module):
    """
    Defines a Residual block.
    This adds an input X to a convolutional layer (applied to x) with the same size input and output.
    These blocks allow a model to learn an effective transformation from one domain to another.
    """

    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        # conv_dim = number of inputs

        # define two convolutional layers + batch normalization that will act as our residual function F(x)
        # layer should have the same shape input as output - suggested kernel_size = 3

        self.conv1 = conv(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, batch_norm=True)
        self.conv2 = conv(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, batch_norm=True)

    def forward(self, x):
        """
        Apply a Relu activation to the outputs of the first layer.
        Return a summed output x + resnet_block(x)
        :param x:
        :return:
        """
        res_out = F.relu(self.conv1(x))
        res_out = self.conv2(res_out)
        return x + res_out


# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """
    Creates a transpose convolutional layer with optional batch normalization.
    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :param batch_norm:
    :return:
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


# define the generator architecture
class CycleGenerator(nn.Module):
    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(CycleGenerator, self).__init__()

        self.conv_dim = conv_dim
        # define the encoder part of the generator
        # 128 X 128 X 3 --> 64 X 64 X 64
        self.conv1 = conv(3, conv_dim, 4, batch_norm=True)
        # 64 X 64 X 64 --> 32 X 32 X 128
        self.conv2 = conv(conv_dim, conv_dim * 2, 4, batch_norm=True)
        # 32 X 32 X 128 --> 16 X 16 X 256
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4, batch_norm=True)

        # define the resnet part of the generator
        res_layers = []
        for layer in range(n_res_blocks):
            res_layers.append(ResidualBlock(conv_dim * 4))
        self.res_blocks = nn.Sequential(*res_layers)

        # define the decoder part of the generator
        # 16 X 16 X 256 --> 32 X 32 X 128
        self.deconv1 = deconv(conv_dim * 4, conv_dim * 2, 4, batch_norm=True)
        # 32 X 32 X 128 --> 64 X 64 X 64
        self.deconv2 = deconv(conv_dim * 2, conv_dim, 4, batch_norm=True)
        # 64 X 64 X 64 --> 128 X 128 X 3
        self.deconv3 = deconv(conv_dim, 3, 4, batch_norm=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.res_blocks(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.tanh(self.deconv3(x))
        return x


# Create the complete network
def create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
    """
    Builds the generator and discriminator
    :param g_conv_dim:
    :param d_conv_dim:
    :param n_res_blocks:
    :return:
    """
    # Instantiate generators
    G_XtoY = CycleGenerator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)
    G_YtoX = CycleGenerator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)
    # Instantiate discriminators
    D_X = Discriminator(conv_dim=d_conv_dim)
    D_Y = Discriminator(conv_dim=d_conv_dim)

    # move models to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU')
    else:
        print('Only CPU available ')

    return G_XtoY, G_YtoX, D_X, D_Y


G_XtoY, G_YtoX, D_X, D_Y = create_model()


# helper function to print model architecture
def print_models(G_XtoY, G_YtoX, D_X, D_Y):
    """
    Prints model information for generators and discriminators
    :param G_XtoY:
    :param G_YtoX:
    :param D_X:
    :param D_Y:
    :return:
    """
    print("                     G_XtoY                    ")
    print("-----------------------------------------------")
    print(G_XtoY)
    print()

    print("                     G_YtoX                    ")
    print("-----------------------------------------------")
    print(G_YtoX)
    print()

    print("                      D_X                      ")
    print("-----------------------------------------------")
    print(D_X)
    print()

    print("                      D_Y                      ")
    print("-----------------------------------------------")
    print(D_Y)
    print()


# print all of the models
print_models(G_XtoY, G_YtoX, D_X, D_Y)


# Define Genarator and Discriminator losses
def real_mse_loss(D_out):
    # how close is the produced output from being "real"?
    return torch.mean((D_out - 1) ** 2)


def fake_mse_loss(D_out):
    # how close is the produced output from being "false"?
    return torch.mean((D_out - 0) ** 2)


def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):
    # calculate reconstruction loss
    reconstr_loss = torch.mean(torch.abs(real_im - reconstructed_im))
    # return weighted loss
    return lambda_weight * reconstr_loss


# Define the optimizers
# hyperparams for Adam optimizers
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters

# Create optimizers for the generators and discriminators
g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
d_x_optimizer = optim.Adam(D_X.parameters(), lr, [beta1, beta2])
d_y_optimizer = optim.Adam(D_Y.parameters(), lr, [beta1, beta2])


# Train the network
def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y,
                  n_epochs=1000):
    print_every = 10

    # keep track of losses over time
    losses = []

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = test_iter_X.next()[0]
    fixed_Y = test_iter_Y.next()[0]
    fixed_X = scale(fixed_X)  # make sure to scale to a range -1 to 1
    fixed_Y = scale(fixed_Y)

    # batches per epoch
    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)
    batches_per_epoch = min(len(iter_X), len(iter_Y))

    for epoch in range(1, n_epochs + 1):

        # Reset iterators for each epoch
        if epoch % batches_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X, _ = iter_X.next()
        images_X = scale(images_X)  # make sure to scale to a range -1 to 1

        images_Y, _ = iter_Y.next()
        images_Y = scale(images_Y)

        # move images to GPU if available (otherwise stay on CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images_X = images_X.to(device)
        images_Y = images_Y.to(device)

        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        ##   First: D_X, real and fake loss components   ##

        # 1. Compute the discriminator losses on real images
        d_x_optimizer.zero_grad()
        out_x = D_X(images_X)
        D_X_real_loss = real_mse_loss(out_x)

        # 2. Generate fake images that look like domain X based on real images in domain Y
        fake_x = G_YtoX(images_Y)

        # 3. Compute the fake loss for D_X
        out_x = D_X(fake_x)
        D_X_fake_loss = fake_mse_loss(out_x)

        # 4. Compute the total loss and perform backprop
        d_x_loss = D_X_real_loss + D_X_fake_loss
        d_x_loss.backward()
        d_x_optimizer.step()

        ##   Second: D_Y, real and fake loss components   ##

        # 1. Compute the discriminator losses on real images
        d_y_optimizer.zero_grad()
        out_y = D_Y(images_Y)
        D_Y_real_loss = real_mse_loss(out_y)

        # 2. Generate fake images that look like domain Y based on real images in domain X
        fake_y = G_XtoY(images_X)

        # 3. Compute the fake loss for D_Y
        out_y = D_Y(fake_y)
        D_Y_fake_loss = fake_mse_loss(out_y)

        # 4. Compute the total loss and perform backprop
        d_y_loss = D_Y_real_loss + D_Y_fake_loss
        d_y_loss.backward()
        d_y_optimizer.step()

        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================

        ##    First: generate fake X images and reconstructed Y images    ##
        g_optimizer.zero_grad()
        # 1. Generate fake images that look like domain X based on real images in domain Y
        fake_x = G_YtoX(images_Y)

        # 2. Compute the generator loss based on domain X
        out_x = D_X(fake_x)
        g_YtoX_loss = real_mse_loss(out_x)

        # 3. Create a reconstructed y
        reconstructed_Y = G_XtoY(fake_x)

        # 4. Compute the cycle consistency loss (the reconstruction loss)
        reconstructed_Y_loss = cycle_consistency_loss(images_Y, reconstructed_Y, lambda_weight=10)

        ##    Second: generate fake Y images and reconstructed X images    ##

        # 1. Generate fake images that look like domain Y based on real images in domain X
        fake_y = G_XtoY(images_X)

        # 2. Compute the generator loss based on domain X
        out_y = D_Y(fake_y)
        g_XtoY_loss = real_mse_loss(out_y)

        # 3. Create a reconstructed x
        reconstructed_X = G_YtoX(fake_y)

        # 4. Compute the cycle consistency loss (the reconstruction loss)
        reconstructed_X_loss = cycle_consistency_loss(images_X, reconstructed_X, lambda_weight=10)

        # 5. Add up all generator and reconstructed losses and perform backprop
        g_total_loss = g_YtoX_loss + g_XtoY_loss + reconstructed_Y_loss + reconstructed_X_loss
        g_total_loss.backward()
        g_optimizer.step()

        # Print the log info
        if epoch % print_every == 0:
            # append real and fake discriminator losses and the generator loss
            losses.append((d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
            print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(
                epoch, n_epochs, d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))

        sample_every = 100
        # Save the generated samples
        if epoch % sample_every == 0:
            G_YtoX.eval()  # set generators to eval mode for sample generation
            G_XtoY.eval()
            save_samples(epoch, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=16)
            G_YtoX.train()
            G_XtoY.train()

            # uncomment these lines, if you want to save your model
            checkpoint_every = 1000
            # Save the model parameters
            if epoch % checkpoint_every == 0:
                checkpoint(epoch, G_XtoY, G_YtoX, D_X, D_Y)

    return losses


n_epochs = 1000  # keep this small when testing if a model first works, then increase it to >=1000

losses = training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, n_epochs=n_epochs)

fig, ax = plt.subplots(figsize=(12,8))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator, X', alpha=0.5)
plt.plot(losses.T[1], label='Discriminator, Y', alpha=0.5)
plt.plot(losses.T[2], label='Generators', alpha=0.5)
plt.title("Training Losses")
plt.legend()

import matplotlib.image as mpimg


# helper visualization code
def view_samples(iteration, sample_dir='samples_cyclegan'):
    # samples are named by iteration
    path_XtoY = os.path.join(sample_dir, 'sample-{:06d}-X-Y.png'.format(iteration))
    path_YtoX = os.path.join(sample_dir, 'sample-{:06d}-Y-X.png'.format(iteration))

    # read in those samples
    try:
        x2y = mpimg.imread(path_XtoY)
        y2x = mpimg.imread(path_YtoX)
    except:
        print('Invalid number of iterations.')

    fig, (ax1, ax2) = plt.subplots(figsize=(18, 20), nrows=2, ncols=1, sharey=True, sharex=True)
    ax1.imshow(x2y)
    ax1.set_title('X to Y')
    ax2.imshow(y2x)
    ax2.set_title('Y to X')

# view samples at iteration 100
view_samples(100, 'samples_cyclegan')

# view samples at iteration 1000
view_samples(1000, 'samples_cyclegan')
