import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


img_path = 'data/udacity_sdc.png'

# load color image
bgr_image = cv2.imread(img_path)
# convert to grayscale
gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

# normalize, rescale entries to lie on [0, 1]
gray_image = gray_image.astype("float32")/255

# plot image
plt.imshow(gray_image, cmap='gray')
plt.show()

# Define and visualize the filters
filter_vals = np.array([[-1, -1, 1, 1],
                        [-1, -1, 1, 1],
                        [-1, -1, 1, 1],
                        [-1, -1, 1, 1]])
print('Filter shape: ', filter_vals.shape)

# Defining four filters all of which are linear combination of filter_vals above
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])

print('Filter 1: \n', filter_1)
print('Filter 2: \n', filter_2)
print('Filter 3: \n', filter_3)
print('Filter 4: \n', filter_4)

# Visualize all 4 filters
fig = plt.figure(figsize=(10, 5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter {}'.format(str(i+1)))
    width, height = filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(filters[i][x][y]),
                        xy = (y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if filters[i][x][y] < 0 else 'black')
plt.show()

# Define a convolutional layer
# Initialize a single convolutional layer so that it contains all your created filters.
# Note that you are not training this network; you are initializing the weights in a convolutional layer
# so that you can visualize what happens after a forward pass through this network!
# To define a neural network in PyTorch, you define the layers of a model in the function __init__ and define the
# forward behavior of a network that applyies those initialized layers to an input (x) in the function forward.
class Net(nn.Module):
    def __init__(self, weight):
        super(Net, self).__init__()
        # initialize the weights of the convolutional layer to be weights of the 4 defined filters
        k_height, k_width = weight.shape[2:]
        # defines the convolutional layer, assumes there are 4 grayscale filters
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv = torch.nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        # calculates the output of the convolution layer
        # pre and post activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)

        # returns both layers
        return conv_x, activated_x


# instantiate the model and set the weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)
print(model)

# Visualize the output of each filter
def viz_layer(layer, n_filters=4):
    fig = plt.figure(figsize=(20, 20))
    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i+1, xticks=[], yticks=[])
        # grab output layers
        ax.imshow(np.squeeze(layer[0, i].data.numpy()), cmap='gray')
        ax.set_title('Output {}'.format(str(i+1)))
    plt.show()

# plot original image
plt.imshow(gray_image, cmap='gray')
plt.show()

# visualize all filters
fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(left=0, right=1.5, bottom=0.8,  top=1, hspace=0.05, wspace=0.05)
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))
plt.show()

# convert the image to an input tensor
gray_image_tensor = torch.from_numpy(gray_image).unsqueeze(0).unsqueeze(1)

# get the convolutional layer (pre and post activation)
conv_layer, output_layer = model(gray_image_tensor)

# visualize the output of a conv layer
viz_layer(conv_layer)

# visualize the output after relu activation
viz_layer(output_layer)


