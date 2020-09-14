# style transfer using vgg16 trained net as a feature extractor

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models

# get the features portion of thr vgg19 model ( we will not need the classifier portion)
vgg = models.vgg19(pretrained=True).features

# freeze all vgg parameters as we are only optimizing the target image
for param in vgg.parameters():
    param.requires_grad = False

# move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg.to(device)


## load content and style image
def load_image(img_path, max_size=400, shape=None):
    """
    Load in and transform an image
    making sure the image is <= 400 pixels in the x-y dims
    :param img_path:
    :param max_size:
    :param shape:
    :return:
    """

    image = Image.open(img_path).convert('RGB')

    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([transforms.Resize(size),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # discard the transparent, alpha channel (i.e., :3) and add the batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image


# load the content and the style image
content = load_image('images/octopus.jpg').to(device)
# resize style to match content, makes code easier
style = load_image('images/hockney.jpg', shape=content.shape[-2:]).to(device)


# helper function for unnormalizing the image and converting it from a Tensor image to a Numpy image
def im_convert(tensor):
    """
    display a tensor as an image
    :param tensor:
    :return:
    """
    image = tensor.to('cpu').clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


# Display the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(style))
plt.show()


# content and style features
def get_features(image, model, layers=None):
    """
    Run an image forward through a model and get the features for a set of layers
    :param image:
    :param model:
    :param layers:
    :return:
    """

    if layers is None:
        layers = {'0': 'conv1_1',   # style
                  '5': 'conv2_1',   # style
                  '10': 'conv3_1',  # style
                  '19': 'conv4_1',  # style
                  '21': 'conv4_2',  # content
                  '28': 'conv5_1'}  # style

    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


## Gram matrix
# the output of every convilutional layer is a tensor with 4 D - batch size, depth (d), height (h) and width (w)_
# to calculate the gram matrix
# - get the depth, height and width from the tensor.size()
# - reshape the tensor so that the spatial dimensions are flattened
# - calculate the gram matrix by multiplying the reshaped tensor by its transpose
def gram_matrix(tensor):
    """
    calculate the gram matrix of a given tensor
    :param tensor:
    :return:
    """
    b, d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram


# get the features from the content and style images
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calculate the gram matrix for each layer of our style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# create a third 'target' image and prep it for change
# its a good idea to start of with the target image as a copy of our content image and then iteratively change its style
target = content.clone().requires_grad_(True).to(device)

# weights for style layers
# weighting earlier layers more will results in 'larger' style artifacts
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}

content_weight = 1  # alpha
style_weight = 1e6  # beta

## define content, style and total losses
# for displaying the target image intermittently
show_every = 400

# iteration hyperparameters
optimizer = optim.Adam([target], lr=0.003)
steps = 2000 # decide how many iterations to update your image

for ii in range(1, steps+1):

    print('Running: ', ii)

    # get the features from target image
    target_features = get_features(target, vgg)
    # calculate the content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    # style loss
    style_loss = 0
    # iterate through each style layer and add to the style loss
    for layer in style_weights:
        # get the target style representation of the layer
        target_feature = target_features[layer]
        _, d, h, w = target_feature.shape

        # calculate the target gram matrix
        target_gram = gram_matrix(target_feature)

        # get the 'style' style representation
        style_gram = style_grams[layer]

        # calculate the style loss for one layer weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)

        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)

    # calculate the total loss
    total_loss = (content_weight * content_loss) + (style_weight * style_loss)

    # update the target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # display intermediate images and print the loss
    if ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(im_convert(target))
        plt.show()


# display content and final, target image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(target))


