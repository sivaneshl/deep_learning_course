import os
import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim


# check if cuda is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU...')
else:
    print('CUDA is available. Training on GPU...')

data_dir = 'flowers/'

# classes are folders in each directory with these names
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

## Transforming the data
# load and transform the data using ImageFolder
# VGG expects the images to be 224X224 input, so we resize all of them
data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.ToTensor()])

flower_data = datasets.ImageFolder(data_dir, transform=data_transform)
num_data = len(flower_data)
indices = list(range(num_data))
np.random.shuffle(indices)
split = int(np.floor(0.2 * num_data))
train_idx, test_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)

## DataLoaders and visualization
# define data loader parameters
batch_size = 20
num_workers = 0

# prepare data loaders
train_loader = torch.utils.data.DataLoader(flower_data, sampler=train_sampler,
                                           batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(flower_data, sampler=test_sampler,
                                          batch_size=batch_size, num_workers=num_workers)

# Visualize sample data
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy()     # convert images to numpy for display

# plot the images in the batch along with their labels
fig = plt.figure(figsize=(25, 4))
for idx in range(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]])
plt.show()

## Define the model
# load the pre-trained VGG16 model
# freeze all the parameters, so that the net acts as a fixed feature extractor
# remove the last layer
# replace the last layer with a linear classifier of our own

# load the pre-trained model from pytorch
vgg16 = models.vgg16(pretrained=True)
print(vgg16)
print(vgg16.classifier[6].in_features)
print(vgg16.classifier[6].out_features)


# freeze training for all "features" layers
for param in vgg16.features.parameters():
    param.requires_grad = False

## DONE: add a last linear layer  that maps n_inputs -> 5 flower classes
## new layers automatically have requires_grad = True
last_layer = nn.Linear(vgg16.classifier[6].in_features, len(classes))
vgg16.classifier[6] = last_layer
print(vgg16)

if train_on_gpu:
    vgg16.cuda()


# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
# specify optimizer (stochastic gradient descent) and learning rate = 0.001
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001)

## Training
# number of epochs to train the model
n_epochs = 2

for e in range(n_epochs):
    train_loss = 0.0

    # train the model
    vgg16.train()
    for batch_i, (data, target) in enumerate(train_loader):
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        # clear the gradients of all optimizer values
        optimizer.zero_grad()
        # forward pass
        output = vgg16(data)
        # calculate loss
        loss = criterion(output, target)
        # backward pass
        loss.backward()
        # optimizer step
        optimizer.step()
        # update the average training loss
        train_loss += loss.item()

        if batch_i % 20 == 19:
            print('Epoch %d, Batch %d Loss: %.16f' % (e, batch_i+1, train_loss/20))
            train_loss = 0.0

    # print training/validation statistics
    print('Epoch {} \t Training Loss: {:.6f}'.format(e, train_loss))

## Testing
test_loss = 0.0
class_correct = list(0. for i in range(5))
class_total = list(0. for i in range(5))

vgg16.eval()

# iterate over test data
for data, target in test_loader:
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()

    # forward pass
    output = vgg16(data)
    # calculate loss
    loss = criterion(output, target)
    # update loss
    test_loss += loss.item() * data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions with true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate the test accuracy of each class object
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(5):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

# Visualize sample results
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.numpy()

if train_on_gpu:
    images, labels = images.cuda(), labels.cuda()

# get sample output
output = vgg16(images)
# convert the output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())


# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))



