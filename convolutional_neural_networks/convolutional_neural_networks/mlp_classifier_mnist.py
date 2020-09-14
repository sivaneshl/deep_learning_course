import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler

# number of sub-processes to use for our data loading
num_workers = 0
# how many samples to load per batch
batch_size = 20
# percentage of the training set to use as validation set
valid_size = 0.2

# convert the data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and testing datasets
train_data = datasets.MNIST('C:\\deep_learning\\deep_learning_course\\neural_networks\\deep_learning_with_pytorch\\MNIST_data\\',
                            download=True, train=True, transform=transform)
test_data = datasets.MNIST('C:\\deep_learning\\deep_learning_course\\neural_networks\\deep_learning_with_pytorch\\MNIST_data\\',
                           download=True, train=False, transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

## Visualising the data
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy()

# plot the images in the batch along with the coresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    # print out the correct label for each image
    ax.set_title(str(labels[idx].item()))   # .item() gets the value contained in the tensor
plt.show()


## View an image in more detail
img = np.squeeze(images[1])
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
tresh = img.max() / 2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y], 2) if img[x][y] != 0 else 0
        ax.annotate(str(val),
                    xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<tresh else 'black')
plt.show()

## Define the network architecture
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # flatten the input
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x

model = Network()
print(model)

## Specify Loss function and Optimizer
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

## Train the network
# The steps for training/learning from a batch of data are described in the comments below:
# Clear the gradients of all optimized variables
# Forward pass: compute predicted outputs by passing inputs to the model
# Calculate the loss
# Backward pass: compute gradient of the loss with respect to model parameters
# Perform a single optimization step (parameter update)
# Update average training loss

# number of epochs
epochs = 30
# Initialize tracker for minimum validation loss
valid_loss_min = np.Inf     # set initial min to infinity

model.train()


for e in range(epochs):
    # monitoring training loss and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    # Training loop
    for data, target in train_loader:
        # clear the gradients for all optimized variables
        optimizer.zero_grad()
        # forward pass : compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass : compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimizer step
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)

    # Validation loop
    for data, target in valid_loader:
        # forward pass : compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update running validation loss
        valid_loss += loss.item() * data.size(0)

    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)

    # print the training statistics
    print('Epoch: {} \tTraining loss: {:.6f} \tValidation loss: {:.6f}'
          .format(e+1, train_loss, valid_loss))

    # save the model if the validation loss is minimum
    if valid_loss <= valid_loss_min:
        print('Validation loss decreases ({:.6f}) --> ({:.6f}). Saving model ...'
              .format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'mnist_model.pth')
        valid_loss_min = valid_loss


## Testing the network

model.load_state_dict(torch.load('mnist_model.pth'))

# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()

for data, target in test_loader:
    # forward pass : compute predicted output by passing the inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update the test loss
    test_loss += loss.item()
    # convert the output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare the predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each class object
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print the average test loss
print('Test Loss: {:.6f}\n'.format(test_loss / len(test_loader)))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (class_total[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

## View sample test results
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()

# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, pred = torch.max(output, 1)
# prep images for display
images = images.numpy()

# plot the images in the batch along with the predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title('{} {}'.format(str(pred[idx].item()), str(labels[idx].item())),
                 color=('green' if pred[idx]==labels[idx] else 'red'))
plt.show()



