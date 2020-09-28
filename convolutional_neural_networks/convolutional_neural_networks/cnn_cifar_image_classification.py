import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# check if cuda is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU...')
else:
    print('CUDA is available. Training on GPU...')

# Load the data
# number of sub processes to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use for validation
valid_size = 0.2

# Convert the data to a normalized torch.FloatTensor
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# chose the training and testing datasets
train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

# obtain the training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# specify the classes for the images
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Visualise a batch of training data
# helper function to un-normalize and visualize the image
def imshow(img):
    img = img / 2 + 0.5  # un-normalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert fro tensor image


# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy()  # convert images to numpy for display

# plot the images in the batch along with the labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in range(20):
    ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])
plt.show()

# View an image in more detail
rgb_image = np.squeeze(images[3])
channels = ['red_channel', 'green_channel', 'blue_channel']

fig = plt.figure(figsize=(36, 36))
for idx in np.arange(rgb_image.shape[0]):
    ax = fig.add_subplot(1, 3, idx + 1)
    img = rgb_image[idx]
    ax.imshow(img, cmap='gray')
    ax.set_title(channels[idx])
    width, height = img.shape
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y], 2) if img[x][y] != 0 else 0
            ax.annotate(str(val),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        size=8,
                        color='white' if img[x][y] < thresh else 'black')
plt.show()


# define the CNN arhitecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# create a complete CNN
model = Net()
print(model)

# move tensors to gpu if available
if train_on_gpu:
    model.cuda()

# specify loss function
criterion = nn.CrossEntropyLoss()
# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# train the model
n_epochs = 20
valid_loss_min = np.Inf

for e in range(1, n_epochs + 1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    # train the model
    model.train()
    for data, target in train_loader:
        # move tensors to gpu if available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        # clear the gradients for all optimized variables
        optimizer.zero_grad()

        # forward pass: compute predicted output by passing inputs to the model
        output = model(data)

        # calculate the batch loss
        loss = criterion(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimizer step
        optimizer.step()

        # update the average training loss
        train_loss += loss.item() * data.size(0)


    # validate the model
    model.eval()
    for data, target in valid_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        # forward pass
        output = model(data)
        # calculate batch loss
        loss = criterion(output, target)
        # update the average validation loss
        valid_loss += loss.item() * data.size(0)

    # calculate the average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)

    # print training/validation statistics
    print('Epoch {} \t Training Loss: {:.6f} \t Validation Loss: {:.6f}'.format(e, train_loss, valid_loss))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased {:.6f} --> {:.6f}. Saving model...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'cifar_model.pt')
        valid_loss_min = valid_loss


# load the model
trained_model = Net()
trained_model.load_state_dict(torch.load('cifar_model.pt'))
print(trained_model)

# Test the trained network
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

trained_model.eval()
# iterate over test data
for data, target in test_loader:
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()

    # forward pass
    output = trained_model(data)
    # calculate loss
    loss = criterion(output, target)
    # update test loss
    test_loss += loss.item() * data.size(0)
    # convert output probablitis to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate the test accuracy of each class object
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

# visualize sample test results
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = next(dataiter)
images.numpy()

if train_on_gpu:
    images, labels = images.cuda(), labels.cuda()

# get sample outputs
output = trained_model(images)
# convert the output probablities to predicted class
_, pred_tensor = torch.max(output, 1)
preds = np.squeeze(pred_tensor.numpy()) if not train_on_gpu else np.squeeze(pred_tensor.cpu().numpy())

# plot the images in the batch along with the predicted labels and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title('{} ({})'.format(classes[preds[idx]], classes[labels[idx]]),
                                  color=('green' if preds[idx]==labels[idx].item() else 'red'))
plt.show()