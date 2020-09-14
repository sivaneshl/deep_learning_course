import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import ImageFile

def imshow(img):
    img_numpy = img.numpy()
    plt.imshow(np.transpose(img_numpy, (1, 2, 0)))

data_dir = 'C:\\deep_learning\\deep_learning_course\\convolutional_neural_networks\\dog_breed_classifier\\data\\dog_images\\'
batch_size = 20

# check if cuda is available
train_on_gpu = torch.cuda.is_available()

train_transform  = transforms.Compose([transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomRotation(10),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


train_dataset = datasets.ImageFolder(data_dir + 'train', transform=train_transform)
valid_dataset = datasets.ImageFolder(data_dir + 'valid', transform=test_transform)
test_dataset = datasets.ImageFolder(data_dir + 'test', transform=test_transform)

train_sampler = SubsetRandomSampler(list(range(len(train_dataset))))
valid_sampler = SubsetRandomSampler(list(range(len(valid_dataset))))
test_sampler = SubsetRandomSampler(list(range(len(test_dataset))))

loaders = {}
loaders['train'] = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
loaders['valid'] = torch.utils.data.DataLoader(valid_dataset, sampler=valid_sampler, batch_size=batch_size)
loaders['test'] = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

dataiter = iter(loaders['train'])
images, labels = next(dataiter)
# images = images.numpy()

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

# plot the images in the batch along with their labels
fig = plt.figure(figsize=(25, 4))
for idx in range(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    unnormalized_image = inv_normalize(images[idx])
    plt.imshow(np.transpose(unnormalized_image.numpy(), (1, 2, 0)))
plt.show()

## Define the CNN Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolution layer: Increase the depth from 3 ==> 16
        self.conv_1 = nn.Conv2d(3, 16, 3, padding=1)
        # Convolution layer 2: Increase the depth from 16 ==> 32
        self.conv_2 = nn.Conv2d(16, 32, 3, padding=1)
        # Convolution layer 3: Increase the depth from 32 ==> 64
        self.conv_3 = nn.Conv2d(32, 64, 3, padding=1)
        # Convolution layer 4: Increase the depth from 64 ==> 128
        self.conv_4 = nn.Conv2d(64, 128, 3, padding=1)
        # Convolution layer 5: Increase the depth from 128 ==> 256
        self.conv_5 = nn.Conv2d(128, 256, 3, padding=1)

        # Max pooling layer: Downsize the image by half
        self.pool = nn.MaxPool2d(2, 2)

        # Linear layer 1: 256 X 7 X 7 ==> 500
        self.fc_1 = nn.Linear(256*7*7, 500)
        # Linear layer 2: 500 ==> 133
        self.fc_2 = nn.Linear(500, 133)

        # Dropout
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv_1(x)))   # 3 X 224 X 224  ==> 16 X 224 X 224 ==> 16 X 112 X 112
        x = self.pool(F.relu(self.conv_2(x)))   # 16 X 112 X 112 ==> 32 X 112 X 112 ==> 32 X 56 X 56
        x = self.pool(F.relu(self.conv_3(x)))   # 32 X 56 X 56   ==> 64 X 56 X 56   ==> 64 X 28 X 28
        x = self.pool(F.relu(self.conv_4(x)))   # 64 X 28 X 28   ==> 128 X 28 X 28  ==> 128 X 14 X 14
        x = self.pool(F.relu(self.conv_5(x)))   # 128 X 14 X 14  ==> 256 X 14 X 14  ==> 256 X 7 X 7
        x = self.dropout(x)                     # Dropout of 25%
        x = x.view(-1, 256 * 7 * 7)             # Flatten the image to 256 X 7 X 7 = 12544
        x = self.dropout(F.relu(self.fc_1(x)))   # 12544 ==> 500
        x = self.fc_2(x)                         # 500   ==> 133
        return x

# instantiate the model
model = Net()
if train_on_gpu:
    model.cuda()

print(model)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Train method implementation
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """
    Returns a trained model
    """

    # initialize the tracker for minimum validation loss
    valid_loss_min = np.Inf

    for e in range(1, n_epochs+1):
        # initialize variables to track training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # train the model
        model.train()
        for data, target in loaders['train']:
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # clear the gradients for optimized variables
            optimizer.zero_grad()
            # forward pass
            output = model(data)
            # calculate loss
            loss = criterion(output, target)
            # backward pass
            loss.backward()
            # optimizer step
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        # validate the model
        model.eval()
        for data, target in loaders['valid']:
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # forward pass
            output = model(data)
            # calculate loss
            loss = criterion(output, target)
            # update loss
            valid_loss += loss.item() * data.size(0)

        # calculate the average losses
        train_loss = train_loss / len(loaders['train'].dataset)
        valid_loss = valid_loss / len(loaders['valid'].dataset)

        # print the training and validation loss statistics
        print('Epoch {}, \t Training loss {}, \t Validation loss {} '.format(e, train_loss, valid_loss))

        # save the model if the validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

    # return the trained model
    return model

# Test method implementation
def test(loaders, model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.0
    correct = 0
    total = 0

    model.eval()
    for data, target in loaders['test']:
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        loss = criterion(output, target)
        test_loss = loss.item() * data.size(0)
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}'.format(test_loss))
    print('Test Accuracy %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))

# train the model
model = train(10, loaders, model, optimizer, criterion, train_on_gpu, 'model_scratch.pth')

# load the model that got the best validation accuracy
model.load_state_dict(torch.load('model_scratch.pth'))

# call test function
test(loaders, model, criterion, train_on_gpu)
