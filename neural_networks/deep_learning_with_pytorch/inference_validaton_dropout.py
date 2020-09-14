import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download and load the training data
trainset = datasets.FashionMNIST('F_MNIST_data\\', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('F_MNIST_data\\', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Flatten the input
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        steps += 1
        optimizer.zero_grad()
        # Get the class probablities
        log_probablities = model(images)
        loss = criterion(log_probablities, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    else:
        model.eval()

        test_loss = 0
        accuracy = 0

        with torch.no_grad():
            for images, labels in testloader:
                log_probablities = model(images)
                loss = criterion(log_probablities, labels)
                test_loss += loss

                probablities = torch.exp(log_probablities)

                top_p, top_class = probablities.topk(1, dim=1)
                # print(top_class[:10, :])
                equals = top_class == labels.view(*top_class.shape)
                # print(equals)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

        print('Epoch {}/{}'.format(e + 1, epochs),
              'Training Loss: {:.3f}'.format(running_loss / len(trainloader)),
              'Test Loss: {:.3f}'.format(test_loss / len(testloader)),
              'Accuracy: {:.3f}'.format(accuracy / len(testloader)))

        running_loss = 0
        model.train()

plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.legend(frameon=False)
plt.show()

# Import helper module (should be in the repo)
import helper

# Test out your network!

model.eval()

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.view(1, 784)

# Calculate the class probabilities (softmax) for img
with torch.no_grad():
    output = model.forward(img)

ps = torch.exp(output)

# Plot the image and probabilities
helper.view_classify(img.view(1, 28, 28), ps, version='Fashion')