import matplotlib.pyplot as plt
from collections import OrderedDict
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

data_dir = 'C:\\ai_programming_with_python\\neural_networks\\deep_learning_with_pytorch\\Cat_Dog_data'

train_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.229, 0.224, 0.225], [0.485, 0.456, 0.406])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.229, 0.224, 0.225], [0.485, 0.456, 0.406])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(nn.Linear(2048, 512),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(512, 2),
                           nn.LogSoftmax(dim=1))
model.fc = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

model.to(device)

epochs = 1
steps = 0
running_loss = 0
print_every = 5

for e in range(epochs):
    for images, labels in trainloader:
        steps += 1

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval()

            test_loss = 0
            accuracy = 0

            for images, labels in testloader:

                images, labels = images.to(device), labels.to(device)

                log_ps = model(images)
                loss = criterion(log_ps, labels)
                test_loss += loss.item()

                ps = torch.exp(log_ps)
                top_ps, top_class = ps.topk(1, dim=1)
                equality = top_class == labels.views(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor))

            print('Epoch {}/{}'.format(e + 1, epochs),
                  'Training Loss: {:.3f}'.format(running_loss / len(print_every)),
                  'Test Loss: {:.3f}'.format(test_loss / len(testloader)),
                  'Accuracy: {:.3f}'.format(accuracy / len(testloader)))

            running_loss = 0
            model.train()


