import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler

model.load_state_dict(torch.load('mnist_model.pth'))

## Testing the network
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


