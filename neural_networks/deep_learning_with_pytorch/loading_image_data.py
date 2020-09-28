import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

import helper

data_dir = 'C:\\ai_programming_with_python\\neural_networks\\deep_learning_with_pytorch\\Cat_Dog_data\\train'
transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Run this to test your data loader
images, labels = next(iter(dataloader))
helper.imshow(images[0], normalize=False)
plt.show()