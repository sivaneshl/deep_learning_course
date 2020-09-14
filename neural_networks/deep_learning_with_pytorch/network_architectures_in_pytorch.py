from torch import nn
import torch.nn.functional as F

# Create a network with 784 input units, a hidden layer with 128 units and a ReLU activation, then a hidden layer
# with 64 units and a ReLU activation, and finally an output layer with a softmax activation as shown above. You can
# use a ReLU activation with the nn.ReLU module or F.relu function.

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.h1 = nn.Linear(784, 128)
        self.h2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.softmax(self.out(x))
        return x