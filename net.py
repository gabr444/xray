import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional Neural Network. Used for images.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Set the first conv2d layer to a channel size of 1 (gray-scale) and the kernel size to 5 (5*5 is the amount of pixels it will process at a time)
        self.c1 = nn.Conv2d(1, 8, 5)
        # Use maxpool to reduce amount of pixels. Maxpool takes the max value of h*w pixels given and puts that one specific pixel into the new image.
        self.pool = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(8, 16, 5)
        self.c3 = nn.Conv2d(16, 32, 5)
        self.c4 = nn.Conv2d(32, 64, 3)
        self.c5 = nn.Conv2d(64, 128, 3)
        # Input size of the first linear layer. Calculated from image in last cnn layer. Channels*height*width.
        self.flattenedSize = 128*111*86
        self.fc1 = nn.Linear(self.flattenedSize, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.pool(x)
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = F.relu(self.c5(x))
        # Flatten the image. Convert 2d to 1d to be used in linear layer.
        x = x.view(-1, self.flattenedSize)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
