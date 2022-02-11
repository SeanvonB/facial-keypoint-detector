# Created: March, 2020 by Sean von Bayern
# Updated:

#Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):

    # Define layers
    def __init__(self):
        super(Net, self).__init__()
        
        # Convolutional layers
        # Conv1 output after pooling: 32 filters of 110x110 size (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        # Conv 2 output after pooling: 64 filters of 53x53 size (64, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 5)
        # Conv 3 output after pooling: 128 filters of 25x25 size (128, 25, 25)
        self.conv3 = nn.Conv2d(64, 128, 3)
        # Conv 4 output after pooling: 256 filters of 11x11 size (256, 11, 11)
        self.conv4 = nn.Conv2d(128, 256, 3)
       
        # Fully-connected layers
        self.fc1 = nn.Linear(256 * 11 * 11, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 136)
        
        # Dropout layers
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)     
    
    # Put layers in order
    def forward(self, x):
        
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop4(x)
        
        # Flatten for linear
        x = x.view(x.size(0), -1)
        
        # Fully-connected layers
        x = F.relu(self.fc1(x))
        x = self.drop5(x)
        x = F.relu(self.fc2(x))
        x = self.drop5(x)
        x = self.fc3(x)
        
        return x
