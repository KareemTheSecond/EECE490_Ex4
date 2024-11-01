import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


class MyDataSet(Dataset): 
    def __init__(self, directoryOfDataset, transform=None): 
        self.data = ImageFolder(directoryOfDataset, transform=transform)
        
    def __getitem__(self, index): 
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class NetClassifier(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 * 128 * 128 => conv1 :6 * 124 * 124 => pool: 6 * 62 * 62 => conv2: 16*58*58 => pool2: 16*29*29 
        self.pool = nn.MaxPool2d(2, 2) # 
        self.conv2 = nn.Conv2d(6, 16, 5) #
        self.fc1 = nn.Linear(16*29*29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 53)

    def forward(self, x): 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
