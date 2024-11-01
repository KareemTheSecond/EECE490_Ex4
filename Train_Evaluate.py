import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from model import MyDataSet, CNN_FF



def Train(directory_train,directory_val,directory_test): 
  transform = transforms.Compose([transforms.Resize((128, 128)),  transforms.ToTensor()])
  dataset_train = MyDataSet(directory_train, transform=transform)
  dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

  
  dataset_val = MyDataSet(directory_val, transform=transform)
  dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)
  dataset_test = MyDataSet(directory_test, transform=transform)
  dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)


  model = CNN_FF()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.1)
  epochs = 10
  for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in dataloader_train:
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader_train):.4f}')


def evaluate_model(loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total









