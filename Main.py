import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from Train_Evaluate import Train,evaluate_model 


directory_train = r"TRAIN_FOLDER_PATH"
directory_val = r"VALIDATION_FOLDER_PATH"
directory_test = r"TEST_FOLDER_PATH"

Train(directory_train,directory_val,directory_test)



