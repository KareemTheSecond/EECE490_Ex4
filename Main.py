import torch
from Train_Evaluate import Train, evaluate_model
from model import MyDataSet, CNN_FF
import torchvision.transforms as transforms

directory_train = r"TRAIN_FOLDER_PATH"
directory_val = r"VALIDATION_FOLDER_PATH"
directory_test = r"TEST_FOLDER_PATH"
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

dataset_train = MyDataSet(directory_train, transform=transform)
dataset_val = MyDataSet(directory_val, transform=transform)
dataset_test = MyDataSet(directory_test, transform=transform)

model = CNN_FF()  
Train(directory_train, directory_val, directory_test) 
val_accuracy = evaluate_model(DataLoader(dataset_val, batch_size=32, shuffle=False))
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

test_accuracy = evaluate_model(DataLoader(dataset_test, batch_size=32, shuffle=False))
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
