import torch
from Train_Evaluate import Train_CNN_FF, evaluate_model
from Models import MyDataSet, CNN_FF
import torchvision.transforms as transforms

directory_train = r"TRAIN_FOLDER_PATH"
directory_val = r"VALIDATION_FOLDER_PATH"
directory_test = r"TEST_FOLDER_PATH"

model, dataloader_val, dataloader_test  = Train_CNN_FF(directory_train, directory_val, directory_test) 
val_accuracy = evaluate_model(model, dataloader_val)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

test_accuracy = evaluate_model(model, dataloader_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


