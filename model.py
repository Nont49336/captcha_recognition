import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


# Define a CNN model


class CharacterCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for the first conv layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization for the second conv layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch normalization for the third conv layer
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Adjusted for third conv layer output size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)  # Output layer
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))  # Conv1 + BN + LeakyReLU + Pool
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))  # Conv2 + BN + LeakyReLU + Pool
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))  # Conv3 + BN + LeakyReLU + Pool
        x = x.view(-1, 128 * 3 * 3)  # Flatten
        x = F.leaky_relu(self.fc1(x))  # Fully connected layer 1 with LeakyReLU
        x = self.dropout(x)  # Dropout for regularization
        x = F.leaky_relu(self.fc2(x))  # Fully connected layer 2 with LeakyReLU
        x = self.fc3(x)  # Output layer
        return x

class CharacterCNN_MNIST(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for the first conv layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization for the second conv layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch normalization for the third conv layer
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Adjusted for third conv layer output size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)  # Output layer
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))  # Conv1 + BN + LeakyReLU + Pool
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))  # Conv2 + BN + LeakyReLU + Pool
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))  # Conv3 + BN + LeakyReLU + Pool
        x = x.view(-1, 128 * 3 * 3)  # Flatten
        x = F.leaky_relu(self.fc1(x))  # Fully connected layer 1 with LeakyReLU
        x = self.dropout(x)  # Dropout for regularization
        x = F.leaky_relu(self.fc2(x))  # Fully connected layer 2 with LeakyReLU
        x = self.fc3(x)  # Output layer
        return x

def get_num_clases(dataset_path):
    num_classes = len(datasets.ImageFolder(dataset_path).classes) 
    return num_classes

def get_classes(dataset_pth):
    return datasets.ImageFolder(dataset_pth).classes