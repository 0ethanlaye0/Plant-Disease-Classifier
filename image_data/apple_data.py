"""Builds datasets and dataloaders for apple leaf images"""

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

train_transform = transforms.Compose([
    transforms.Resize(size = (128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees = 15),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(size = (128, 128)),
    transforms.ToTensor()
])


train_data = ImageFolder(root = "images/Apple/Train", transform = train_transform)
test_data = ImageFolder(root = "images/Apple/Test", transform = test_transform)

class_names = train_data.classes

train_batches = DataLoader(dataset = train_data,
                           batch_size = 32,
                           shuffle = True)

test_batches = DataLoader(dataset = test_data,
                           batch_size = 32,
                           shuffle = False)

