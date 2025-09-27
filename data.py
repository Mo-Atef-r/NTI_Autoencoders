import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import yaml

# ----- Hyperparameters -----
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

IMAGE_SIZE = config['dataset']['image_size']
CHANNELS = config['dataset']['channels']
VALIDATION_SPLIT = config['dataset']['validation_split']

EMBEDDING_DIM = config['model']['embedding_dim']

BATCH_SIZE = config['training']['batch_size']

# ----- Data Preparation -----

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,))  # Normalize to mean=0.5, std=0.5
])

full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

full_train_size = len(full_train_dataset)
val_size = int(VALIDATION_SPLIT * full_train_size)
train_size = full_train_size - val_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# DataLoaders 
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
