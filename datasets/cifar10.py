import os

from torch.utils.data import DataLoader
from datasets.transforms import train_transform, valid_transform
from datasets.settings import *
from torchvision.datasets import CIFAR10

ROOT = os.path.join(DATA_DIR, "CIFAR10")
train_data = CIFAR10(root="./data", train=True, transform=train_transform, download=True)
valid_data = CIFAR10(root="./data", train=False, transform=valid_transform, download=True)
train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_data, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=2)
